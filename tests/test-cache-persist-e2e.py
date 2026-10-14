#!/usr/bin/env python3
"""Persistent prompt-cache tier, end to end through llama-server.

Runs the real server binary against a generated tiny model (no downloads, no
GPU) and checks the whole restart path: argument gating, a clean-shutdown
persist with swap-file promotion, a rehydrated restore served from the
persisted state after a restart, and a same-name/different-content model whose
persisted cache is refused and deleted.

Usage:
    test-cache-persist-e2e.py --server <llama-server> --model <tiny model.gguf>
"""

import argparse
import glob
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(process: subprocess.Popen, url: str, log_path: Path, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early ({process.returncode}); log:\n{log_path.read_text(errors='replace')}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"server readiness timeout; log:\n{log_path.read_text(errors='replace')}")


def post(url: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def start_server(server: Path, model: Path, port: int, log_path: Path, extra: list[str]) -> subprocess.Popen:
    command = [
        str(server), "-m", str(model),
        "-c", "256", "-b", "128", "-ub", "64",
        "--parallel", "1", "--port", str(port), "--no-warmup", "-lv", "5",
    ] + extra
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=log, stderr=log, env=dict(os.environ))


def stop_server(process: subprocess.Popen) -> None:
    # a clean shutdown: this is what triggers the persist save
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        raise AssertionError("the server did not exit on SIGTERM; the save must run on a clean shutdown")


def persist_files(directory: Path) -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(str(directory / "pcp-*")))


def distinct_prompts(n: int, n_words: int = 5) -> list[str]:
    # the generated tiny models train on 128 tokens, so keep prompts short and
    # distinct enough to fill separate cache entries
    random.seed(11)
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
        "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    ]
    prompts = []
    for i in range(n):
        random.shuffle(words)
        rotated = words[i % 10:] + words[:i % 10]
        prompts.append(" ".join(rotated * n_words) + f" end{i}")
    return prompts


def test_gating(server: Path, model: Path, workdir: Path) -> None:
    """--cache-persist requires the RAM prompt cache; without it the tier is refused."""
    persist_dir = workdir / "persist-gating"
    persist_dir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "gating.log"
    port = free_port()

    process = start_server(server, model, port, log_path, ["--cache-ram", "0", "--cache-persist", str(persist_dir)])
    try:
        wait_ready(process, f"http://127.0.0.1:{port}", log_path)
    finally:
        stop_server(process)

    log = log_path.read_text(errors="replace")
    assert "--cache-persist requires --cache-ram" in log, "the tier must refuse to run without --cache-ram"
    assert "persistent prompt cache enabled" not in log, "the tier must not be enabled without --cache-ram"
    assert persist_files(persist_dir) == [], "a refused tier must not write files"


def test_persist_round_trip(server: Path, model: Path, workdir: Path) -> None:
    """A clean shutdown persists; a restart restores, including promoted spills."""
    persist_dir = workdir / "persist-roundtrip"
    swap_dir = workdir / "swap-roundtrip"
    persist_dir.mkdir(parents=True, exist_ok=True)
    swap_dir.mkdir(parents=True, exist_ok=True)
    port_1 = free_port()
    port_2 = free_port()
    url_1 = f"http://127.0.0.1:{port_1}"
    url_2 = f"http://127.0.0.1:{port_2}"

    prompts = distinct_prompts(10)

    # --- run 1: fill the cache past its RAM budget so entries spill, then re-
    # visit the first prompt so its (spilled) entry is confirmed restorable, and
    # shut down cleanly
    log_1 = workdir / "roundtrip-run1.log"
    process = start_server(server, model, port_1, log_1, [
        "--cache-ram", "1",                 # 1 MiB: tiny-model entries overflow it
        "--cache-disk", str(swap_dir),
        "--cache-persist", str(persist_dir),
    ])
    try:
        wait_ready(process, url_1, log_1)

        for prompt in prompts:
            post(url_1, "/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": True})

        timings = post(url_1, "/completion", {
            "prompt": prompts[0], "n_predict": 1, "temperature": 0, "cache_prompt": True,
        }).get("timings", {})
        assert timings.get("prompt_n", 10**9) < 10, (
            f"a mid-session re-visit must reuse the cache, got prompt_n={timings.get('prompt_n')!r}")

        log = log_1.read_text(errors="replace")
        spilled_swap = sorted(os.path.basename(p) for p in glob.glob(str(swap_dir / "pcd-*")))
        assert "moving oldest RAM entry to disk" in log and spilled_swap, (
            "RAM pressure must spill an entry to the session swap tier before shutdown")
    finally:
        stop_server(process)

    # --- clean shutdown must have emptied the swap tier into the persist tier
    assert not glob.glob(str(swap_dir / "pcd-*")), "promoted entries must not stay in the swap directory"
    persisted = persist_files(persist_dir)
    assert persisted, "a clean shutdown must persist cache entries"
    assert (persist_dir / "index.bin").is_file(), "the sidecar index must be published"
    assert not glob.glob(str(persist_dir / "*.tmp")), "no staging file may survive the publish"

    log = log_1.read_text(errors="replace")
    marker = "persisted prompt cache saved: "
    n_saved = int(log.rstrip().rsplit(marker, 1)[1].split(" ", 1)[0])
    assert n_saved == len(persisted), (
        f"the index must reference exactly the published files: saved {n_saved}, files {persisted}")

    # --- run 2: the rehydrated cache must serve the same prompt from disk,
    # including the entry that was promoted from the swap tier
    log_2 = workdir / "roundtrip-run2.log"
    process = start_server(server, model, port_2, log_2, [
        "--cache-ram", "1",
        "--cache-disk", str(swap_dir),
        "--cache-persist", str(persist_dir),
    ])
    try:
        wait_ready(process, url_2, log_2)
        log = log_2.read_text(errors="replace")
        assert "persisted prompt cache loaded" in log, "startup must report the rehydration"
        marker = "persisted prompt cache loaded: "
        n_loaded = int(log.rstrip().rsplit(marker, 1)[1].split(" ", 1)[0])
        assert n_loaded == n_saved, f"every persisted entry must be adopted (loaded {n_loaded}/{n_saved})"

        timings = post(url_2, "/completion", {
            "prompt": prompts[0], "n_predict": 1, "temperature": 0, "cache_prompt": True,
        }).get("timings", {})
        assert timings.get("cache_source") == "disk", (
            f"a persisted restore must be served from the cached state, got {timings.get('cache_source')!r}")
        assert timings.get("prompt_n", 10**9) < 10, (
            f"a persisted restore must skip prompt processing, got prompt_n={timings.get('prompt_n')!r}")
    finally:
        stop_server(process)


def test_wrong_model_refused(server: Path, model: Path, workdir: Path) -> None:
    """A same-name model with different content must not get the old cache."""
    persist_dir = workdir / "persist-finetune"
    persist_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    log_path = workdir / "finetune.log"

    prompts = distinct_prompts(3)

    # run 1: persist an entry
    process = start_server(server, model, port, log_path, [
        "--cache-ram", "64", "--cache-persist", str(persist_dir),
    ])
    try:
        wait_ready(process, url, log_path)
        post(url, "/completion", {"prompt": prompts[0], "n_predict": 1, "temperature": 0, "cache_prompt": True})
        post(url, "/completion", {"prompt": prompts[1], "n_predict": 1, "temperature": 0, "cache_prompt": True})
        post(url, "/completion", {"prompt": prompts[2], "n_predict": 1, "temperature": 0, "cache_prompt": True})
    finally:
        stop_server(process)

    assert persist_files(persist_dir), "run 1 must persist an entry"

    # run 2: same name, different content. Copying already changes the mtime,
    # and a byte flip near the file tail (tensor data, sampled by the fixed
    # last slice) changes the content hash as well.
    fine_tune = workdir / "fine-tune.gguf"
    shutil.copyfile(model, fine_tune)
    with open(fine_tune, "r+b") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(size - 4096)
        byte = f.read(1)
        f.seek(size - 4096)
        f.write(bytes([byte[0] ^ 0xFF]))

    port_2 = free_port()
    url_2 = f"http://127.0.0.1:{port_2}"

    process = start_server(server, fine_tune, port_2, log_path, [
        "--cache-ram", "64", "--cache-persist", str(persist_dir),
    ])
    try:
        wait_ready(process, url_2, log_path)
        log = log_path.read_text(errors="replace")
        assert "persisted prompt cache loaded: 0 entries" in log, (
            "a different model file must not adopt the persisted cache")
        timings = post(url_2, "/completion", {
            "prompt": prompts[0], "n_predict": 1, "temperature": 0, "cache_prompt": True,
        }).get("timings", {})
        assert timings.get("cache_source") != "disk", (
            "a different model file must not serve persisted state")
        assert timings.get("prompt_n", 0) > 10, (
            f"the request must be processed cold, got prompt_n={timings.get('prompt_n')!r}")
    finally:
        stop_server(process)

    assert not persist_files(persist_dir), "refused persisted entries must be deleted, not kept"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()

    if not args.server.is_file():
        print(f"server binary not found: {args.server}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="beellama-cache-persist-"))
    try:
        test_gating(args.server, args.model, workdir)
        test_persist_round_trip(args.server, args.model, workdir)
        test_wrong_model_refused(args.server, args.model, workdir)
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1
    finally:
        if not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    print("cache-persist e2e passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())