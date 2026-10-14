#!/usr/bin/env python3
"""Automatic disk swap tier, end to end through llama-server.

Runs the real server binary against a generated tiny model (no downloads, no GPU)
and checks the whole path: argument gating, the identity-keyed directory, an
actual spill under RAM pressure, a restore served from disk, and cleanup.

Usage:
    test-cache-disk-e2e.py --server <llama-server> --model <tiny model.gguf>
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
        "-c", "4096", "-b", "512", "-ub", "128",
        "--parallel", "1", "--port", str(port), "--no-warmup", "-lv", "5",
    ] + extra
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, stdout=log, stderr=log, env=dict(os.environ))


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def swap_files(directory: Path) -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(str(directory / "pcd-*")))


def test_gating(server: Path, model: Path, workdir: Path) -> None:
    """--cache-disk requires the RAM prompt cache; without it the tier is refused."""
    swap_dir = workdir / "swap-gating"
    swap_dir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "gating.log"
    port = free_port()

    process = start_server(server, model, port, log_path, ["--cache-ram", "0", "--cache-disk", str(swap_dir)])
    try:
        wait_ready(process, f"http://127.0.0.1:{port}", log_path)
    finally:
        stop_server(process)

    log = log_path.read_text(errors="replace")
    assert "--cache-disk requires --cache-ram" in log, "the tier must refuse to run without --cache-ram"
    assert "disk prompt cache enabled" not in log, "the tier must not be enabled without --cache-ram"
    assert swap_files(swap_dir) == [], "a refused tier must not write files"


def distinct_prompts(n: int) -> list[str]:
    random.seed(7)
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
        "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    ]
    prompts = []
    for i in range(n):
        random.shuffle(words)
        rotated = words[i % len(words):] + words[:i % len(words)]
        prompts.append(" ".join(rotated) * 3 + f" end{i}")
    return prompts


def test_swap_round_trip(server: Path, model: Path, workdir: Path) -> None:
    """Under RAM pressure an entry spills, and a later request is restored from disk."""
    swap_dir = workdir / "swap-roundtrip"
    swap_dir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "roundtrip.log"
    port = free_port()
    url = f"http://127.0.0.1:{port}"

    process = start_server(server, model, port, log_path, [
        "--cache-ram", "1",                 # 1 MiB: a few tiny-model entries overflow it
        "--cache-disk", str(swap_dir),
        "--cache-idle-slots",
    ])
    try:
        wait_ready(process, url, log_path)

        log = log_path.read_text(errors="replace")
        assert "disk prompt cache enabled" in log, "the tier must report that it is enabled"

        prompts = distinct_prompts(9)

        cold_prompt_n = None
        for i, prompt in enumerate(prompts):
            timings = post(url, "/completion", {
                "prompt": prompt, "n_predict": 1, "temperature": 0, "cache_prompt": True,
            }).get("timings", {})
            if i == 0:
                cold_prompt_n = timings.get("prompt_n")
        assert cold_prompt_n and cold_prompt_n > 8, f"unexpected cold prompt size: {cold_prompt_n}"

        log = log_path.read_text(errors="replace")
        assert "moving oldest RAM entry to disk" in log, "RAM pressure must spill an entry to disk"
        spilled = swap_files(swap_dir)
        assert spilled, "a spill must leave a published swap file"

        # the first prompt is the oldest entry, so it is the one most likely spilled
        timings = post(url, "/completion", {
            "prompt": prompts[0], "n_predict": 1, "temperature": 0, "cache_prompt": True,
        }).get("timings", {})

        assert timings.get("cache_source") == "disk", (
            f"expected a disk restore, got source={timings.get('cache_source')!r} "
            f"reason={timings.get('cache_reason')!r}; swap files: {swap_files(swap_dir)}")
        assert timings.get("prompt_n", cold_prompt_n) < cold_prompt_n, (
            "a disk restore must skip prompt processing")
    finally:
        stop_server(process)

    assert swap_files(swap_dir) == [], "shutdown must clean the swap directory"


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

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="beellama-cache-disk-"))
    try:
        test_gating(args.server, args.model, workdir)
        print("cache disk e2e: argument gating OK")
        test_swap_round_trip(args.server, args.model, workdir)
        print("cache disk e2e: spill and disk restore OK")
    finally:
        if args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)

    print("cache disk e2e passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
