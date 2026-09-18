#!/usr/bin/env python3
"""Behavioural test for llama-embedding-dump's per-layer K/V dump.

The dump exists so that a shared-KV draft head (for example the Gemma 4 MTP
assistant) can be trained against the attention inputs of a frozen target, so
the recorded rows have to be exactly what the target's KV cache holds, one row
per token, in token order.

The checks below are self-contained and need no reference implementation:

* chunk invariance - the rows must not depend on how the sequence was split
  into batches. This pins down both the per-chunk accumulation and the row
  order, and it is what would break first if rows were permuted or overwritten.
* rope signature - within layer 0 the attention input of a token is just its own
  embedding, so two positions carrying the same token id share an input. A
  post-rope K must therefore differ between them, while V, which is never
  rotated, must be bit-identical. That is what distinguishes the cache
  representation from the raw projections.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    # the dump is verified numerically, so there is nothing to check without it.
    # 77 is registered as a skip code in CMakeLists.txt
    print("test-embedding-dump-kv: numpy is not available, skipping")
    sys.exit(77)


CORPUS = [
    {"messages": [{"role": "user", "content": "the the the the the little cat sat on the the mat"}]},
    {"messages": [{"role": "user", "content": "a short story about a dog and a cat in a big green park"}]},
]


def run_dump(dump, model, corpus, outdir, extra):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(dump), "-m", str(model), "--jsonl", str(corpus), "--output-dir", str(outdir),
           "--no-dump-hidden", "-c", "256", "-t", "2", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    return proc


def dump_ok(dump, model, corpus, outdir, extra):
    proc = run_dump(dump, model, corpus, outdir, extra)
    assert proc.returncode == 0, f"dump failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"

    files = sorted(outdir.glob("*.npz"))
    assert files, f"no .npz files were written:\n{proc.stdout}\n{proc.stderr}"

    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="path to the llama-embedding-dump binary")
    ap.add_argument("--model", required=True, help="path to the test model")
    ap.add_argument("--workdir", required=True, help="scratch directory for the dumps")
    args = ap.parse_args()

    dump = Path(args.dump)
    model = Path(args.model)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    assert dump.is_file(), f"missing dump binary: {dump}"
    assert model.is_file(), f"missing model: {model}"

    corpus = work / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(d) + "\n" for d in CORPUS), encoding="utf-8")

    # --- one layer, one batch -------------------------------------------------
    # f32 for the reference runs, so the comparisons below are on exact values
    big = dump_ok(dump, model, corpus, work / "big",
                  ["--dump-kv-layers", "0", "-b", "64", "--dump-dtype", "f32"])
    ref = np.load(big[0])

    for key in ("k_l0", "v_l0", "kv_layers", "input_ids", "labels"):
        assert key in ref.files, f"missing '{key}' in {big[0].name}, got {sorted(ref.files)}"

    assert np.array_equal(ref["kv_layers"], np.array([0], dtype=np.int32)), (
        f"kv_layers must record the selected layer, got {ref['kv_layers']}"
    )

    ids = ref["input_ids"]
    n_tok = len(ids)

    assert ref["k_l0"].shape == (n_tok, ref["k_l0"].shape[1]), "K rows must be one per token"
    assert ref["v_l0"].shape == (n_tok, ref["v_l0"].shape[1]), "V rows must be one per token"
    assert ref["k_l0"].shape[1] == ref["v_l0"].shape[1], "K and V must have the same row width"
    assert ref["k_l0"].shape[1] > 0, "K row width must be non-zero"
    assert np.isfinite(ref["k_l0"]).all() and np.isfinite(ref["v_l0"]).all(), "dump must be finite"
    assert np.abs(ref["k_l0"]).max() > 0 and np.abs(ref["v_l0"]).max() > 0, "dump must not be all zero"

    # labels are the next-token targets
    assert np.array_equal(ref["labels"][:-1], ids[1:]), "labels must be input_ids shifted by one"

    # --- chunk invariance ----------------------------------------------------
    # the same document split into much smaller batches must produce bit-identical rows
    small = dump_ok(dump, model, corpus, work / "small",
                    ["--dump-kv-layers", "0", "-b", "8", "--dump-dtype", "f32"])
    chunked = np.load(small[0])

    for key in ("k_l0", "v_l0", "input_ids", "labels"):
        assert np.array_equal(chunked[key], ref[key]), (
            f"'{key}' changed with the batch split; rows are not stable across chunks"
        )

    # --- rope signature ------------------------------------------------------
    # positions sharing a token id have the same layer-0 input
    pairs = [(i, j) for i in range(n_tok) for j in range(i + 1, n_tok) if ids[i] == ids[j]]
    assert pairs, "test corpus must contain a repeated token to exercise the rope check"

    for i, j in pairs:
        assert not np.array_equal(ref["k_l0"][i], ref["k_l0"][j]), (
            f"K is identical for token {ids[i]} at positions {i} and {j}; "
            "the dump looks pre-rope, but the cache stores post-rope K"
        )
        assert np.array_equal(ref["v_l0"][i], ref["v_l0"][j]), (
            f"V differs for token {ids[i]} at positions {i} and {j}; "
            "V must not be position dependent"
        )

    # --- layer selection -----------------------------------------------------
    two = dump_ok(dump, model, corpus, work / "two", ["--dump-kv-layers", "0,2", "-b", "64"])
    sel = np.load(two[0])

    assert np.array_equal(sel["kv_layers"], np.array([0, 2], dtype=np.int32)), (
        f"kv_layers must be the sorted selection, got {sel['kv_layers']}"
    )
    for key in ("k_l0", "v_l0", "k_l2", "v_l2"):
        assert key in sel.files, f"missing '{key}' for a two-layer selection, got {sorted(sel.files)}"
    assert not np.array_equal(sel["k_l0"], sel["k_l2"]), "different layers must not dump the same K"

    # layers are de-duplicated and sorted
    dup = dump_ok(dump, model, corpus, work / "dup", ["--dump-kv-layers", "2,0,2", "-b", "64"])
    assert np.array_equal(np.load(dup[0])["kv_layers"], np.array([0, 2], dtype=np.int32)), (
        "a repeated layer index must be de-duplicated"
    )

    # --- dtype ---------------------------------------------------------------
    # f16 is the default - the dump is the bulk of the disk cost
    default = np.load(dump_ok(dump, model, corpus, work / "default", ["--dump-kv-layers", "0", "-b", "64"])[0])
    assert default["k_l0"].dtype == np.float16, "the default dump dtype should be f16"

    half = dump_ok(dump, model, corpus, work / "f16",
                   ["--dump-kv-layers", "0", "-b", "64", "--dump-dtype", "f16"])
    h = np.load(half[0])

    assert h["k_l0"].dtype == np.float16, f"expected f16 output, got {h['k_l0'].dtype}"
    assert h["k_l0"].shape == ref["k_l0"].shape, "the dtype must not change the shape"
    assert np.abs(h["k_l0"].astype(np.float64) - ref["k_l0"].astype(np.float64)).max() < 5e-2, (
        "f16 output must stay close to the f32 output"
    )

    # q8_0 uses ggml's block format and is written as an int8 payload plus the
    # per-32-value f16 scales
    q = np.load(dump_ok(dump, model, corpus, work / "q8_0",
                        ["--dump-kv-layers", "0", "-b", "64", "--dump-dtype", "q8_0"])[0])

    assert q["k_l0"].dtype == np.int8, f"expected an int8 payload, got {q['k_l0'].dtype}"
    assert q["k_l0_scales"].dtype == np.float16, f"expected f16 scales, got {q['k_l0_scales'].dtype}"

    width = ref["k_l0"].shape[1]
    assert q["k_l0"].shape == (n_tok, width), "the q8_0 payload must keep the shape"
    assert q["k_l0_scales"].shape == (n_tok, width // 32), "q8_0 needs one scale per 32 values"

    deq = q["k_l0"].astype(np.float64) * np.repeat(q["k_l0_scales"].astype(np.float64), 32, axis=1)
    amax = np.abs(ref["k_l0"]).max()
    assert np.abs(deq - ref["k_l0"].astype(np.float64)).max() < amax / 100.0, (
        "dequantized q8_0 must track the f32 output"
    )

    # --- resume ---------------------------------------------------------------
    # a long dump has to be restartable, so a re-run with --skip-existing must
    # leave the documents that are already on disk untouched
    resume = work / "resume"
    before = {f.name: (f.stat().st_mtime_ns, f.stat().st_size)
              for f in dump_ok(dump, model, corpus, resume, ["--dump-kv-layers", "0", "-b", "64"])}

    proc = run_dump(dump, model, corpus, resume,
                    ["--dump-kv-layers", "0", "-b", "64", "--skip-existing"])
    assert proc.returncode == 0, f"resume run failed:\n{proc.stdout}\n{proc.stderr}"
    assert "skipped" in proc.stdout + proc.stderr, "a resume run must report what it skipped"

    after = {f.name: (f.stat().st_mtime_ns, f.stat().st_size)
             for f in sorted(resume.glob("*.npz"))}
    assert after == before, "an existing dump must not be rewritten when --skip-existing is set"

    # --- rejected input ------------------------------------------------------
    bad = run_dump(dump, model, corpus, work / "bad", ["--dump-kv-layers", "4096", "-b", "64"])
    assert bad.returncode != 0, "an out-of-range layer index must be rejected"
    assert "out of range" in (bad.stdout + bad.stderr), (
        "the out-of-range error should say so, got:\n" + bad.stdout + bad.stderr
    )

    bad_dtype = run_dump(dump, model, corpus, work / "bad_dtype",
                         ["--dump-kv-layers", "0", "--dump-dtype", "f64"])
    assert bad_dtype.returncode != 0, "an unknown dump dtype must be rejected"

    print("test-embedding-dump-kv: OK")


if __name__ == "__main__":
    main()
