#!/usr/bin/env python3
"""Behavioural test for llama-embedding-dump's distillation-target dump.

The targets are the teacher's next-token distribution for every position,
truncated at --target-top-p and capped at --target-top-k, with the full
vocabulary logsumexp recorded beside them. Nothing below needs a reference
implementation: each check is a property the format has to satisfy on its own.

* decode             - the packed stream must consume exactly the bits the
                       archive holds, and every code must resolve inside the
                       stored table or be a raw id
* nucleus            - the kept probabilities must reach top-p unless the
                       position was cut at top-k; this is what makes the
                       truncation mean anything
* mass               - kept mass never exceeds 1, and target_lse is finite
* chunk invariance   - the same document dumped with -b 32 and -b 128 gives
                       identical arrays, which pins per-position alignment
                       across chunk borders
* determinism        - two runs produce byte-identical archives; the write
                       verification path compares bytes, so a selection that
                       broke ties differently would make it useless
* probability modes  - f16 stores p, i8 stores -16*ln(p) rounded, and the two
                       must describe the same distribution
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
    print("test-embedding-dump-targets: numpy is not available, skipping")
    sys.exit(77)


CORPUS = [
    {"messages": [{"role": "user", "content": "the the the the little cat sat on the mat"}]},
    {"messages": [{"role": "user", "content": "a short story about a dog and a cat in a big green park"}]},
]

BASE_ARGS = ["--dump-targets", "--no-dump-hidden", "--dump-kv-layers", "none", "-c", "256", "-t", "2"]


def run_dump(dump, model, corpus, outdir, extra):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(dump), "-m", str(model), "--jsonl", str(corpus), "--output-dir", str(outdir), *BASE_ARGS, *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def dump_ok(dump, model, corpus, outdir, extra):
    proc = run_dump(dump, model, corpus, outdir, extra)
    assert proc.returncode == 0, f"dump failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"

    files = sorted(outdir.glob("*.npz"))
    assert files, f"no .npz files were written:\n{proc.stdout}\n{proc.stderr}"
    return files


def decode(path):
    """Decode the packed id stream into a list of id lists, one per position."""
    z = np.load(path)
    for key in ("target_meta", "target_counts", "target_lse", "target_ids",
                "target_probs", "target_table"):
        assert key in z.files, f"{path}: missing '{key}'"

    meta = [int(v) for v in z["target_meta"]]
    assert len(meta) == 6, f"target_meta has {len(meta)} fields"
    _version, top_p_milli, top_k, depth, n_ids, flags = meta

    counts = z["target_counts"].astype(np.int64)
    probs = z["target_probs"].astype(np.float64)
    lse = z["target_lse"].astype(np.float64)
    table = z["target_table"]
    blob = z["target_ids"].tobytes()

    bit = 0

    def take(nbits):
        nonlocal bit
        v = 0
        for _ in range(nbits):
            v = (v << 1) | ((blob[bit >> 3] >> (7 - (bit & 7))) & 1)
            bit += 1
        return v

    rows = []
    for c in counts:
        ids = []
        for _ in range(int(c)):
            if take(1) == 0:
                idx = take(7)
                assert idx < len(table), "8-bit code outside the table"
                ids.append(int(table[idx]))
            elif take(1) == 0:
                idx = 128 + take(10)
                assert idx < len(table) and idx < depth, "12-bit code outside the table"
                ids.append(int(table[idx]))
            elif take(1) == 0:
                idx = 1152 + take(13)
                assert idx < len(table) and idx < depth, "16-bit code outside the table"
                ids.append(int(table[idx]))
            else:
                ids.append(take(21))
        rows.append(ids)

    return {
        "counts": counts, "stored": probs, "lse": lse, "table": table,
        "rows": rows, "bits_used": bit, "bits_total": len(blob) * 8,
        "top_p": top_p_milli / 1000.0, "top_k": top_k, "depth": depth,
        "n_ids": n_ids, "i8": bool(flags & 1),
    }


def check_document(path):
    d = decode(path)

    # i8 stores q = round(-16*ln p), f16 stores p itself
    values = np.exp(-d["stored"] / 16.0) if d["i8"] else d["stored"]
    d["probs"] = values

    # the truncation itself is decided on exact probabilities; only the stored
    # values are lossy, so the i8 mass can sit up to half a quantisation step
    # (~3.2 %) per id away from the exact sum
    tol = 0.05 if d["i8"] else 1e-3

    assert d["n_ids"] == int(d["counts"].sum()) == len(values), \
        f"{path}: n_ids/counts/probs disagree ({d['n_ids']}, {int(d['counts'].sum())}, {len(values)})"
    assert d["n_ids"] == sum(len(r) for r in d["rows"]), f"{path}: decoded id count mismatch"
    # the stream is padded to a byte boundary, so at most 7 trailing bits are
    # unused; anything else means a code is misaligned or the archive is short
    assert 0 <= d["bits_total"] - d["bits_used"] < 8, \
        f"{path}: stream consumed {d['bits_used']} of {d['bits_total']} bits (padding or misalignment)"
    assert len(d["table"]) <= d["depth"], f"{path}: table longer than its recorded depth"
    assert np.all(d["counts"] >= 1) and np.all(d["counts"] <= d["top_k"])

    p = 0
    for i, c in enumerate(d["counts"]):
        row = values[p:p + c]
        p += c
        mass = float(row.sum())
        assert mass <= 1.0 + tol, f"{path}: position {i} keeps mass {mass:.6f} > 1"
        if c == d["top_k"]:
            continue  # truncated at the cap, mass may fall short of top-p
        assert mass >= d["top_p"] - tol, \
            f"{path}: position {i} keeps only {mass:.6f} < top-p {d['top_p']} and is not at the cap"

    assert np.all(np.isfinite(d["lse"])), f"{path}: non-finite logsumexp"

    return d


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

    # --- format and semantics ------------------------------------------------
    base = dump_ok(dump, model, corpus, work / "base", ["-b", "128"])
    ref = check_document(base[0])

    # targets-only mode must not drag in the hidden state or any K/V array
    names = set(np.load(base[0]).files)
    assert "hidden" not in names, "a targets-only dump wrote 'hidden'"
    assert not any(n.startswith(("k_l", "v_l")) for n in names), "a targets-only dump wrote K/V"

    # --- chunk invariance ----------------------------------------------------
    # the same document split differently must produce the same targets: this is
    # what catches a row/label misalignment at a chunk border
    small = dump_ok(dump, model, corpus, work / "small", ["-b", "32"])
    for key in ("target_ids", "target_counts", "target_probs", "target_lse", "target_table", "target_meta"):
        a = np.load(base[0])[key]
        b = np.load(small[0])[key]
        assert np.array_equal(a, b), f"'{key}' depends on the batch split"

    # --- determinism ---------------------------------------------------------
    again = dump_ok(dump, model, corpus, work / "again", ["-b", "128"])
    assert base[0].read_bytes() == again[0].read_bytes(), \
        "two identical runs produced different bytes (selection is not deterministic)"

    # --- a wider nucleus must keep at least as much ------------------------
    wide = dump_ok(dump, model, corpus, work / "wide", ["-b", "128", "--target-top-p", "0.95"])
    wide_doc = check_document(wide[0])
    assert wide_doc["counts"].sum() >= ref["counts"].sum(), "top-p 0.95 kept fewer ids than 0.80"

    # --- probability modes ---------------------------------------------------
    # i8 stores q = round(-16*ln p); the two dumps must describe one distribution
    i8 = dump_ok(dump, model, corpus, work / "i8", ["-b", "128", "--target-probs-dtype", "i8"])
    i8_doc = check_document(i8[0])
    assert i8_doc["i8"], "i8 dump did not mark its probability mode"

    f16 = ref["probs"]
    p8 = i8_doc["probs"]
    assert len(p8) == len(f16), "i8 and f16 dumps kept different id counts"
    rel = np.abs(p8 - f16) / np.maximum(f16, 1e-12)
    # half a quantisation step is exp(1/32) - 1 ~= 3.2 %
    assert np.all(rel <= 0.05), f"i8 probabilities differ from f16 by up to {rel.max():.3f}"

    # --- explicit table depth ------------------------------------------------
    for depth in ("128", "1152", "9344"):
        got = dump_ok(dump, model, corpus, work / f"depth{depth}",
                      ["-b", "128", "--target-table-depth", depth])
        d = check_document(got[0])
        assert d["depth"] == int(depth), f"--target-table-depth {depth} recorded {d['depth']}"

    print("test-embedding-dump-targets: all checks passed")


if __name__ == "__main__":
    main()
