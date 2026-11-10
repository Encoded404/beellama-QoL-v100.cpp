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
* stick codec        - the v2 layout (target_prob_meta) decodes to the f16
                       distribution, beats i8 on KL and on worst-case error at
                       one byte per position more, and a wider body code beats a
                       narrower one
"""

import argparse
import json
import os
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


def kl_div(a, b):
    a = a / a.sum()
    b = b / b.sum()
    return float((a * np.log(a / np.maximum(b, 1e-300))).sum())


CORPUS = [
    {"messages": [{"role": "user", "content": "the the the the little cat sat on the mat"}]},
    {"messages": [{"role": "user", "content": "a short story about a dog and a cat in a big green park"}]},
]

BASE_ARGS = ["--dump-targets", "--no-dump-hidden", "--dump-kv-layers", "none", "-c", "256", "-t", "2"]


def run_dump(dump, model, corpus, outdir, extra, env=None):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(dump), "-m", str(model), "--jsonl", str(corpus), "--output-dir", str(outdir), *BASE_ARGS, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})


def dump_ok(dump, model, corpus, outdir, extra, env=None):
    proc = run_dump(dump, model, corpus, outdir, extra, env)
    assert proc.returncode == 0, f"dump failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"

    files = sorted(outdir.glob("*.npz"))
    assert files, f"no .npz files were written:\n{proc.stdout}\n{proc.stderr}"
    return files


def decode(path):
    """Decode the packed id stream and the probability stream.

    target_prob_meta[0] names the probability layout, so nothing here has to
    infer it from dtype or shape (the v1 i8 stream and the v2 stick stream are
    both <u1 with one byte per id, and cannot be told apart otherwise).
    """
    z = np.load(path)
    for key in ("target_meta", "target_counts", "target_lse", "target_ids",
                "target_probs", "target_table", "target_prob_meta", "target_sel_meta"):
        assert key in z.files, f"{path}: missing '{key}'"

    meta = [int(v) for v in z["target_meta"]]
    assert len(meta) == 6, f"target_meta has {len(meta)} fields"
    version, top_p_milli, top_k, depth, n_ids, flags = meta
    assert version == 3, f"{path}: unexpected target_meta version {version}"

    smeta = [int(v) for v in z["target_sel_meta"]]
    assert len(smeta) == 6, f"target_sel_meta has {len(smeta)} fields"
    min_ids, min_p_micro, extra_ids, extra_min, extra_max, has_tail = smeta
    assert min_p_micro >= 0 and min_ids >= 0 and extra_ids >= 0
    if extra_ids:
        assert 1 <= extra_min <= extra_max

    tail_n = tail_deg = tail_vocab = 0
    tail_poly = None
    if has_tail:
        assert "target_tail_meta" in z.files and "target_tail_poly" in z.files, \
            f"{path}: selection says tail present but the arrays are missing"
        tail_n, tail_deg, tail_vocab = (int(v) for v in z["target_tail_meta"])
        tail_poly = z["target_tail_poly"].astype(np.float64)
        assert tail_n > 0 and 1 <= tail_deg <= 3 and tail_vocab > 0
        assert tail_poly.shape == (len(z["target_counts"]), 5), \
            f"{path}: target_tail_poly shape {tail_poly.shape}"
        assert np.all(np.isfinite(tail_poly)), f"{path}: non-finite tail coefficients"
    else:
        assert "target_tail_poly" not in z.files, f"{path}: tail arrays present but flagged off"

    pmeta = [int(v) for v in z["target_prob_meta"]]
    assert len(pmeta) == 4, f"target_prob_meta has {len(pmeta)} fields"
    mode, body_bits, range_milli, head_bits = pmeta

    # the legacy flag bits and the meta array have to describe one layout
    if mode == 0:
        assert flags == 0, f"{path}: f16 layout with flags {flags}"
    elif mode == 1:
        assert flags == 1, f"{path}: i8 layout with flags {flags}"
    else:
        assert mode == 2 and flags == 2, f"{path}: stick layout with flags {flags}"
        assert body_bits >= 6 and range_milli > 0 and head_bits > body_bits

    counts = z["target_counts"].astype(np.int64)
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

    # --- probability stream --------------------------------------------------
    stored = z["target_probs"]
    prob_bits_used = prob_bits_total = 0

    if mode == 0:
        assert stored.dtype == np.float16, f"{path}: f16 layout stored as {stored.dtype}"
        values = stored.astype(np.float64)
    elif mode == 1:
        assert stored.dtype == np.uint8, f"{path}: i8 layout stored as {stored.dtype}"
        values = np.exp(-stored.astype(np.float64) / 16.0)
    else:
        assert stored.dtype == np.uint8, f"{path}: stick layout stored as {stored.dtype}"
        pblob = stored.tobytes()

        def step(b):
            return (range_milli / 1000.0) / ((1 << b) - 1)

        pbit = 0

        def take_bits(nbits):
            nonlocal pbit
            v = 0
            for _ in range(nbits):
                v = (v << 1) | ((pblob[pbit >> 3] >> (7 - (pbit & 7))) & 1)
                pbit += 1
            return v

        vals = []
        for c in counts:
            R = 1.0
            for j in range(int(c)):
                b = head_bits if j == 0 else body_bits
                s = 2.0 ** (-take_bits(b) * step(b))
                p_val = s * R
                vals.append(p_val)
                R = max(R - p_val, 0.0)
        values = np.array(vals)
        prob_bits_used = pbit
        prob_bits_total = len(pblob) * 8

    ce_mask = None
    ce = None
    if "target_ce_mask" in z.files:
        assert "target_ce" in z.files, f"{path}: CE mask without CE values"
        ce_mask = z["target_ce_mask"].tobytes()
        ce = z["target_ce"].astype(np.float64)
        assert len(ce_mask) == (len(counts) + 7) // 8, \
            f"{path}: CE mask is {len(ce_mask)} bytes for {len(counts)} positions"

    return {
        "counts": counts, "probs": values, "lse": lse, "table": table,
        "labels": z["labels"].astype(np.int64), "ce_mask": ce_mask, "ce": ce,
        "rows": rows, "bits_used": bit, "bits_total": len(blob) * 8,
        "prob_bits_used": prob_bits_used, "prob_bits_total": prob_bits_total,
        "top_p": top_p_milli / 1000.0, "top_k": top_k, "depth": depth,
        "n_ids": n_ids, "mode": mode, "flags": flags,
        "bits": body_bits, "head_bits": head_bits, "range": range_milli / 1000.0,
        "version": version, "min_ids": min_ids, "min_p": min_p_micro / 1e6,
        "extra_ids": extra_ids, "extra_min": extra_min, "extra_max": extra_max,
        "tail_n": tail_n, "tail_deg": tail_deg, "tail_vocab": tail_vocab,
        "tail_poly": tail_poly,
    }


def check_document(path):
    d = decode(path)

    values = d["probs"]

    # the truncation itself is decided on exact probabilities; only the stored
    # values are lossy, so the stored mass can sit a quantisation step away from
    # the exact sum. i8 is 3.2 % per id, the stick code is a body step plus the
    # remainder drift, and f16 is exact to rounding.
    tol = {0: 1e-3, 1: 0.05, 2: 0.01}[d["mode"]]

    assert d["n_ids"] == int(d["counts"].sum()) == len(values), \
        f"{path}: n_ids/counts/probs disagree ({d['n_ids']}, {int(d['counts'].sum())}, {len(values)})"
    assert d["n_ids"] == sum(len(r) for r in d["rows"]), f"{path}: decoded id count mismatch"
    # the stream is padded to a byte boundary, so at most 7 trailing bits are
    # unused; anything else means a code is misaligned or the archive is short
    assert 0 <= d["bits_total"] - d["bits_used"] < 8, \
        f"{path}: stream consumed {d['bits_used']} of {d['bits_total']} bits (padding or misalignment)"
    if d["mode"] == 2:
        assert 0 <= d["prob_bits_total"] - d["prob_bits_used"] < 8, \
            f"{path}: probability stream consumed {d['prob_bits_used']} of " \
            f"{d['prob_bits_total']} bits (padding or misalignment)"
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

    # the kept count never goes below the floor the rules ask for, and every
    # position has at least one id
    assert np.all(d["counts"] >= 1), f"{path}: a position kept nothing"
    if d["min_ids"] > 0:
        floor = min(d["min_ids"], d["top_k"])
        assert np.all(d["counts"] >= floor), \
            f"{path}: min-ids {d['min_ids']} not honoured (min count {d['counts'].min()})"

    # CE is stored exactly where the gold token is *not* among the kept ids, so
    # the flag and the membership have to agree position by position, and every
    # stored value must be at least as large as -ln(p_last_kept) because the gold
    # ranks below everything that was kept.
    if d["ce_mask"] is not None:
        mask, vals = d["ce_mask"], d["ce"]
        got = sum((mask[i >> 3] >> (7 - (i & 7))) & 1 for i in range(len(d["counts"])))
        assert got == len(vals), f"{path}: {got} flags but {len(vals)} CE values"

        vi = 0
        p = 0
        for i, (ids, c) in enumerate(zip(d["rows"], d["counts"])):
            flagged = bool((mask[i >> 3] >> (7 - (i & 7))) & 1)
            inside = int(d["labels"][i]) in ids
            assert flagged != inside, \
                f"{path}: position {i} flagged {flagged} but gold inside {inside}"
            if flagged:
                floor = -np.log(float(values[p + c - 1]))
                assert vals[vi] >= floor - 1e-3, \
                    f"{path}: position {i} CE {vals[vi]:.4f} below -ln(p_last) {floor:.4f}"
                assert np.isfinite(vals[vi]) and vals[vi] >= 0.0, \
                    f"{path}: position {i} CE {vals[vi]}"
                vi += 1
            p += c

    # the tail shape, when present, has to agree with the kept mass: the ids and
    # the fitted window are disjoint, so their masses cannot sum past 1
    if d["tail_poly"] is not None:
        p = 0
        for i, c in enumerate(d["counts"]):
            kept = float(values[p:p + c].sum())
            p += c
            band = float(d["tail_poly"][i, 4])
            assert np.isfinite(band) and band >= 0.0, f"{path}: position {i} band mass {band}"
            # the stored kept mass carries the mode's own quantisation error,
            # which is why the bound uses the same tolerance as above
            assert kept + band <= 1.0 + max(tol, 2e-3), \
                f"{path}: position {i} kept {kept:.4f} + band {band:.4f} > 1"

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
    for key in ("target_ids", "target_counts", "target_probs", "target_lse",
                "target_table", "target_meta", "target_prob_meta",
                "target_sel_meta", "target_tail_meta", "target_tail_poly",
                "target_ce_mask", "target_ce"):
        a = np.load(base[0])[key]
        b = np.load(small[0])[key]
        assert np.array_equal(a, b), f"'{key}' depends on the batch split"

    # --- determinism ---------------------------------------------------------
    again = dump_ok(dump, model, corpus, work / "again", ["-b", "128"])
    assert base[0].read_bytes() == again[0].read_bytes(), \
        "two identical runs produced different bytes (selection is not deterministic)"

    # --- the vectorised inner loop agrees with the portable one ---------------
    # both paths run over the same document. the ids and counts come out of the
    # same comparisons and must match exactly; the partition function is summed
    # by different code, so it may move by a few ulp - far inside the f16 the
    # probabilities are stored as.
    port = dump_ok(dump, model, corpus, work / "portable", ["-b", "128"],
                   env={"LLAMA_DUMP_TARGET_LOOP": "portable"})
    a = np.load(base[0])
    b = np.load(port[0])
    assert np.array_equal(a["target_counts"], b["target_counts"]), \
        "the avx2 and portable inner loops kept different id counts"
    assert np.array_equal(a["target_ids"], b["target_ids"]), \
        "the avx2 and portable inner loops selected different ids"
    dl = np.abs(a["target_lse"].astype(np.float64) - b["target_lse"].astype(np.float64))
    assert dl.max() <= 1e-4, f"target_lse differs by {dl.max():.3e} between the inner loops"
    pa = a["target_probs"].astype(np.float64)
    pb = b["target_probs"].astype(np.float64)
    rel = np.abs(pa - pb) / np.maximum(pa, 1e-12)
    assert np.all(rel <= 1e-3), f"probabilities differ by {rel.max():.3e} between the inner loops"

    # --- a wider nucleus must keep at least as much ------------------------
    wide = dump_ok(dump, model, corpus, work / "wide", ["-b", "128", "--target-top-p", "0.95"])
    wide_doc = check_document(wide[0])
    assert wide_doc["counts"].sum() >= ref["counts"].sum(), "top-p 0.95 kept fewer ids than 0.80"

    # --- the cap is a u16, not a byte ---------------------------------------
    # target_counts is stored as u2, so a cap above 255 has to round-trip: it
    # must not be clamped by the codec or by the count array.
    big = dump_ok(dump, model, corpus, work / "cap300",
                  ["-b", "128", "--target-top-p", "0.99", "--target-top-k", "300"])
    big_doc = check_document(big[0])
    assert big_doc["top_k"] == 300, f"cap recorded as {big_doc['top_k']}"
    assert big_doc["counts"].max() > 255, "a cap of 300 never produced a count above 255"

    # --- probability modes ---------------------------------------------------
    # i8 stores q = round(-16*ln p); the two dumps must describe one distribution
    i8 = dump_ok(dump, model, corpus, work / "i8", ["-b", "128", "--target-probs-dtype", "i8"])
    i8_doc = check_document(i8[0])
    assert i8_doc["mode"] == 1, "i8 dump did not mark its probability mode"

    f16 = ref["probs"]
    p8 = i8_doc["probs"]
    assert len(p8) == len(f16), "i8 and f16 dumps kept different id counts"
    rel = np.abs(p8 - f16) / np.maximum(f16, 1e-12)
    # half a quantisation step is exp(1/32) - 1 ~= 3.2 %
    assert np.all(rel <= 0.05), f"i8 probabilities differ from f16 by up to {rel.max():.3f}"

    # --- stick (log) probability codec ---------------------------------------
    # 8 bits per id plus one 16-bit head per position. At that price it has to
    # beat i8 on accuracy (the point of the layout) while staying smaller than the
    # f16 dump, and it has to decode to the same distribution as f16.
    log8 = dump_ok(dump, model, corpus, work / "log8",
                   ["-b", "128", "--target-probs-dtype", "log", "--target-prob-bits", "8"])
    log8_doc = check_document(log8[0])
    assert log8_doc["mode"] == 2, "log dump did not record its probability layout"
    assert log8_doc["bits"] == 8 and log8_doc["head_bits"] > 8

    pl = log8_doc["probs"]
    assert len(pl) == len(f16), "log and f16 dumps kept different id counts"
    rel_log = np.abs(pl - f16) / np.maximum(f16, 1e-12)
    rel_i8 = np.abs(p8 - f16) / np.maximum(f16, 1e-12)
    assert np.all(rel_log <= 0.05), f"log probabilities differ from f16 by up to {rel_log.max():.3f}"

    # better than i8 where it matters: the distribution and the worst id
    kl_log = kl_div(f16, pl)
    kl_i8 = kl_div(f16, p8)
    assert kl_log < kl_i8, f"log KL {kl_log:.3e} is not better than i8 KL {kl_i8:.3e}"
    # the worst single id is not the metric: a position that keeps many ids
    # accumulates remainder drift in the stick decoder, so its max can exceed
    # i8's fixed 3.2 % while its mean, p99 and KL stay well below. the aggregate
    # is what a training run sees.
    assert np.percentile(rel_log, 99) < np.percentile(rel_i8, 99), \
        "log p99 error is not better than i8"
    assert rel_log.mean() < rel_i8.mean(), "log mean error is not better than i8"

    # the 16-bit head costs one byte per position over i8, nothing else
    n_pos = len(log8_doc["counts"])
    over = log8[0].stat().st_size - i8[0].stat().st_size
    assert over <= n_pos + 128, f"log dump costs {over} B over i8 for {n_pos} positions"
    assert log8[0].stat().st_size < base[0].stat().st_size, "log dump is not smaller than f16"

    # a wider body code must be strictly more accurate than a narrower one
    log12 = dump_ok(dump, model, corpus, work / "log12",
                    ["-b", "128", "--target-probs-dtype", "log", "--target-prob-bits", "12"])
    log12_doc = check_document(log12[0])
    rel12 = np.abs(log12_doc["probs"] - f16) / np.maximum(f16, 1e-12)
    assert rel12.max() < rel_log.max(), "12 bits is not more accurate than 8 bits"
    assert log12[0].stat().st_size > log8[0].stat().st_size, "12 bits did not cost more bytes"

    # the head width is part of the layout, not a guess: the decoder needs it
    assert log12_doc["head_bits"] == log8_doc["head_bits"], "head width changed between runs"

    # --- selection rules -----------------------------------------------------
    # every floor off must reproduce the plain nucleus, which is what this dump
    # produced before the rules existed
    off = ["-b", "128", "--target-min-ids", "0", "--target-min-p", "0",
           "--target-extra-ids", "0", "--target-tail-n", "0"]
    plain = dump_ok(dump, model, corpus, work / "plain", off)
    plain_doc = check_document(plain[0])
    assert plain_doc["min_ids"] == 0 and plain_doc["extra_ids"] == 0
    assert plain_doc["tail_poly"] is None
    assert "target_tail_poly" not in np.load(plain[0]).files

    # with the defaults, no position is below the floor, and the floors together
    # keep strictly more ids than the nucleus alone
    on = dump_ok(dump, model, corpus, work / "on", ["-b", "128"])
    on_doc = check_document(on[0])
    assert on_doc["min_ids"] == 4 and abs(on_doc["min_p"] - 0.005) < 1e-9
    assert on_doc["extra_ids"] == 1 and on_doc["extra_min"] == 4 and on_doc["extra_max"] == 48
    assert on_doc["counts"].min() >= 4, f"min-ids not honoured: {on_doc['counts'].min()}"
    assert on_doc["counts"].sum() > plain_doc["counts"].sum(), "the floors kept no extra ids"
    # min-ids 4 lands inside [4, 48], so the extra rule makes 5 the effective floor
    assert on_doc["counts"].min() == 5, f"effective floor is {on_doc['counts'].min()}, expected 5"

    # --- tail shape ----------------------------------------------------------
    assert on_doc["tail_n"] == 1024 and on_doc["tail_deg"] == 3
    assert on_doc["tail_poly"].shape == (len(on_doc["counts"]), 5)
    assert on_doc["tail_vocab"] > 1024

    t0 = dump_ok(dump, model, corpus, work / "tail0", ["-b", "128", "--target-tail-n", "0"])
    check_document(t0[0])
    assert "target_tail_poly" not in np.load(t0[0]).files, "tail arrays written with the tail off"

    t256 = dump_ok(dump, model, corpus, work / "tail256", ["-b", "128", "--target-tail-n", "256"])
    t256_doc = check_document(t256[0])
    assert t256_doc["tail_n"] == 256
    # a narrower window holds less of the tail's mass but costs the same: N is a
    # fitting window, and the stored shape is 4 coefficients + the mass either way
    assert np.mean(t256_doc["tail_poly"][:, 4]) < np.mean(on_doc["tail_poly"][:, 4]), \
        "a 256-rank window captured more mass than a 1024-rank one"
    assert t256[0].stat().st_size == on[0].stat().st_size, \
        "changing the fit window changed the file size"
    assert t0[0].stat().st_size < t256[0].stat().st_size, "turning the tail off did not save bytes"

    # --- gold cross-entropy --------------------------------------------------
    # stored only where the gold falls outside the kept ids, so the count is a
    # property of the selection, not of the document
    assert on_doc["ce_mask"] is not None and on_doc["ce"] is not None
    n_flagged = int(sum((on_doc["ce_mask"][i >> 3] >> (7 - (i & 7))) & 1
                        for i in range(len(on_doc["counts"]))))
    assert n_flagged > 0, "no position ever had its gold outside the kept set"
    assert n_flagged == len(on_doc["ce"])

    # and it can be turned off, which removes both arrays
    noce = dump_ok(dump, model, corpus, work / "noce", ["-b", "128", "--no-target-ce"])
    noce_doc = check_document(noce[0])
    assert noce_doc["ce_mask"] is None
    assert "target_ce" not in np.load(noce[0]).files
    assert noce[0].stat().st_size < on[0].stat().st_size

    # --- explicit table depth ------------------------------------------------
    for depth in ("128", "1152", "9344"):
        got = dump_ok(dump, model, corpus, work / f"depth{depth}",
                      ["-b", "128", "--target-table-depth", depth])
        d = check_document(got[0])
        assert d["depth"] == int(depth), f"--target-table-depth {depth} recorded {d['depth']}"

    print("test-embedding-dump-targets: all checks passed")


if __name__ == "__main__":
    main()
