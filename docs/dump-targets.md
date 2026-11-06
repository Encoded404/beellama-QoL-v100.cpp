# Distillation-target dump (`--dump-targets`)

`llama-embedding-dump` can record, for every position of every document, the
model's next-token distribution truncated at a nucleus threshold and capped at
a hard limit. Unlike `--dump-hidden`, this needs only logits, so it works for
any architecture and it does not require the target to expose `t_h_nextn`.

A targets-only dump is roughly **20 bytes per token** instead of ~24 KB, which
is what makes a corpus-scale distillation set practical:

```bash
llama-embedding-dump -m teacher.gguf --jsonl corpus.jsonl --output-dir targets \
    --dump-targets --no-dump-hidden --dump-kv-layers none -b 512
```

## Options

| option | default | meaning |
| --- | --- | --- |
| `--dump-targets` | off | record the per-position distribution |
| `--target-top-p` | `0.80` | keep the smallest id set whose mass reaches this |
| `--target-top-k` | `128` | hard cap on kept ids per position |
| `--target-table-depth` | `auto` | id table depth: `auto`, `128`, `1152` or `9344` |
| `--target-probs-dtype` | `f16` | `f16` stores `p`, `i8` stores `-16·ln(p)` |

## Archive layout

The six arrays are added to the same per-document `.npz` as the rest of the
dump, so document identity, write verification and `--skip-existing` resume all
apply unchanged.

| entry | dtype | shape | meaning |
| --- | --- | --- | --- |
| `target_meta` | `<i4` | `(6,)` | `[version, top_p_milli, top_k, table_depth, n_ids, flags]` |
| `target_counts` | `<u2` | `(T,)` | kept ids at each position, `1..top_k` |
| `target_lse` | `<f4` | `(T,)` | full-vocabulary `logsumexp` of the logits |
| `target_ids` | `<u1` | `(N,)` | packed code stream |
| `target_probs` | `<f2` or `<u1` | `(n_ids,)` | `p`, or `q = round(-16·ln p)`; see `flags` |
| `target_table` | `<u4` | `(Tbl,)` | per-document id table indexed by the short codes |

`target_meta[3]` is the table depth **actually used for this document**, which
under `auto` varies; a reader must take it from there or from
`len(target_table)`, never assume 9344. `target_meta[5]` bit 0 marks the `i8`
probability mode.

## Codec

Records are concatenated in position order; ids inside a position are in
descending probability. Bits are packed MSB-first, and the stream is padded to
a byte boundary (fewer than 8 unused trailing bits).

```
0   + 7 bits   ->  table[0 .. 127]           8 bits
10  + 10       ->  table[128 .. 1151]      12 bits
110 + 13       ->  table[1152 .. 9343]     16 bits
111 + 21       ->  raw token id            24 bits
```

The raw tier carries 21 payload bits because the vocabulary is `2^18`; 24 bits
round-trips any id and keeps the tier byte-aligned.

The table is built from **that document's kept ids**, by descending frequency
with ties on the lower id. Fitting it to the document's own *text* frequency
instead costs several bits per id, and a corpus-wide shared table is worse again
once the dump mixes domains. `auto` picks the depth that minimizes
`packed_id_bytes + 4 · table_entries` for the document; the ranking is one
ordering, so every depth is a prefix of it and the evaluation costs nothing
extra.

## The logsumexp is not optional

`target_lse` is the full-vocabulary partition. With it a consumer can measure
the mass the truncation dropped and renormalize deliberately:

```python
kept_mass   = p.sum()                  # over the kept ids
dropped     = 1.0 - kept_mass         # what top-p discarded
```

Without it, a small `--target-top-p` silently turns the teacher into a
near-greedy one: the kept ids renormalize to 1, nothing records that most of the
mass was never written, and the student learns to imitate over-confidence.

## Decoding

```python
import numpy as np

def decode(path):
    z = np.load(path)
    _, top_p_milli, top_k, depth, n_ids, flags = (int(v) for v in z["target_meta"])
    counts = z["target_counts"].astype(np.int64)
    probs  = z["target_probs"]
    probs  = np.exp(-probs / 16.0) if (flags & 1) else probs.astype(np.float64)
    lse    = z["target_lse"].astype(np.float64)
    table  = z["target_table"]
    blob   = z["target_ids"].tobytes()

    bit = 0
    def take(n):
        nonlocal bit
        v = 0
        for _ in range(n):
            v = (v << 1) | ((blob[bit >> 3] >> (7 - (bit & 7))) & 1)
            bit += 1
        return v

    out, p = [], 0
    for c in counts:
        ids = []
        for _ in range(c):
            if take(1) == 0:
                ids.append(int(table[take(7)]))
            elif take(1) == 0:
                ids.append(int(table[128 + take(10)]))
            elif take(1) == 0:
                ids.append(int(table[1152 + take(13)]))
            else:
                ids.append(take(21))
        out.append((ids, probs[p:p + c], lse[len(out)]))
        p += c
    return out
```

## Storage

Measured on a chat corpus at `--target-top-p 0.80`, `--target-top-k 128`:
10.6 kept ids per position on average (median 2, p90 23, 2.5 % at the cap), so

| component | per position |
| --- | --- |
| counts (`u2`) | 2 B |
| logsumexp (`f4`) | 4 B |
| ids, `auto` depth | 1.5–2 B per id |
| probabilities, `f16` | 2 B per id |
| probabilities, `i8` | 1 B per id |
| **total** | **~43 B (`f16`) / ~32 B (`i8`)** |

250 M positions is therefore ~10.8 GB at `f16` and ~8.2 GB at `i8`, against
~6 TB for the same corpus dumped with hidden states and K/V.

## Behaviour worth knowing

* **Deterministic.** Ties in the top-k selection break on the lower id, and the
  table is a stable ordering, so re-dumping a document is byte-identical — which
  is what the `--dump-verify-writes` comparison depends on. Identical *within*
  one binary: the partition function is summed by an AVX2 loop on machines that
  have it and by a portable loop elsewhere, and the two agree to about 2e-6 in
  `target_lse` (three orders of magnitude inside the `f16` the probabilities are
  stored as, but enough to move a value across an `f16` rounding boundary now and
  then). `LLAMA_DUMP_TARGET_LOOP=portable` forces the fallback, which is how the
  test compares the two.
* **The partition function is dense.** Truncating terms far below the maximum
  cannot make `target_lse` cheap: on real rows only ~0.06 % of the vocabulary is
  more than 30 nats down, because the sum is set by the bulk of the distribution
  rather than by its head. The cost is therefore one pass of `n_vocab` `exp`
  evaluations per position, and the implementation is built around that — a
  single cache-blocked pass, f64 lanes for the sum, and a top-k that only walks
  the eight lanes whose group can still beat the worst kept candidate. Measured
  at ~41 µs per position at 16 threads on a Threadripper 2950X, against ~200 µs
  for the straight per-element formulation.
* **Batching is capped.** Every token requests logits, and the context keeps
  `n_vocab · n_outputs_max` floats of host output buffer, so a targets run caps
  the output batch at 512 tokens. `-b` larger than that is reduced, with a log
  line.
* **`--skip-existing` is array-aware.** A document is skipped only if it carries
  every array the current options write. A file that is missing some is replaced
  through a verifying write to `<name>.part` followed by a rename, so a verified
  document is never swapped for an unverified one.
* **Probabilities are lossy only in storage.** The truncation is decided on
  exact probabilities; `f16` rounds the stored value, and `i8` rounds
  `-16·ln p`, which is at most 3.2 % relative per id.
