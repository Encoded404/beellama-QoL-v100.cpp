# Distillation-target dump (`--dump-targets`)

`llama-embedding-dump` can record, for every position of every document, the
model's next-token distribution truncated at a nucleus threshold and capped at
a hard limit. Unlike `--dump-hidden`, this needs only logits, so it works for
any architecture and it does not require the target to expose `t_h_nextn`.

A targets-only dump is roughly **45 to 70 bytes per token** instead of ~24 KB,
which is what makes a corpus-scale distillation set practical:

```bash
llama-embedding-dump -m teacher.gguf --jsonl corpus.jsonl --output-dir targets \
    --dump-targets --no-dump-hidden --dump-kv-layers none \
    --target-probs-dtype log -b 512
```

## Options

| option | default | meaning |
| --- | --- | --- |
| `--dump-targets` | off | record the per-position distribution |
| `--target-top-p` | `0.85` | keep the smallest id set whose mass reaches this |
| `--target-top-k` | `256` | hard cap on kept ids per position (max 4096) |
| `--target-table-depth` | `auto` | id table depth: `auto`, `128`, `1152` or `9344` |
| `--target-probs-dtype` | `f16` | `f16` stores `p`, `i8` stores `-16·ln(p)`, `log` stores the stick code |
| `--target-prob-bits` | `8` | body code width for `log`, 6..12 |
| `--target-min-ids` | `4` | keep at least this many ids per position (0 = off) |
| `--target-min-p` | `0.005` | keep ids whose own probability is at least this (0 = off) |
| `--target-extra-ids` | `1` | ids added when a position's count lands in the range below |
| `--target-extra-min` / `--target-extra-max` | `4` / `48` | the range the extra rule applies to |
| `--target-tail-n` | `1024` | ranks of tail shape fitted after the kept set (0 = off) |
| `--target-tail-degree` | `3` | polynomial degree of the tail shape, 1..3 |
| `--target-ce` / `--no-target-ce` | on | record the gold token's cross-entropy where it is outside the kept set |

## Archive layout

The arrays are added to the same per-document `.npz` as the rest of the dump,
so document identity, write verification and `--skip-existing` resume all apply
unchanged.

| entry | dtype | shape | meaning |
| --- | --- | --- | --- |
| `target_meta` | `<i4` | `(6,)` | `[version, top_p_milli, top_k, table_depth, n_ids, flags]` |
| `target_prob_meta` | `<i4` | `(4,)` | `[prob_mode, body_bits, range_milli, head_bits]` |
| `target_sel_meta` | `<i4` | `(6,)` | `[min_ids, min_p_micro, extra_ids, extra_min, extra_max, has_tail]` |
| `target_tail_meta` | `<i4` | `(3,)` | `[tail_n, degree, n_vocab]`, only when `has_tail` |
| `target_tail_poly` | `<f2` | `(T, 5)` | `[c0, c1, c2, c3, band_mass]`, only when `has_tail` |
| `target_ce_mask` | `<u1` | `(ceil(T/8),)` | bit `i` (MSB first) set = CE stored for position `i` |
| `target_ce` | `<f2` | `(n_ce,)` | `-ln p_gold` for the flagged positions, in order |
| `target_counts` | `<u2` | `(T,)` | kept ids at each position, `1..top_k` |
| `target_lse` | `<f4` | `(T,)` | full-vocabulary `logsumexp` of the logits |
| `target_ids` | `<u1` | `(N,)` | packed code stream |
| `target_probs` | `<f2`, `<u1` or `<u1` | `(n_ids,)` or `(B,)` | see `prob_mode` |
| `target_table` | `<u4` | `(Tbl,)` | per-document id table indexed by the short codes |

`version` is `3`. `target_meta[3]` is the table depth **actually used for this
document**, which under `auto` varies; a reader must take it from there or from
`len(target_table)`, never assume 9344.

`prob_mode` is `0` for f16, `1` for the version-1 absolute log code, and `2` for
the stick code. `target_meta[5]` mirrors it: bit 0 is mode 1, bit 1 is mode 2,
and they are mutually exclusive, so a reader that predates version 2 sees
bit 0 clear with a `<u1` array and fails on the mass instead of decoding a
packed stream as one value per id.

Version 2 added `target_prob_meta` because the layout cannot be inferred from
the array: the version-1 i8 stream and the version-2 stick stream are both
`<u1` with exactly one byte per id. **Read `target_prob_meta` (or check
`version`) before touching `target_probs`.**

## Selection rules

A position keeps the **maximum** of four independently computed counts, then the
extra rule widens it once:

| rule | count it produces |
| --- | --- |
| nucleus | first rank where the cumulative mass reaches `top_p`, capped at `top_k` |
| min-p | first rank whose own probability drops below `min_p` |
| min-ids | `min_ids` |
| extra | the result so far, plus `extra_ids`, if it lands in `[extra_min, extra_max]` |

Every rule only ever *extends* the set, so a reader that wants the plain nucleus
back can recompute the cumulative mass from the stored probabilities and cut at
`top_p` again. Nothing is lost, and the extensions are ignorable tail.

`min_p` is **absolute**, not relative to the top id. A relative floor
(`p >= min_p * p_max`) raises the bar on exactly the peaked positions that
already keep few ids, which is the opposite of the intent; absolute is also what
bounds the cost, since at most `1/min_p` ids can clear it.

The extra rule runs on the final count, so with the defaults a position that the
nucleus would have cut at one id ends up at **five**: `min_ids` lifts it to 4,
which lands inside `[4, 48]`, and the extra rule adds one. `--target-min-ids 3`
gives an effective floor of four.

`target_sel_meta` records all of it, so a reader can tell a nucleus cut from a
floor and can reproduce either. One consequence worth knowing: forcing ids onto
a position whose head is already 0.999 spends bytes on ids carrying ~1e-5 each.
That is rare on real text (0.8 % of positions on the model this was measured
with) but it is what `min_p` is better at than `min_ids`, because a probability
floor stops on its own there.

## Id codec

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

## Probability codec (`log`)

The probability array is a second packed stream, one code per kept id, in the
same order as the ids. It stores the **stick-breaking share** of the mass that
was still unallocated when the id was reached:

```
R_0 = 1                        the full distribution, truncated tail included
s_i = p_i / R_i                in (0, 1]
R_{i+1} = R_i - p_i
```

`L_i = -log2(s_i)` is quantised as a fixed-width integer over a per-document
range `[0, range]`:

```
step = range / (2^bits - 1)
code = clamp(round(L / step), 0, 2^bits - 1)
```

Rank 0 of every position uses `head_bits` (currently 16) and every later rank
uses `body_bits` (`--target-prob-bits`, default 8). The head is wider because
everything after it is measured against the remainder it leaves behind: a head
at 0.99 rounded to 8 bits can round up to exactly 1.0, which sets the decoder's
remainder to zero and turns the rest of the position into zeros. That failure is
why the rank-0 width is part of the format rather than a tuning knob.

`range_milli` is the document's own scale, `ceil(max L / 0.25) · 0.25` clamped to
`[0.5, 16]`, stored in thousandths. It is a plain max over the exact
probabilities, so it is deterministic; the 0.25 grid makes it round-trip exactly
through integer milli-units. It costs 4 bytes per document, not per position. A
single very flat position widens the range for the whole document, which costs
resolution everywhere; that is why the range is not shared across documents
either.

Decoding is

```
R = 1
for each code:
    s = 2 ** (-code * range / (2^width - 1))
    p = s * R
    R = max(R - p, 0)
```

so the reconstruction is a sub-probability by construction and a position's kept
mass can never exceed 1 however the codes round.

**Accuracy.** The per-id relative error is the code's half step plus a slow
remainder drift that grows with the number of ids in a position and does not
shrink when the step does. Measured on 12,242 ids from a 15M model, against the
f16 dump as reference:

| mode | bytes vs i8 | p99 rel | mean rel | KL |
| --- | --- | --- | --- | --- |
| `f16` | +25 % | (reference) | | |
| `i8` | — | 3.12 % | 1.58 % | 1.8e-4 |
| `log` @ 8 | +0.4 % | 1.64 % | 0.71 % | 2.2e-5 |
| `log` @ 10 | +6.4 % | 0.42 % | 0.18 % | 1.3e-6 |
| `log` @ 12 | +12.4 % | 0.12 % | 0.05 % | 1.1e-7 |

At 8 bits the code costs one byte per position more than `i8` (the 16-bit head)
and is roughly 8× closer to the teacher by KL. Below 8 bits it degrades past
`i8`: the body step alone reaches 10 % at 6 bits, so `i8` remains the better
choice there.

The table leaves out the single worst id on purpose. Once a position keeps many
ids the stick decoder's remainder drifts, and a position whose head is already
0.999 can end up with a 20 % error on an id carrying 2e-5 — real, harmless, and
not improved by more bits, because it is drift rather than step size. The
aggregate columns are the ones a training run sees. If the worst case matters
more than the aggregate, `--target-prob-bits 10` shrinks it somewhat and the
`i8` mode bounds it at 3.2 % by construction.

## Tail shape (`target_tail_poly`)

The kept ids describe the head exactly; this records how the *rest* of the mass
is spread over the next `tail_n` ranks, which no id/probability list can carry
cheaply. It is fitted in log-log space because a next-token distribution is close
to a power law there — a cubic in raw rank spans four orders of magnitude and is
numerically useless, while a cubic in log rank is well conditioned and captures
the bend where the power law runs out of mass:

```
p(r) = exp(c0 + c1*t + c2*t^2 + c3*t^3),   t = (ln r - ln r0)/(ln r1 - ln r0)
r0 = k + 1                     first rank after the kept ids
r1 = min(k + tail_n, n_vocab)  last rank of the window
```

`k` is the position's `target_counts`, and `n_vocab` is in `target_tail_meta`, so
the window needs nothing per position. The row is 5 fp16 values: four
coefficients and `band_mass`, the **exact** mass over the same window. Use
`band_mass` when you need the tail's weight; use the coefficients when you need
its shape. Storing the exact mass also makes the row checkable — the curve's own
integral should agree with it, and a disagreement is a fit-quality signal.

The cost is fixed at 10 bytes per position whatever `tail_n` is: `tail_n` is a
*fitting* window, not a storage knob. Turning it off with `--target-tail-n 0`
removes the two arrays.

A cubic is not monotone in general. A fit that comes out non-monotone or
negative inside its own window is refitted as a power law (degree 1, higher
coefficients zero), so a reader always evaluates the same expression and no
per-position degree has to be stored.

Measured against the true tail of 126 positions (fitted cubic against the
teacher's actual probabilities at sampled ranks):

| metric | median | p90 | max |
| --- | --- | --- | --- |
| `|ln(fit) - ln(true)|` | 0.012 nats | 0.041 nats | 0.67 nats |
| band mass relative error | 0.018 % | 0.031 % | 0.05 % |

So the shape is good to ~1 % in probability at the median and ~4 % at p90; the
band mass is exact to the fp16 it is stored as.

**What it is not.** The curve describes mass *at a rank*, not *which token* is at
that rank. A per-token distillation loss computes `sum p_i log q_i` and needs
identity, so it cannot consume this. What can: rank-profile terms (teacher mass
in ranks `[k+1, k+256)` against the student's mass in its own ranks), tail-mass
or entropy regularizers, calibration analysis, and reconstructing "what would a
larger `top_p` have kept" on a corpus that is already dumped.

## Gold cross-entropy (`target_ce`)

`CE = lse - logit_gold` is the teacher's loss on the token that actually came
next. It costs one array read per position — the partition function and the row
are already in hand — and it is the cheapest scalar in the archive to produce.

It is only stored where it is **not** already recoverable. When the gold is among
the kept ids its probability is in `target_probs`, so `-ln(p_gold)` comes straight
out of the distribution and the id is in `labels`. Only the positions where the
gold fell outside the kept set need a value, and those are exactly the ones where
the archived distribution says nothing about the token that occurred.

Storage is a bit mask plus f16 values. `f16` rather than the `-16·ln(p)` code
because that code saturates at 15.9 nats, which is precisely the range where a
teacher has lost the thread and a CE filter is most useful. The mask costs 1/8
byte per position and the values 2 bytes per flagged position, so at the ~67 %
flag rate measured here it is about 1.5 B per position; `--no-target-ce` removes
both arrays.

Measured against `-ln p_gold` from a near-complete dump: median error 0.0017
nats, p90 0.0038, max 0.021 — the f16 quantization, nothing else.

Two things it is for. First, the gold term: a consumer that wants
`p_gold · (-log q_gold)` in the loss has the id in `labels` and now the
probability too, without widening the kept set. Second, and usually worth more:
per-position teacher quality. Positions where the teacher itself was lost are
noise for a distillation run, and this is the column to filter, weight, or
curriculum on.

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
    version, top_p_milli, top_k, depth, n_ids, flags = (int(v) for v in z["target_meta"])
    mode, body_bits, range_milli, head_bits = (int(v) for v in z["target_prob_meta"])
    min_ids, min_p_micro, extra_ids, extra_min, extra_max, has_tail = (
        int(v) for v in z["target_sel_meta"])
    assert version == 3, version

    counts = z["target_counts"].astype(np.int64)
    lse    = z["target_lse"].astype(np.float64)
    table  = z["target_table"]

    # --- ids ---
    blob = z["target_ids"].tobytes()
    bit = 0
    def take(n):
        nonlocal bit
        v = 0
        for _ in range(n):
            v = (v << 1) | ((blob[bit >> 3] >> (7 - (bit & 7))) & 1)
            bit += 1
        return v

    rows = []
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
        rows.append(ids)

    # --- probabilities ---
    if mode == 0:
        probs = z["target_probs"].astype(np.float64)
    elif mode == 1:
        probs = np.exp(-z["target_probs"].astype(np.float64) / 16.0)
    elif mode == 2:
        pblob = z["target_probs"].tobytes()
        pbit = 0
        def take_p(n):
            nonlocal pbit
            v = 0
            for _ in range(n):
                v = (v << 1) | ((pblob[pbit >> 3] >> (7 - (pbit & 7))) & 1)
                pbit += 1
            return v
        def step(w):
            return (range_milli / 1000.0) / ((1 << w) - 1)
        probs = []
        for c in counts:
            R = 1.0
            for j in range(c):
                w = head_bits if j == 0 else body_bits
                p = (2.0 ** (-take_p(w) * step(w))) * R
                probs.append(p)
                R = max(R - p, 0.0)
        probs = np.array(probs)
    else:
        raise ValueError(f"unknown probability mode {mode}")

    out, p = [], 0
    for i, c in enumerate(counts):
        out.append((rows[i], probs[p:p + c], lse[i]))
        p += c
    return out
```

The tail shape, if present, is a separate array:

```python
def tail_shape(z, k):
    """Return p(r) for r in k+1 .. min(k+tail_n, n_vocab), or None."""
    if int(z["target_sel_meta"][5]) == 0:
        return None
    n, degree, vocab = (int(v) for v in z["target_tail_meta"])
    coeffs = z["target_tail_poly"].astype(np.float64)

    def for_position(i):
        c = coeffs[i]
        r0 = k + 1
        r1 = min(k + n, vocab)
        if r1 <= r0:
            return None
        t = (np.log(np.arange(r0, r1 + 1)) - np.log(r0)) / (np.log(r1) - np.log(r0))
        return r0, np.exp(c[0] + t * (c[1] + t * (c[2] + t * c[3]))), float(c[4])

    return for_position
```

`tail_shape(z, k)(i)` gives the rank range, the probabilities over it, and the
exact mass the window holds.

The gold cross-entropy is one mask lookup plus a running index over the values:

```python
def gold_ce(z):
    """Map position index -> CE, for the positions that carry one."""
    if "target_ce_mask" not in z.files:
        return {}
    mask = z["target_ce_mask"].tobytes()
    vals = z["target_ce"].astype(np.float64)
    out, vi = {}, 0
    for i in range(len(z["target_counts"])):
        if (mask[i >> 3] >> (7 - (i & 7))) & 1:
            out[i] = vals[vi]
            vi += 1
    return out
```

For every other position, `CE = -ln p_gold`, with `p_gold` found by matching
`labels[i]` against the position's kept ids.

## Choosing the threshold and the cap

The two options do different jobs. `--target-top-p` decides how much of the
distribution is modelled at all; `--target-top-k` only decides what happens to
positions too flat to reach that mass within the cap. A cap costs bytes only on
the positions that reach it, so raising it is nearly free at a fixed threshold,
while raising the threshold is charged on every position.

Measured on 32,768 positions of a chat corpus with teacher logits reconstructed
through the final softcap, `f16` probabilities:

| setting | ids/pos | p90 | capped | mass on capped | B/pos | 250 M |
| --- | --- | --- | --- | --- | --- | --- |
| `0.80 / 128` | 10.6 | 23 | 2.6 % | 0.728 | 44 | 11.0 GB |
| `0.85 / 128` | 14.1 | 38 | 4.4 % | 0.769 | 57 | 14.2 GB |
| **`0.85 / 256`** | 18.0 | 38 | 2.2 % | 0.793 | 71 | 17.7 GB |
| `0.90 / 128` | 19.3 | 74 | 7.3 % | 0.813 | 76 | 18.9 GB |

The median position keeps 2-3 ids; the mean is set by a minority of flat
positions. At `0.85` the cap of 256 truncates *fewer* positions than the old
`0.80 / 128` default did while modelling more mass, which is why it is now the
default; `0.90 / 128` is worse than `0.85 / 256` on both axes.

Truncation is not silent: a truncated position is exactly one whose
`target_counts` equals the cap while its kept mass, summed from `target_probs`,
stays below `target_top_p`. A consumer can drop those positions, downweight
them, or renormalize over what is there, but it cannot recover the ids that
were never written.

## Storage

At `--target-top-p 0.85`, `--target-top-k 256`, a position costs:

| component | per position |
| --- | --- |
| counts (`u2`) | 2 B |
| logsumexp (`f4`) | 4 B |
| ids, `auto` depth | 1.5-2 B per id |
| probabilities, `f16` | 2 B per id |
| probabilities, `i8` | 1 B per id |
| probabilities, `log` @ 8 | 1 B per id + 1 B (the 16-bit head on rank 0) |
| tail shape (when on) | 10 B, whatever `tail_n` is |
| gold CE (when on) | 0.125 B + 2 B per position whose gold is outside the kept set |

So ~71 B at `f16` (~17.7 GB for 250 M positions), ~53 B at `i8` (~13.2 GB), and
~54 B at `log` @ 8 (~13.5 GB) with about a ninth of the `i8` error. The default
selection (`min_ids 4`, `min_p 0.005`, extra +1 in `[4,48]`) adds a few ids per
sparse position, and the tail shape adds 10 B fixed. `i8` rounds
`-16·ln p` to a byte, which costs ~3 % relative error per probability and maps
everything above 0.97 to `p = 1`; `log` does not saturate, and its error is
uniform in relative terms across ranks.

For reference, the same corpus dumped with hidden states and K/V would be
~6 TB.

## Behaviour worth knowing

* **Deterministic.** Ties in the top-k selection break on the lower id, the
  table is a stable ordering, and the stick encoder's scale is a plain max over
  the exact probabilities, so re-dumping a document is byte-identical — which is
  what the `--dump-verify-writes` comparison depends on. Identical *within* one
  binary: the partition function is summed by an AVX2 loop on machines that have
  it and by a portable loop elsewhere, and the two agree to about 2e-6 in
  `target_lse` (three orders of magnitude inside the `f16` the probabilities are
  stored as, but enough to move a value across a rounding boundary now and
  then); the stick code adds `log2` and `exp2` from the platform libm, which can
  differ by an ulp across libm versions for the same reason. `LLAMA_DUMP_TARGET_LOOP=portable`
  forces the fallback, which is how the test compares the two.
* **`--skip-existing` is array-aware.** A document is skipped only if it carries
  every array the current options write. A version-1 target document is missing
  `target_prob_meta`, so resuming a directory written before version 2 replaces
  those documents (through a verifying write to `<name>.part` followed by a
  rename) instead of mixing two layouts. A file that is missing some is replaced
  the same way, so a verified document is never swapped for an unverified one.
* **The head width is part of the format.** `target_prob_meta[3]` records it so a
  reader never has to assume 16, and changing it would be a new `version`.
* **The partition function is dense.** Truncating terms far below the maximum
  cannot make `target_lse` cheap: on real rows only ~0.06 % of the vocabulary is
  more than 30 nats down, because the sum is set by the bulk of the distribution
  rather than by its head. The cost is therefore one pass of `n_vocab` `exp`
  evaluations per position, and the implementation is built around that — a
  single cache-blocked pass, f64 lanes for the sum, and a top-k that only walks
  the eight lanes whose group can still beat the worst kept candidate. Measured
  at ~41 µs per position at 16 threads on a Threadripper 2950X, against ~200 µs
  for the straight per-element formulation. The probability codec is O(n_ids)
  per document and does not touch that path.
* **Batching is capped.** Every token requests logits, and the context keeps
  `n_vocab · n_outputs_max` floats of host output buffer, so a targets run caps
  the output batch at 2048 tokens.
* **Probabilities are lossy only in storage.** The truncation is decided on
  exact probabilities; the code then rounds the stored value.
