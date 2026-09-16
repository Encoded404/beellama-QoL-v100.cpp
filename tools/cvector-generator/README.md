# cvector-generator

This example demonstrates how to generate a control vector using gguf models.

Related PRs:
- [Add support for control vectors](https://github.com/ggml-org/llama.cpp/pull/5970)
- (Issue) [Generate control vector using llama.cpp](https://github.com/ggml-org/llama.cpp/issues/6880)
- [Add cvector-generator example](https://github.com/ggml-org/llama.cpp/pull/7514)
- (Issue) [Improve cvector-generator](https://github.com/ggml-org/llama.cpp/issues/8724) — closed upstream as not planned

## Examples

```sh
# CPU only
./cvector-generator -m ./llama-3.Q4_K_M.gguf

# With GPU
./cvector-generator -m ./llama-3.Q4_K_M.gguf -ngl 99

# With advanced options
./cvector-generator -m ./llama-3.Q4_K_M.gguf -ngl 99 --pca-iter 2000 --pca-batch 100

# Using mean value instead of PCA
./cvector-generator -m ./llama-3.Q4_K_M.gguf --method mean

# To see help message
./cvector-generator -h
# Then, have a look at "cvector" section
```

## Token position selection

The original tool compared `positive` and `negative` **position-by-position over the full
sequence**. That is the CAA-adjacent behaviour only when the two members of a pair are nearly
identical token sequences; for things like "verbose chain vs concise chain" it compares
unrelated tokens.

`--pos-mode` controls which positions contribute:

| mode | behaviour |
|---|---|
| `all` (default) | position-by-position, over the range where **both** sequences have real tokens |
| `last` | only the final token of each sequence — this is what CAA and ASC extract |
| `range` | positions `[START, END)` of each sequence, see `--pos-range` |

`--pos-mode last` is the one to use when reproducing published steering-vector work, and it is
what makes the tool usable for chain-of-thought traces of differing length.

Both members of a pair are decoded **unpadded**; no padding token is ever introduced, and
positions outside the selected range are never read. (Previously the shorter member was padded
with spaces, which compared real tokens against padding and then tried to filter the resulting
garbage out afterwards.)

## Layer indexing

Emitted vectors are named `direction.N` and are applied to the residual stream **at the end of
layer N**. The tool emits `direction.1 .. direction.M` where `M` is the highest layer it could
capture. Layer 0 is never steerable: `llama_adapter_cvec` leaves `tensors[0]` null, `apply()`
starts at layer 1, and the loader rejects a `direction.0` tensor outright.

Each `direction.N` is extracted from the same activation it is later applied to (`l_out-N`), so
the layer index in the file matches the layer index at inference.

The capture callback keys off the layer index parsed from the tensor **name** (`l_out-<il>`)
rather than relying on capture order. That matters because architectures differ in whether the
final layer is capturable:

- Architectures that row-select their last layer down to the output positions (`inp_out_ids`,
  so `ne[1]` becomes 1 instead of the sequence length) cannot contribute it. The tool detects
  this and prints a note; this is why the previous version's fixed `n_layer-2` bound existed.
- Architectures that do **not** row-select it there (Gemma 4, Qwen3-Next, Qwen3.5, and others
  with `embeddings_nextn` support) can, and the tool now emits `direction.(n_layer-1)` for them
  instead of silently dropping the layer.

All layers between 1 and `M` are required; only the final layer may be absent. A missing interior
layer is an error rather than a silently truncated vector set.

## Provenance

Every written file carries metadata describing how it was produced:

- `controlvector.source_model_path` / `source_model_sha256` / `source_model_desc` / `source_model_size`
- `controlvector.positive_file` / `positive_sha256`, `negative_file` / `negative_sha256`
- `controlvector.method`, `pos_mode`, `pos_start`, `pos_end`, `n_pairs`, `n_cols`, `n_samples`
- `controlvector.n_components`, `component_index`, `pca_batch`, `pca_iterations`
- `controlvector.n_embd`, `layer_first`, `layer_scale`

`layer_scale` holds one value per emitted layer: the eigenvalue for `--method pca`, or the
pre-normalization norm of the mean vector for `--method mean`. Because emitted vectors are
L2-normalized, the scale reported by `--control-vector-scaled` is dimensionless; `layer_scale`
is what lets you compare across layers and models.

Use `--no-hash` to skip hashing the model file (it is read in full, which takes a few seconds on
a multi-GB gguf).

## Multiple components and null calibration

```sh
# top-3 principal components per layer; component 1 goes to control_vector.gguf,
# components 2+ to control_vector_c2.gguf, control_vector_c3.gguf
./cvector-generator -m ./model.gguf --pos-mode last --n-components 3

# calibrate the eigenvalue spectrum against a pairing-breaking null
./cvector-generator -m ./model.gguf --pos-mode range --pos-range 0 512 \
    --n-components 2 --null-permutations 32 --stats
```

`--null-permutations N` recomputes the spectrum `N` times with the pos/neg pairing broken
(positive `i` is differenced against negative `π(i)`), and reports the null distribution of
`lambda1/trace` and `lambda1/lambda2`. It uses the **same iteration budget** as the main run
(comparing a statistic computed under one budget against the same statistic under another is not
a valid comparison), so it costs about `N` times as much.

**Read `mean_frac` first, and read the null for what it actually tests.** Two things are easy to
get wrong here:

- **`mean_frac` is the quantity that matters for a steering vector.** It is the fraction of the
  (uncentered) scatter mass lying along the mean difference direction — i.e. how much of
  pos−neg is a single shared direction. It is **permutation-invariant**, so the null cannot see
  it at all.
- **The null therefore tests something narrower:** *does the identity pairing add top-eigenvalue
  mass beyond arbitrary pairings of the same negatives?* A strong shared direction gives
  `real == null` (verified on synthetic data with unambiguous structure). So `real ≈ null` means
  "no pairing-specific structure", **not** "no structure". Conversely a large `mean_frac` with
  `real ≈ null` is a perfectly coherent and expected combination.

Other caveats:

- **The null must break the pairing.** Permuting sample order, or flipping pos↔neg, leaves the
  scatter spectrum *exactly* invariant, so those make degenerate nulls that will "confirm"
  anything.
- **Small pair counts make weak nulls.** With four prompt pairs there are only 23 distinct
  non-identity permutations, so the smallest attainable one-sided p is ~0.04, and the reported
  sd carries a large relative error. Use hundreds of pairs for a meaningful test.
- **The statistic is a mean ± sd, not a p-value**, and it is not corrected for the number of
  layers tested.
- **Power iteration under-reports the eigenvalue when it has not converged.** The tool prints the
  worst terminal residual and a warning if it exceeds the tolerance; raise `--pca-iter` if you
  see it.

`--null-permutations` requires at least 2 pairs and every pair to contribute the same number of
positions, so it must be combined with `--pos-mode last` or a fixed `--pos-range`.

## Notes on the estimator

- The reduction runs on the **uncentered** scatter matrix `S = sum_s x_s x_s^T`. A principal
  component is only the top principal component of the covariance when the difference vectors
  are mean-zero; with an uncentered matrix the mean direction is folded into the top component.
  `mean_frac` tells you how much.
- An eigenvector is defined only up to sign. The tool orients each direction so that
  `dot(v, mean(pos) - mean(neg)) > 0`; without this the emitted sign was a coin flip. If the mean
  shift is numerically zero the sign is left as-is and a warning is printed.
- The eigenvalue is computed as the exact Rayleigh quotient `v^T S v` (one device matvec plus a
  host dot product), not read back from inside the power-iteration graph. Reading a non-output
  tensor back after `ggml_backend_graph_compute` is unsound: ggml's allocator decides liveness
  from use counts, so a tensor not marked with `ggml_set_output()` is considered dead once its
  last consumer has run and a later tensor is assigned the same buffer.
- The power iteration is started from a **fixed** seed so that repeated runs produce
  bit-identical files (given the same thread count and backend).
- `--n-components` extracts the top-k components by Gram deflation (`S <- S - λ v v^T`). Note
  that deflating by the Rayleigh quotient is only the true rank-1 update once the iteration has
  converged; with a small `--pca-iter` on a near-degenerate spectrum the extracted eigenvalues
  are not guaranteed to be ordered.

## Tips and tricks

If you have multiple lines per prompt, you can escape the newline character (change it to `\n`). For example:

```
<|im_start|>system\nAct like a person who is extremely happy.<|im_end|>
<|im_start|>system\nYou are in a very good mood today<|im_end|>
```

Example to use output file with `llama-cli`:

(Tips: The control vector works better when apply to layers higher than 10)

```sh
./llama-cli -m ./llama-3.Q4_K_M.gguf -p "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSing a song<|im_end|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" --special --control-vector-scaled ./control_vector.gguf 0.8 --control-vector-layer-range 10 31
```
