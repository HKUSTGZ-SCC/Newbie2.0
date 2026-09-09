# Submission and judging contract

## Submission

Submit:

1. a GitHub repository URL;
2. an immutable commit SHA; and
3. a short `SUBMISSION_NOTES.md` describing changed files, required packages,
   and whether the solution uses an engine such as TensorRT.

The submitted repository must retain the standard paths and CLI accepted by
`scripts/evaluation/world_model_interaction.py`. The organizer owns the actual
launcher and overwrites `scripts/run_compute_benchmark.sh` during evaluation.
Changing a copied launcher, printing a fabricated time, or supplying public
case outputs does not affect the score.

## Semantics that must remain unchanged

- Five complete interaction trajectories; no reduced `n_iter`.
- `320x512`, 16 generated frames, 50 DDIM steps, `eta=1.0`, and the official
  frame strides and guidance settings.
- The official checkpoint and judge-provided initial inputs/RNG state.
- Decision and world-model videos for every iteration, written under
  `--savedir/inference/` using the standard path layout.

The public five caches are development-only. Final ranking uses different,
private caches produced with the same schema. Do not assume a case filename,
dataset name, prompt, video id, or public cache content identifies a final
case.

Keep `configs/inference/world_model_interaction.yaml` byte-identical. Its
SHA-256 is part of the cache provenance check; source-level inference
optimizations and prebuilt engines are allowed, but a semantic configuration
change is not a valid optimization for this task.

## Scoring and correctness

The judge runs the submitted commit in an isolated, offline container with one
exclusive GPU. It measures elapsed time outside the submitted process; output
from `TimingCollector`, TensorBoard, or shell `time` is ignored.

Before a time is accepted, the judge compares every produced MP4 frame with a
private reference result. Strict judging uses zero pixel error. If the rules
explicitly allow numerically approximate engines, the published tolerance is
used instead. Missing, extra, malformed, or symlinked result videos fail
correctness.

The default score is the external wall-clock time of the complete fixed
launcher. It includes imports, model construction, checkpoint loading,
inference, and required result writes. The judge may separately publish a
model-compute diagnostic, but it is not trusted for ranking.

## Prohibited shortcuts

- Replay, lookup tables, public-output caches, or branching on a known public
  case to avoid inference.
- Network access, background processes, or cross-submission files.
- Altering judge-mounted inputs/checkpoints or accessing reference outputs.
- Reducing the required generation workload, skipping output generation, or
  returning a fabricated result tree.

The final hidden inputs and external output comparison enforce these rules;
the rules text alone is not the protection.
