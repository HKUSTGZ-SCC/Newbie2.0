# ASC 2026 UnifoLM-WMA Optimization Challenge

This directory supplies the documents copied into the participant package. The
participant package is a development kit, not the final judge environment.
It includes `configs/train/meta.json` only because the upstream observation
encoder reads that legacy metadata path while constructing the inference model;
it is not a training configuration.

## Development benchmark

The public development set contains five fixed input caches. It is useful for
checking that an optimization still produces the complete interaction videos:

```bash
bash scripts/run_compute_benchmark.sh
```

The launcher uses the original workload: five cases with `12/7/11/8/11`
interaction rounds. The cache fixes the initial tensors and the RNG state just
before the first DDIM sampler invocation. It does not cache CLIP/VAE encoding,
DDIM sampling, later interaction rounds, or model weights.

The printed `Pure model compute time` is a local diagnostic only. It is not a
competition score because participant code can modify it.

## What is submitted

Submit a GitHub repository URL and an immutable commit SHA. Retain this
interface, even if the implementation behind it changes:

```text
scripts/evaluation/world_model_interaction.py
configs/inference/world_model_interaction.yaml
scripts/run_compute_benchmark.sh
```

The judge replaces `scripts/run_compute_benchmark.sh` after checkout. It then
provides hidden input caches and a read-only official checkpoint. Your code
must accept the command-line arguments used by the supplied launcher and write
the normal `inference/` MP4 result tree below `--savedir`.

Do not commit checkpoints, generated videos, public-output caches, virtual
environments, or compiled artifacts tied to a public input. See
[`SUBMISSION.md`](SUBMISSION.md) for the complete contract.
