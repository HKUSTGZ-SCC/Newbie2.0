# Private organizer tooling

`evaluate_submission.py` is intentionally excluded from the participant
package. Run it only inside an offline container/VM where the submission runs
as an unprivileged user and the official checkpoint, private caches, and
reference outputs are mounted read-only outside the submission worktree.

The script owns the launcher, measures time in its parent process, and checks
the produced MP4 frame tensors after the timed process exits. It is a judge
orchestrator, not a sandbox: process, GPU, filesystem and network isolation
must be supplied by the container scheduler.

Private input layout:

```text
private_cases/
  manifest.json
  preprocessed_inputs/<dataset>-fs<stride>.pt
private_reference/
  <dataset>/inference/.../*.mp4
```

Create `private_reference/` by running the locked baseline once with precisely
the same private caches and storing the resulting `--savedir` tree. Do not use
the public `ground_truth/` videos: they are demonstrations, not the model
output oracle.

Example, from inside the sealed judge image:

```bash
python3 organizer/evaluate_submission.py \
  --repo https://github.com/team/submission.git \
  --revision <full-commit-sha> \
  --case-dir /judge/private_cases \
  --reference-dir /judge/private_reference \
  --checkpoint /judge/unifolm_wma_dual.ckpt \
  --work-dir /judge/work/team_a \
  --log-dir /judge/logs/team_a \
  --execute
```

Without `--execute`, the script only clones, validates inputs and installs the
judge launcher. This is the safe mode for inspecting a submission.
