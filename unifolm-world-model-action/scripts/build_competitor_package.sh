#!/usr/bin/env bash
set -euo pipefail

# Build a compact, public participant package without organizer tooling,
# checkpoints, generated outputs, raw source prompts, or deployment/training
# code. The output path must not already exist to avoid overwriting a release.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_dir="${1:-${repo_root}/dist/ASC2026-unifolm-wma-competition}"

if [[ -e "${release_dir}" ]]; then
    echo "Refusing to overwrite existing release directory: ${release_dir}" >&2
    exit 1
fi

mkdir -p "${release_dir}"

copy_file() {
    local source_path="$1"
    local destination_path="$2"
    install -D -m 0644 "${source_path}" "${release_dir}/${destination_path}"
}

copy_tree() {
    local source_path="$1"
    local destination_path="$2"
    mkdir -p "${release_dir}/${destination_path}"
    cp -a "${source_path}/." "${release_dir}/${destination_path}/"
}

copy_file "${repo_root}/LICENSE" "LICENSE"
copy_file "${repo_root}/pyproject.toml" "pyproject.toml"
copy_file "${repo_root}/competition/README.md" "README.md"
copy_file "${repo_root}/competition/SUBMISSION.md" "SUBMISSION.md"
copy_file "${repo_root}/competition/SUBMISSION_NOTES.md" "SUBMISSION_NOTES.md"
copy_file "${repo_root}/competition/.gitignore" ".gitignore"
copy_file "${repo_root}/configs/inference/world_model_interaction.yaml" \
    "configs/inference/world_model_interaction.yaml"
# MultiImageObsEncoder loads this metadata through a legacy default path while
# constructing the model. It is inference runtime data, not a training config.
copy_file "${repo_root}/configs/train/meta.json" "configs/train/meta.json"
copy_file "${repo_root}/scripts/run_compute_benchmark.sh" \
    "scripts/run_compute_benchmark.sh"
copy_file "${repo_root}/scripts/create_compute_benchmark_env.sh" \
    "scripts/create_compute_benchmark_env.sh"
copy_file "${repo_root}/scripts/evaluation/world_model_interaction.py" \
    "scripts/evaluation/world_model_interaction.py"
copy_file "${repo_root}/scripts/evaluation/eval_utils.py" \
    "scripts/evaluation/eval_utils.py"
copy_file "${repo_root}/examples/compute_benchmark/manifest.json" \
    "examples/compute_benchmark/manifest.json"
copy_file "${repo_root}/competition/DEVELOPMENT_SET.md" \
    "examples/compute_benchmark/README.md"
copy_tree "${repo_root}/examples/compute_benchmark/preprocessed_inputs" \
    "examples/compute_benchmark/preprocessed_inputs"
copy_tree "${repo_root}/src/unifolm_wma" "src/unifolm_wma"

chmod 0755 "${release_dir}/scripts/run_compute_benchmark.sh"
chmod 0755 "${release_dir}/scripts/create_compute_benchmark_env.sh"

cat <<EOF
Participant package created at: ${release_dir}

Included: inference source, fixed configuration, five public development
caches, and participant documentation.
Excluded: checkpoints, private judge inputs/references, generated outputs,
raw prompts, training, deployment, and organizer tooling.
EOF
