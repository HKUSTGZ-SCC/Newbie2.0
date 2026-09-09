#!/usr/bin/env bash
set -euo pipefail

# Creates an isolated environment for the fixed UnifoLM-WMA benchmark. Keep it
# outside source-controlled files; it contains several GB of CUDA packages.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_prefix="${UNIFOLM_BENCH_ENV:-${repo_root}/.conda/compute-benchmark}"

if [[ ! -x "${env_prefix}/bin/python" ]]; then
    conda create --yes --prefix "${env_prefix}" python=3.10.18 pip
else
    echo "Resuming existing environment: ${env_prefix}"
fi

# The H200 driver is backward-compatible with CUDA 12.1. These versions match
# the upstream pyproject.toml and the xFormers 0.0.27 wheels.
conda run --prefix "${env_prefix}" pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 torchvision==0.18.1

cd "${repo_root}"
conda run --prefix "${env_prefix}" pip install -e .

echo
echo "Environment created: ${env_prefix}"
echo "Run the benchmark with:"
echo "  conda run --prefix ${env_prefix} bash scripts/run_compute_benchmark.sh"
