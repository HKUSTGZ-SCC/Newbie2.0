#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

seed="${SEED:-123}"
checkpoint="${CKPT_PATH:-downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt}"
result_dir="${RESULT_DIR:-outputs/compute_benchmark}"
profile_dir="${PROFILE_DIR:-${result_dir}/profiles}"
input_cache_dir="${INPUT_CACHE_DIR:-examples/compute_benchmark/preprocessed_inputs}"
# The default is the complete source trajectory for every case. Set
# BENCHMARK_N_ITER to an integer only for a local smoke test; it is not the
# competition workload.
benchmark_n_iter="${BENCHMARK_N_ITER:-original}"

export PYTHONHASHSEED="${seed}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    echo "Download it with: hf download unitreerobotics/UnifoLM-WMA-0-Dual unifolm_wma_dual.ckpt --local-dir downloads/UnifoLM-WMA-0-Dual" >&2
    exit 1
fi

mkdir -p "${profile_dir}"

datasets=(
    "unitree_z1_stackbox"
    "unitree_z1_dual_arm_stackbox"
    "unitree_z1_dual_arm_stackbox_v2"
    "unitree_z1_dual_arm_cleanup_pencils"
    "unitree_g1_pack_camera"
)
frame_strides=(4 4 4 4 6)
original_n_iters=(12 7 11 8 11)

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    frame_stride="${frame_strides[$i]}"
    if [[ "${benchmark_n_iter}" == "original" ]]; then
        n_iter="${original_n_iters[$i]}"
    else
        n_iter="${benchmark_n_iter}"
    fi

    python3 scripts/evaluation/world_model_interaction.py \
        --seed "${seed}" \
        --ckpt_path "${checkpoint}" \
        --config configs/inference/world_model_interaction.yaml \
        --dataset "${dataset}" \
        --savedir "${result_dir}/${dataset}" \
        --profile_path "${profile_dir}/${dataset}.json" \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 1.0 \
        --ddim_steps 50 --ddim_eta 1.0 \
        --video_length 16 --frame_stride "${frame_stride}" \
        --exe_steps 16 --n_iter "${n_iter}" \
        --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
        --perframe_ae \
        --input_cache_dir "${input_cache_dir}" --require_input_cache
done
