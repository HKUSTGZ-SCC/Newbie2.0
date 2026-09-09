#!/usr/bin/env bash
set -euo pipefail

# Materialize the exact initial tensors used by the five fixed benchmark cases
# and capture the RNG position immediately before their first sampler call.
# This script intentionally performs no DDIM sampling and writes no videos.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

seed="${SEED:-123}"
checkpoint="${CKPT_PATH:-downloads/UnifoLM-WMA-0-Dual/unifolm_wma_dual.ckpt}"
prompt_dir="${PROMPT_DIR:-examples/compute_benchmark/world_model_interaction_prompts}"
input_cache_dir="${INPUT_CACHE_DIR:-examples/compute_benchmark/preprocessed_inputs}"
prepare_result_dir="${PREPARE_RESULT_DIR:-outputs/compute_benchmark/input_cache_preparation}"

export PYTHONHASHSEED="${seed}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}" >&2
    exit 1
fi

datasets=(
    "unitree_z1_stackbox"
    "unitree_z1_dual_arm_stackbox"
    "unitree_z1_dual_arm_stackbox_v2"
    "unitree_z1_dual_arm_cleanup_pencils"
    "unitree_g1_pack_camera"
)
frame_strides=(4 4 4 4 6)

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    frame_stride="${frame_strides[$i]}"

    python3 scripts/evaluation/world_model_interaction.py \
        --seed "${seed}" \
        --ckpt_path "${checkpoint}" \
        --config configs/inference/world_model_interaction.yaml \
        --prompt_dir "${prompt_dir}" \
        --data_dir "${prompt_dir}" \
        --dataset "${dataset}" \
        --savedir "${prepare_result_dir}/${dataset}" \
        --input_cache_dir "${input_cache_dir}" \
        --write_input_cache --prepare_input_cache_only \
        --bs 1 --height 320 --width 512 \
        --unconditional_guidance_scale 1.0 \
        --ddim_steps 50 --ddim_eta 1.0 \
        --video_length 16 --frame_stride "${frame_stride}" \
        --exe_steps 16 \
        --timestep_spacing uniform_trailing --guidance_rescale 0.7 \
        --perframe_ae
done
