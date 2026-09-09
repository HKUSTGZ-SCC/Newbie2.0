#!/usr/bin/env bash
set -euo pipefail

model_name=testing
ckpt=/path/to/model/checkpoint
config=configs/inference/world_model_interaction.yaml
# Override with, for example: SEED=2026 bash scripts/run_world_model_interaction.sh
seed="${SEED:-123}"
res_dir="/path/to/result/directory"

# These must be configured before Python imports torch.  They make Python hash
# ordering and cuBLAS GEMM selection reproducible across independent launches.
export PYTHONHASHSEED="${seed}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Running deterministic world-model interaction with seed=${seed}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

datasets=(
    "unitree_z1_stackbox"
    "unitree_z1_dual_arm_stackbox"
    "unitree_z1_dual_arm_stackbox_v2"
    "unitree_z1_dual_arm_cleanup_pencils"
    "unitree_g1_pack_camera"
)

n_iters=(12 7 11 8 11)
fses=(4 4 4 4 6)

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    n_iter=${n_iters[$i]}
    fs=${fses[$i]}

    python3 scripts/evaluation/world_model_interaction.py \
    --seed ${seed} \
    --ckpt_path $ckpt \
    --config $config \
    --savedir "${res_dir}/${model_name}/${dataset}" \
    --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 1.0 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --prompt_dir "/path/to/unifolm-world-model-action/examples/world_model_interaction_prompts" \
    --dataset ${dataset} \
    --video_length 16 \
    --frame_stride ${fs} \
    --n_action_steps 16 \
    --exe_steps 16 \
    --n_iter ${n_iter} \
    --timestep_spacing 'uniform_trailing' \
    --guidance_rescale 0.7 \
    --perframe_ae
done
