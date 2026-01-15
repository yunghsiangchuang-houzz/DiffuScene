#!/bin/bash
cd ./scripts

exp_dir="../output/houzz_bathroom_v1.1_no_aug_batch128_fixed_iou10x"
config="../config/uncond/diffusion_houzz_bathroom_v1.1_no_aug_fixed_iou.yaml"
exp_name="houzz_bathroom_v1.1_no_aug_batch128_fixed_iou10x"

# Create output directory
mkdir -p ../$exp_dir

python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --multi_gpu \
    --with_wandb_logger
