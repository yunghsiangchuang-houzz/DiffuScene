#!/bin/bash
cd ./scripts
exp_dir="../output/houzz_bathroom_v1.2_no_aug_fixed_iou"
config="../config/uncond/diffusion_houzz_bathroom_v1.2_no_aug_fixed_iou.yaml"
exp_name="houzz_bathroom_v1.2_no_aug_fixed_iou"

# Create output directory
mkdir -p ../$exp_dir

python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --multi_gpu \
    --with_wandb_logger
