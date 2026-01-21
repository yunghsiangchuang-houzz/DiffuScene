#!/bin/bash
cd ./scripts
exp_dir="../output/houzz_bathroom_v1.3_aug_partial_condition_iou"
config="../config/uncond/diffusion_houzz_bathroom_v1.3_aug_partial_condition_iou.yaml"
exp_name="houzz_bathroom_v1.3_aug_partial_condition_iouu"

# Create output directory
mkdir -p ../$exp_dirs

python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --multi_gpu \
    --with_wandb_logger
