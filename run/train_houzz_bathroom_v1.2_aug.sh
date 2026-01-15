#!/bin/bash
cd ./scripts
# if train houzz_bathroom_v1.2_no_aug : remember to set the iou function
exp_dir="../output/houzz_bathroom_v1.2_aug_from_23k"
config="../config/uncond/diffusion_houzz_bathroom_v1.2_aug.yaml"
exp_name="houzz_bathroom_v1.2_aug_from_23k"

# Create output directory
mkdir -p ../$exp_dir


python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --multi_gpu \
    --with_wandb_logger
