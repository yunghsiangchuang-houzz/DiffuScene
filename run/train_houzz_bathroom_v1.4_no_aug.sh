#!/bin/bash
cd ./scripts
exp_dir="../output/houzz_bathroom_v1.4_no_aug"
config="../config/uncond/diffusion_houzz_bathroom_v1.4_no_aug.yaml"
exp_name="houzz_bathroom_v1.4_no_aug"

# Create output directory
mkdir -p ../$exp_dirs

python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --multi_gpu \
    --with_wandb_logger
