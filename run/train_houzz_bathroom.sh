#!/bin/bash
cd ./scripts

exp_dir="/output/houzz_bathroom_v1"
config="../config/uncond/diffusion_houzz_bathroom_v1.yaml"
exp_name="houzz_bathroom_v1"

# Create output directory
mkdir -p ../$exp_dir

python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --with_wandb_logger
