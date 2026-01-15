#!/bin/bash
cd ./scripts

exp_dir="../output/houzz_bathroom_text_v1_batch128"
config="../config/text/diffusion_houzz_bathroom_text_v1.yaml"
exp_name="houzz_bathroom_text_v1_batch128"

# Create output directory
mkdir -p ../$exp_dir

python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --multi_gpu \
    --with_wandb_logger

