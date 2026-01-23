#!/bin/bash
cd ./scripts
exp_dir="../output/houzz_bathroom_v1.3_no_aug_finetuned"
config="../config/uncond/diffusion_houzz_bathroom_v1.3_no_aug_finetuned.yaml"
exp_name="houzz_bathroom_v1.3_no_aug_finetuned"

# Create output directory
mkdir -p ../$exp_dirs

python train_diffusion.py $config $exp_dir \
    --experiment_tag $exp_name \
    --multi_gpu \
    --with_wandb_logger
