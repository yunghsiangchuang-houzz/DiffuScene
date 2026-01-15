#!/bin/bash
cd ./scripts

exp_dir="output/houzz_bathroom_v1.1_no_aug_batch128_fixed_iou/generated_results_model_40000_same_seed"
config="../config/uncond/diffusion_houzz_bathroom_v1.1_no_aug_fixed_iou.yaml"
exp_name="houzz_bathroom_v1.1_no_aug_batch128_fixed_iou"
weight_file="../output/houzz_bathroom_v1.1_no_aug_batch128_fixed_iou/houzz_bathroom_v1.1_no_aug_batch128_fixed_iou/model_40000"
# Create output directory
mkdir -p ../$exp_dir

echo "Generating boxes to $exp_dir with weights from $weight_file"

python generate_houzz_bathroom.py $config ../$exp_dir \
    --weight_file $weight_file \
    --n_samples 10 \
    --clip_denoised 