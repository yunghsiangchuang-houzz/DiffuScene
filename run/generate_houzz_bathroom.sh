#!/bin/bash
cd ./scripts

exp_dir="output/houzz_bathroom_v1/generated_results"
config="../config/uncond/diffusion_houzz_bathroom_v1.yaml"
exp_name="houzz_bathroom_v1"
weight_file="../model/model_40000"
# Create output directory
mkdir -p ../$exp_dir

echo "Generating boxes to $exp_dir with weights from $weight_file"

python generate_houzz_bathroom.py $config ../$exp_dir \
    --weight_file $weight_file \
    --n_samples 6\
    --clip_denoised
