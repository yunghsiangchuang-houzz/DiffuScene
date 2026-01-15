#!/bin/bash
cd ./scripts

exp_dir="output/houzz_bathroom_text_v1/generated_results_6200"
config="../config/text/diffusion_houzz_bathroom_text_v1.yaml"
exp_name="houzz_bathroom_text_v1"
weight_file="../output/houzz_bathroom_text_v1/houzz_bathroom_text_v1/model_06200"
# Create output directory
mkdir -p ../$exp_dir

echo "Generating boxes to $exp_dir with weights from $weight_file"

python generate_houzz_bathroom.py $config ../$exp_dir \
    --weight_file $weight_file \
    --n_samples 6\
    --clip_denoised \
    --use_text_description
