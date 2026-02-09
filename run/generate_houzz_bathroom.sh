#!/bin/bash
cd ./scripts

exp_dir="output/houzz_bathroom_v1.4_no_aug/generated_results_model_best_16k"
config="../config/uncond/diffusion_houzz_bathroom_v1.4_no_aug.yaml"
exp_name="houzz_bathroom_v1.4_no_aug"
weight_file="../output/houzz_bathroom_v1.4_no_aug/houzz_bathroom_v1.4_no_aug/model_best"
# Create output directory
mkdir -p ../$exp_dir

echo "Generating boxes to $exp_dir with weights from $weight_file"

python generate_houzz_bathroom.py $config ../$exp_dir \
    --weight_file $weight_file \
    --n_samples 8 \
    --batch_size 64 \
    --inference_split test \
    --clip_denoised 

# Visualize results (from terminal lines 85-86)
python visualize_houzz_results.py --input_dir ../$exp_dir