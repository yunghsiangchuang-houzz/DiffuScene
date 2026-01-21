#!/bin/bash
cd ./scripts

exp_dir="output/houzz_bathroom_v1.2_no_aug/generated_results_model_test_set_51000"
config="../config/uncond/diffusion_houzz_bathroom_v1.2_no_aug.yaml"
exp_name="houzz_bathroom_v1.2_no_aug"
weight_file="../output/houzz_bathroom_v1.2_no_aug/houzz_bathroom_v1.2_no_aug/model_51000"
# Create output directory
mkdir -p ../$exp_dir

echo "Generating boxes to $exp_dir with weights from $weight_file"

python generate_houzz_bathroom.py $config ../$exp_dir \
    --weight_file $weight_file \
    --n_samples 6 \
    --batch_size 128 \
    --clip_denoised 

# Visualize results (from terminal lines 85-86)
python visualize_houzz_results.py --input_dir ../$exp_dirs