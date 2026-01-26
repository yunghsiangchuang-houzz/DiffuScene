#!/bin/bash
cd ./scripts

# Configuration for the 3 model versions to evaluate
declare -a versions=("v1.3_aug_fixed_iou")

# Common parameters
n_samples=10
scene_filter="../output/model_eval_2/selected_scene_ids.txt"
noise_cache_dir="../output/noise_caches"
inference_split="test"

echo "=========================================="
echo "Model Evaluation on Selected 30 Scenes"
echo "=========================================="
echo "Noise cache dir: $noise_cache_dir"
echo "Scene filter: $scene_filter"
echo "N samples: $n_samples"
echo ""

# Loop through each version
for version in "${versions[@]}"; do
    echo "=========================================="
    echo "Processing: $version"
    echo "=========================================="
    
    # Set paths based on version
    config="../config/uncond/diffusion_houzz_bathroom_${version}.yaml"
    output_dir="../output/model_eval_2/${version}"
    weight_file="../output/houzz_bathroom_${version}/houzz_bathroom_${version}/model_best"
    
    # Create output directory
    mkdir -p $output_dir
    
    # Check if weight file exists
    if [ ! -f "$weight_file" ]; then
        echo "ERROR: Weight file not found: $weight_file"
        echo "Skipping $version"
        echo ""
        continue
    fi
    
    echo "Config: $config"
    echo "Output: $output_dir"
    echo "Weights: $weight_file"
    echo ""
    
    # Run generation with cached noise
    python generate_houzz_bathroom_with_cache.py $config $output_dir \
        --weight_file $weight_file \
        --noise_cache_dir $noise_cache_dir \
        --n_samples $n_samples \
        --inference_split $inference_split \
        --scene_id_filter_file $scene_filter \
        --clip_denoised
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully completed: $version"
    else
        echo ""
        echo "✗ Failed: $version"
    fi
    
    echo ""
    echo "------------------------------------------"
    echo ""
done

echo "=========================================="
echo "All model evaluations completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
for version in "${versions[@]}"; do
    echo "  - output/model_eval/${version}/"
done
echo ""

