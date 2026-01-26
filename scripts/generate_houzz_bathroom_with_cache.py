"""
Generate scenes using a previously trained model with deterministic noise caches.

This script loads pre-generated noise caches for each scene, enabling
fair comparison across multiple checkpoints.

Usage:
    # First, generate noise caches:
    python scripts/generate_noise_cache.py \
        config/uncond/diffusion_houzz_bathroom_v1.2_no_aug_fixed_iou.yaml \
        --output_dir output/noise_caches --split test

    # Then, run generation with cache:
    python scripts/generate_houzz_bathroom_with_cache.py \
        config/uncond/diffusion_houzz_bathroom_v1.2_no_aug_fixed_iou.yaml \
        output/my_checkpoint_results \
        --weight_file output/my_checkpoint.pt \
        --noise_cache_dir output/noise_caches \
        --n_samples 6
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from training_utils import load_config
from utils import floor_plan_from_scene

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.networks import build_network
from scene_synthesis.networks.noise_utils import DiffusionNoiseCache


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes with deterministic noise caches for multi-checkpoint comparison"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--split_file",
        default=None,
        help="The split file to be used for inference (optional, overrides config)"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--noise_cache_dir",
        required=True,
        help="Path to directory containing pre-generated noise caches"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=6,
        help="The number of samples to generate for each scene"
    )
    parser.add_argument(
        "--use_text_description",
        action="store_true",
        help="if use text description"
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="if clip_denoised"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning (optional, for single scene)"
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="if use ddim"
    )
    parser.add_argument(
        "--inference_split",
        default="test",
        help="The split to be used for inference"
    )
    parser.add_argument(
        "--scene_id_filter_file",
        default=None,
        help="Path to text file containing scene IDs to process (one per line). If provided, only these scenes will be generated."
    )

    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Verify noise cache directory exists
    if not os.path.exists(args.noise_cache_dir):
        print(f"ERROR: Noise cache directory not found: {args.noise_cache_dir}")
        print("Please run generate_noise_cache.py first to create noise caches.")
        sys.exit(1)

    config = load_config(args.config_file)
    
    # Fix relative paths in config (resolve relative to scripts/ directory, as the config expects)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve dataset_directory path
    if config["data"].get("dataset_directory", "").startswith(".."):
        config["data"]["dataset_directory"] = os.path.normpath(
            os.path.join(scripts_dir, config["data"]["dataset_directory"])
        )
    
    # Resolve train_stats_file path
    if config["network"]["diffusion_kwargs"].get("train_stats_file", "").startswith(".."):
        config["network"]["diffusion_kwargs"]["train_stats_file"] = os.path.normpath(
            os.path.join(scripts_dir, config["network"]["diffusion_kwargs"]["train_stats_file"])
        )
    
    # Resolve annotation_file (split file) path from config
    if config["data"].get("annotation_file", "").startswith(".."):
        config["data"]["annotation_file"] = os.path.normpath(
            os.path.join(scripts_dir, config["data"]["annotation_file"])
        )
    
    # Override splits file path only if explicitly provided
    if args.split_file is not None:
        config["data"]["annotation_file"] = args.split_file
        print(f"Overriding split file with: {args.split_file}")
    
    print(f"Using split file: {config['data']['annotation_file']}")
    print(f"Using train stats: {config['network']['diffusion_kwargs'].get('train_stats_file', 'N/A')}")
    
    # Override for inference
    if 'text' in config["data"]["encoding_type"]:
        if 'textfix' not in config["data"]["encoding_type"]:
            config["data"]["encoding_type"] = config["data"]["encoding_type"].replace('text', 'textfix')

    if "no_prm" not in config["data"]["encoding_type"]:
        print('NO PERM AUG in test')
        config["data"]["encoding_type"] = config["data"]["encoding_type"] + "_no_prm"
    print('encoding type :', config["data"]["encoding_type"])

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=[args.inference_split]
        ),
        split=[args.inference_split]
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Determine which scene(s) to process
    given_scene_id = None
    given_scene_id_str = None
    if args.scene_id:
        if args.scene_id.isdigit():
            try:
                matching_indices = [idx for idx, tag in enumerate(raw_dataset._tags) if args.scene_id in tag]
                if matching_indices:
                    given_scene_id = matching_indices[0]
                    given_scene_id_str = args.scene_id
                    print(f"Found scene ID '{args.scene_id}' at index {given_scene_id}")
                else:
                    given_scene_id = int(args.scene_id)
                    print(f"Using fixed scene at index {given_scene_id}")
            except Exception:
                given_scene_id = int(args.scene_id)
                print(f"Using fixed scene at index {given_scene_id}")
        else:
            try:
                matching_indices = [idx for idx, tag in enumerate(raw_dataset._tags) if args.scene_id in tag]
                if matching_indices:
                    given_scene_id = matching_indices[0]
                    given_scene_id_str = args.scene_id
                    print(f"Found scene ID '{args.scene_id}' at index {given_scene_id}")
                else:
                    print(f"Scene ID '{args.scene_id}' not found in dataset!")
            except Exception as e:
                print(f"Error searching for scene ID: {e}")

    print(f"Generating samples for {len(dataset)} sequences with cached noise...")
    import shutil

    # Load scene ID filter if provided
    scene_id_filter = None
    filtered_scene_indices = None
    if args.scene_id_filter_file:
        if not os.path.exists(args.scene_id_filter_file):
            print(f"ERROR: Scene ID filter file not found: {args.scene_id_filter_file}")
            sys.exit(1)
        
        scene_id_filter = set()
        with open(args.scene_id_filter_file, 'r') as f:
            for line in f:
                scene_id = line.strip()
                if scene_id:  # Skip empty lines
                    scene_id_filter.add(scene_id)
        
        print(f"Loaded {len(scene_id_filter)} scene IDs from filter file: {args.scene_id_filter_file}")
        
        # Pre-filter: find indices of scenes that match the filter
        filtered_scene_indices = []
        for idx in range(len(dataset)):
            try:
                scene_id = raw_dataset._tags[idx]
            except Exception:
                scene_id = f"scene_{idx}"
            
            if scene_id in scene_id_filter:
                filtered_scene_indices.append(idx)
        
        print(f"Found {len(filtered_scene_indices)} matching scenes in dataset (out of {len(dataset)} total)")
        print(f"Will process only these {len(filtered_scene_indices)} scenes.")

    # If a specific scene ID is given, only run for that one scene
    if given_scene_id is not None:
        scene_range = [given_scene_id]
        print(f"Running for single scene: index {given_scene_id}")
    elif filtered_scene_indices is not None:
        scene_range = filtered_scene_indices
        print(f"Running for {len(scene_range)} filtered scenes")
    else:
        scene_range = range(len(dataset))

    # Track missing caches
    missing_caches = []
    
    for i in tqdm(scene_range, desc="Generating scenes", unit="scene"):
        scene_idx = i
        samples = dataset[scene_idx]
        
        try:
            actual_scene_id = raw_dataset._tags[scene_idx]
        except Exception:
            actual_scene_id = f"scene_{scene_idx}"
        
        tqdm.write(f"Processing scene {actual_scene_id} (index {scene_idx})")
        
        # Load noise cache for this scene
        cache_path = os.path.join(args.noise_cache_dir, f"{actual_scene_id}.pt")
        if not os.path.exists(cache_path):
            tqdm.write(f"  WARNING: Noise cache not found: {cache_path}")
            missing_caches.append(actual_scene_id)
            continue
        
        noise_cache = DiffusionNoiseCache.load(cache_path, device=str(device))
        tqdm.write(f"  Loaded noise cache (seed={noise_cache.seed}, batch_size={noise_cache.batch_size})")
        
        # Handle batch size mismatch
        if noise_cache.batch_size < args.n_samples:
            tqdm.write(f"  ERROR: Cache has {noise_cache.batch_size} samples but {args.n_samples} requested!")
            tqdm.write(f"         Regenerate cache with --n_samples>={args.n_samples}")
            missing_caches.append(actual_scene_id)
            continue
        elif noise_cache.batch_size > args.n_samples:
            tqdm.write(f"  Slicing cache from {noise_cache.batch_size} to {args.n_samples} samples")
            noise_cache = noise_cache.slice(args.n_samples)
        
        # Prepare conditioned partial boxes
        room_layout_size = config["data"].get("room_layout_size", "64,64").split(",")
        room_layout_h, room_layout_w = int(room_layout_size[0]), int(room_layout_size[1])
        room_mask = torch.zeros((1, 1, room_layout_h, room_layout_w), device=device)
        
        partial_num_points = config["network"].get("partial_num_points", 40)
        translations = samples['translations'][:partial_num_points]
        sizes = samples['sizes'][:partial_num_points]
        angles = samples['angles'][:partial_num_points]
        class_labels = samples['class_labels'][:partial_num_points]
        
        comp_tensor = lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x
        partial_boxes = torch.cat([
            comp_tensor(translations),
            comp_tensor(sizes),
            comp_tensor(angles),
            comp_tensor(class_labels)
        ], dim=-1).unsqueeze(0).to(device)  # (1, 40, 17)

        # Output directory
        sample_dir = os.path.join(args.output_directory, actual_scene_id)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        if args.use_text_description:
            text_description = "The bathroom has one vanity, one toilet, and one tub. "
        else:
            text_description = None

        tqdm.write(f"  Text description: {text_description}")

        # Generate samples in a BATCH with noise cache
        tqdm.write(f"  Generating {args.n_samples} samples in batch with cached noise...")
        with torch.no_grad():
            # Expand inputs for batch processing
            room_mask_batch = room_mask.expand(args.n_samples, -1, -1, -1)  # (n_samples, 1, H, W)
            partial_boxes_batch = partial_boxes.expand(args.n_samples, -1, -1)  # (n_samples, num_partial, 17)
            
            # Single-path generation with cached noise (dual_path_compare=False)
            bbox_params = network.complete_scene(
                room_mask=room_mask_batch,
                num_points=config["network"]["sample_num_points"],
                point_dim=config["network"]["point_dim"],
                partial_boxes=partial_boxes_batch,
                text=text_description,
                batch_size=args.n_samples,
                device=device,
                clip_denoised=args.clip_denoised,
                batch_seeds=None,  # Not needed when using noise_cache
                ddim=args.ddim,
                keep_empty=True,
                dual_path_compare=False,
                noise_cache=noise_cache  # <-- Use cached noise!
            )
        
        # Post-process
        boxes = dataset.post_process(bbox_params)
        
        # Save each sample from the batch
        for j in range(args.n_samples):
            sample_boxes = {}
            for k, v in boxes.items():
                if isinstance(v, torch.Tensor):
                    sample_boxes[k] = v[j].cpu().numpy()
                else:
                    sample_boxes[k] = v[j]
            npz_path = os.path.join(sample_dir, f"boxes_{j}.npz")
            np.savez(npz_path, **sample_boxes)
        
        tqdm.write(f"  Saved {args.n_samples} samples to {sample_dir}")

        # Copy assets
        if "_simple_design_filtered" in actual_scene_id:
            source_folder_name = actual_scene_id.replace("_simple_design_filtered", "")
            source_dir = os.path.join(config["data"]["dataset_directory"], "..", source_folder_name)
            for filename in ["simple_design_filtered.json", "simple_design_label.svg", "simple_design.json"]:
                source_file = os.path.join(source_dir, filename)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, os.path.join(sample_dir, filename))
    
    # Report missing caches
    if missing_caches:
        print(f"\nWARNING: {len(missing_caches)} scenes skipped due to missing noise caches:")
        for scene_id in missing_caches[:10]:
            print(f"  - {scene_id}")
        if len(missing_caches) > 10:
            print(f"  ... and {len(missing_caches) - 10} more")
    
    # Report summary
    if filtered_scene_indices is not None:
        processed_count = len(filtered_scene_indices) - len(missing_caches)
        print(f"\nSummary:")
        print(f"  Requested scenes: {len(scene_id_filter)}")
        print(f"  Found in dataset: {len(filtered_scene_indices)}")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Missing caches: {len(missing_caches)}")
    
    print("\nDone generation with cached noise.")


if __name__ == "__main__":
    main(sys.argv[1:])

