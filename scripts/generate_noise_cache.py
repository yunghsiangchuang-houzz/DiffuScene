"""
Generate noise caches for all scenes in a split (e.g., test set).

This pre-generates all random noise needed for diffusion sampling,
allowing deterministic comparison across multiple checkpoints.

Usage:
    python scripts/generate_noise_cache.py \
        config/uncond/diffusion_houzz_bathroom_v1.2_no_aug_fixed_iou.yaml \
        --split_file config/houzz_bathroom_splits_v1.2.csv \
        --output_dir output/noise_caches \
        --split test \
        --n_samples 6 \
        --base_seed 42
"""

import argparse
import os
import sys
import csv
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from training_utils import load_config
from scene_synthesis.networks.noise_utils import DiffusionNoiseCache


def main():
    parser = argparse.ArgumentParser(
        description="Generate noise caches for deterministic diffusion sampling"
    )
    parser.add_argument(
        "config_file",
        help="Path to the config file (to get network dimensions)"
    )
    parser.add_argument(
        "--split_file",
        default="../config/houzz_bathroom_splits_v1.2.csv",
        help="Path to the split file with scene IDs"
    )
    parser.add_argument(
        "--output_dir",
        default="output/noise_caches",
        help="Directory to save noise caches"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to generate caches for (test, train, val)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=6,
        help="Number of samples per scene"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for noise generation (each scene uses base_seed + scene_index)"
    )
    args = parser.parse_args()

    # Load config to get network dimensions
    config = load_config(args.config_file)
    
    num_points = config["network"]["sample_num_points"]
    point_dim = config["network"]["point_dim"]
    partial_num_points = config["network"].get("partial_num_points", 40)
    num_timesteps = config["network"]["diffusion_kwargs"]["time_num"]
    
    print(f"Network configuration:")
    print(f"  - sample_num_points: {num_points}")
    print(f"  - point_dim: {point_dim}")
    print(f"  - partial_num_points: {partial_num_points}")
    print(f"  - num_timesteps: {num_timesteps}")
    print(f"  - n_samples per scene: {args.n_samples}")
    print(f"  - base_seed: {args.base_seed}")

    # Read scene IDs from split file
    scene_ids = []
    with open(args.split_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                scene_id, split = row[0].strip(), row[1].strip()
                if split == args.split:
                    scene_ids.append(scene_id)
    
    print(f"\nFound {len(scene_ids)} scenes in '{args.split}' split")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate noise cache for each scene
    print(f"\nGenerating noise caches...")
    for idx, scene_id in enumerate(tqdm(scene_ids, desc="Generating caches", unit="scene")):
        # Use scene index as seed offset for reproducibility
        scene_seed = args.base_seed + idx
        
        # Partial boxes shape: (n_samples, partial_num_points, point_dim)
        partial_boxes_shape = (args.n_samples, partial_num_points, point_dim)
        
        # Generate cache
        cache = DiffusionNoiseCache.generate(
            seed=scene_seed,
            batch_size=args.n_samples,
            num_points=num_points,
            point_dim=point_dim,
            num_timesteps=num_timesteps,
            partial_boxes_shape=partial_boxes_shape,
            device='cpu'
        )
        
        # Save cache
        cache_path = os.path.join(args.output_dir, f"{scene_id}.pt")
        cache.save(cache_path)
    
    # Save metadata
    metadata = {
        'config_file': args.config_file,
        'split_file': args.split_file,
        'split': args.split,
        'n_samples': args.n_samples,
        'base_seed': args.base_seed,
        'num_points': num_points,
        'point_dim': point_dim,
        'partial_num_points': partial_num_points,
        'num_timesteps': num_timesteps,
        'scene_ids': scene_ids,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.pt")
    torch.save(metadata, metadata_path)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(scene_ids)} noise caches in: {args.output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

