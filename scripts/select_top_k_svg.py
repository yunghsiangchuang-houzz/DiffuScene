#!/usr/bin/env python3
"""
Select Top-K SVG Samples from Generated Bathroom Scenes

This script analyzes generated bathroom scene samples (boxes_*.npz files),
scores them using the BathroomSceneScorer, and copies the top-K corresponding
SVG files to a 'top_k' subdirectory.

Usage:
    python scripts/select_top_k_svg.py \
        output/v1.2_no_aug/191106600_Bath_US_simple_design_filtered \
        --top_k 5

    # Process multiple folders:
    python scripts/select_top_k_svg.py \
        output/v1.2_no_aug --recursive --top_k 3

    # Custom output directory:
    python scripts/select_top_k_svg.py \
        output/v1.2_no_aug/191106600_Bath_US_simple_design_filtered \
        --output_dir output/v1.2_no_aug/191106600_Bath_US_simple_design_filtered/my_top_k
"""

import os
import sys
import shutil
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene_scorer import (
    BathroomSceneScorer,
    ScoringWeights,
    compute_scene_bounds,
    get_bounds_from_json,
    visualize_scene_to_svg,
    extract_svg_inner,
    create_comparison_grid_svg,
)

# Class labels for 8-class format (without "empty")
CLASS_LABELS_8 = ["vanity", "toilet", "shower", "tub", "floor", "wall", "door", "window"]
FIXTURE_TYPES = {"vanity", "toilet", "shower", "tub"}
ARCH_TYPES = {"floor", "wall", "door", "window"}

# Indices for quick lookup
IDX_VANITY = 0
IDX_TOILET = 1
IDX_SHOWER = 2
IDX_TUB = 3
IDX_FLOOR = 4
IDX_WALL = 5
IDX_DOOR = 6
IDX_WINDOW = 7

FIXTURE_INDICES = {IDX_VANITY, IDX_TOILET, IDX_SHOWER, IDX_TUB}
ARCH_INDICES = {IDX_FLOOR, IDX_WALL, IDX_DOOR, IDX_WINDOW}


def load_samples_from_folder(folder_path: str) -> Tuple[List[Dict], Optional[np.ndarray], List[str]]:
    """
    Load all boxes_*.npz samples from a folder.
    
    Returns:
        samples: List of dicts with translations, sizes, angles, class_labels
        bounds: Scene bounds computed from first sample (for consistent visualization)
        svg_paths: Corresponding SVG file paths
    """
    folder = Path(folder_path)
    npz_files = sorted(folder.glob("boxes_*.npz"), key=lambda p: int(p.stem.split('_')[1]))
    
    if not npz_files:
        raise ValueError(f"No boxes_*.npz files found in {folder_path}")
    
    samples = []
    svg_paths = []
    bounds = None
    
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        sample = {
            'translations': data['translations'],
            'sizes': data['sizes'],
            'angles': data['angles'],
            'class_labels': data['class_labels'],
            'npz_path': str(npz_path),
        }
        samples.append(sample)
        
        # Corresponding SVG path
        svg_path = npz_path.with_suffix('.svg')
        svg_paths.append(str(svg_path))
        
        # Compute bounds from first sample for consistent visualization
        if bounds is None:
            bounds = compute_scene_bounds(
                sample['translations'], sample['sizes'],
                sample['angles'], sample['class_labels']
            )
    
    return samples, bounds, svg_paths


def extract_valid_items(
    translations: np.ndarray,
    sizes: np.ndarray,
    angles: np.ndarray,
    class_labels: np.ndarray,
    arch_range: Tuple[int, int] = (0, 40),
    fixture_range: Tuple[int, int] = (40, 50),
) -> Tuple[Dict, Dict]:
    """
    Extract valid (non-empty) architecture and fixture items from sample.
    
    Data structure:
    - Points 0-39 (arch_range): Architecture space (walls, doors, floors, windows)
    - Points 40-49 (fixture_range): Fixtures space (vanity, toilet, shower, tub)
    - Items with zero size are empty/padding regardless of class label
    
    Returns:
        fixtures: Dict mapping fixture type to list of item dicts
        arch: Dict with 'walls', 'doors', 'floors', 'windows' lists
    """
    # Get class indices
    if class_labels.ndim == 2:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels
    
    # Process angles
    angles_flat = angles
    if angles.ndim == 2 and angles.shape[1] == 1:
        angles_flat = angles.flatten()
    
    # Size threshold for valid items (filter out zero-size padding)
    size_norms = np.linalg.norm(sizes, axis=-1)
    valid_mask = size_norms > 0.01
    
    fixtures = {name: [] for name in FIXTURE_TYPES}
    arch = {'walls': [], 'doors': [], 'floors': [], 'windows': []}
    
    # Extract architecture from arch_range (0-39)
    for i in range(arch_range[0], min(arch_range[1], len(cls_indices))):
        if not valid_mask[i]:
            continue
            
        cls_idx = cls_indices[i]
        if cls_idx >= len(CLASS_LABELS_8):
            continue
            
        label = CLASS_LABELS_8[cls_idx]
        
        # Only accept architecture types in arch range
        if label not in ARCH_TYPES:
            continue
        
        item = {
            'center': translations[i],
            'size': sizes[i],
            'angle': angles_flat[i] if i < len(angles_flat) else 0,
            'bbox': _get_aabb(translations[i], sizes[i] / 2),  # sizes are full, convert to half
        }
        
        if label == 'wall':
            arch['walls'].append(item)
        elif label == 'door':
            arch['doors'].append(item)
        elif label == 'floor':
            arch['floors'].append(item)
        elif label == 'window':
            arch['windows'].append(item)
    
    # Extract fixtures from fixture_range (40-49)
    for i in range(fixture_range[0], min(fixture_range[1], len(cls_indices))):
        if not valid_mask[i]:
            continue
            
        cls_idx = cls_indices[i]
        if cls_idx >= len(CLASS_LABELS_8):
            continue
            
        label = CLASS_LABELS_8[cls_idx]
        
        # Only accept fixture types in fixture range
        if label not in FIXTURE_TYPES:
            continue
        
        item = {
            'center': translations[i],
            'size': sizes[i],
            'angle': angles_flat[i] if i < len(angles_flat) else 0,
            'bbox': _get_aabb(translations[i], sizes[i] / 2),
        }
        
        fixtures[label].append(item)
    
    return fixtures, arch


def _get_aabb(center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    """Get axis-aligned bounding box as [x1, y1, z1, x2, y2, z2]."""
    return np.concatenate([center - half_size, center + half_size])


def _boxes_overlap_2d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute 2D overlap (floor plan) between two AABBs. Uses X and Z axes."""
    lt = np.maximum(bbox1[[0, 2]], bbox2[[0, 2]])
    rb = np.minimum(bbox1[[3, 5]], bbox2[[3, 5]])
    wh = np.maximum(rb - lt, 0)
    return wh[0] * wh[1]


def _distance_2d(center1: np.ndarray, center2: np.ndarray) -> float:
    """2D distance on floor plan (X, Z)."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[2] - center2[2])**2)


def score_single_sample(
    translations: np.ndarray,
    sizes: np.ndarray,
    angles: np.ndarray,
    class_labels: np.ndarray,
    arch_range: Tuple[int, int] = (0, 40),
    fixture_range: Tuple[int, int] = (40, 50),
) -> Tuple[float, Dict]:
    """
    Score a single sample using bathroom layout quality metrics.
    
    Returns:
        score: Quality score (higher is better)
        breakdown: Dict with score components
    """
    fixtures, arch = extract_valid_items(
        translations, sizes, angles, class_labels,
        arch_range=arch_range, fixture_range=fixture_range
    )
    
    walls = arch['walls']
    doors = arch['doors']
    floors = arch['floors']
    
    # Start with base score
    score = 100.0
    breakdown = {}
    
    # === 1. COLLISION DETECTION (fixtures overlapping each other) ===
    all_fixtures = []
    for fixture_list in fixtures.values():
        all_fixtures.extend(fixture_list)
    
    collision_penalty = 0.0
    if len(all_fixtures) >= 2:
        for i in range(len(all_fixtures)):
            for j in range(i + 1, len(all_fixtures)):
                overlap = _boxes_overlap_2d(all_fixtures[i]['bbox'], all_fixtures[j]['bbox'])
                if overlap > 0.01:
                    area_i = all_fixtures[i]['size'][0] * all_fixtures[i]['size'][2]
                    collision_penalty += overlap / max(area_i, 0.01)
    
    score -= collision_penalty * 15.0
    breakdown['collision'] = -collision_penalty * 15.0
    
    # === 2. FIXTURE-WALL COLLISION ===
    wall_collision_penalty = 0.0
    penetration_threshold = 0.10  # Allow 10cm for attachment
    
    for fix in all_fixtures:
        for wall in walls:
            overlap = _boxes_overlap_2d(fix['bbox'], wall['bbox'])
            if overlap > 0:
                # Check penetration depth
                wall_sizes_2d = wall['size'][[0, 2]]
                thin_axis = np.argmin(wall_sizes_2d)
                
                fix_center = fix['center'][[0, 2]][thin_axis]
                wall_center = wall['center'][[0, 2]][thin_axis]
                
                penetration = abs(fix_center - wall_center) - wall_sizes_2d[thin_axis]
                if penetration > penetration_threshold:
                    wall_collision_penalty += (penetration - penetration_threshold)
    
    score -= wall_collision_penalty * 10.0
    breakdown['wall_collision'] = -wall_collision_penalty * 10.0
    
    # === 3. DOOR BLOCKING ===
    door_blocking_penalty = 0.0
    for door in doors:
        door_width = max(door['size'][0], door['size'][2])
        swing_bbox = door['bbox'].copy()
        swing_extension = door_width * 0.8
        
        swing_bbox[0] -= swing_extension * 0.5
        swing_bbox[2] -= swing_extension * 0.5
        swing_bbox[3] += swing_extension * 0.5
        swing_bbox[5] += swing_extension * 0.5
        
        for fix in all_fixtures:
            overlap = _boxes_overlap_2d(fix['bbox'], swing_bbox)
            if overlap > 0.01:
                door_blocking_penalty += overlap / (door_width * door_width + 0.01)
    
    score -= door_blocking_penalty * 10.0
    breakdown['door_blocking'] = -door_blocking_penalty * 10.0
    
    # === 4. BOUNDARY VIOLATION ===
    boundary_penalty = 0.0
    if floors:
        floor_min = np.array([np.inf, np.inf, np.inf])
        floor_max = np.array([-np.inf, -np.inf, -np.inf])
        for floor in floors:
            floor_min = np.minimum(floor_min, floor['bbox'][:3])
            floor_max = np.maximum(floor_max, floor['bbox'][3:])
        
        for fix in all_fixtures:
            fix_bbox = fix['bbox']
            outside_min = np.maximum(floor_min[[0, 2]] - fix_bbox[[0, 2]], 0)
            outside_max = np.maximum(fix_bbox[[3, 5]] - floor_max[[0, 2]], 0)
            boundary_penalty += np.sum(outside_min) + np.sum(outside_max)
    
    score -= boundary_penalty * 10.0
    breakdown['boundary'] = -boundary_penalty * 10.0
    
    # === 5. COMPLETENESS (toilet + vanity required) ===
    completeness_bonus = 0.0
    if fixtures['toilet']:
        completeness_bonus += 5.0
    if fixtures['vanity']:
        completeness_bonus += 5.0
    if not fixtures['toilet'] and not fixtures['vanity']:
        completeness_bonus -= 15.0  # Penalty for missing both
    
    score += completeness_bonus
    breakdown['completeness'] = completeness_bonus
    
    # === 6. WALL ATTACHMENT BONUS ===
    wall_attachment_bonus = 0.0
    attachment_threshold = 0.20
    
    if walls and all_fixtures:
        attached_count = 0
        for fix in all_fixtures:
            for wall in walls:
                # Simple proximity check
                dist = _min_fixture_wall_dist(fix, wall)
                if dist < attachment_threshold:
                    attached_count += 1
                    break
        
        wall_attachment_bonus = (attached_count / len(all_fixtures)) * 5.0
    
    score += wall_attachment_bonus
    breakdown['wall_attachment'] = wall_attachment_bonus
    
    # === 7. VANITY-DOOR PROXIMITY ===
    vanity_door_bonus = 0.0
    if fixtures['vanity'] and doors:
        vanity_center = fixtures['vanity'][0]['center']
        door_center = doors[0]['center']
        dist = _distance_2d(vanity_center, door_center)
        vanity_door_bonus = max(0, (3.0 - dist) / 3.0) * 3.0  # Closer is better, up to 3m
    
    score += vanity_door_bonus
    breakdown['vanity_door'] = vanity_door_bonus
    
    # === 8. CIRCULATION (having all fixture types) ===
    circulation_bonus = 0.0
    fixture_count = sum(1 for fl in fixtures.values() if fl)
    circulation_bonus = (fixture_count / 4.0) * 5.0  # Max 4 fixture types
    
    score += circulation_bonus
    breakdown['circulation'] = circulation_bonus
    
    return max(0.0, score), breakdown


def _min_fixture_wall_dist(fixture: Dict, wall: Dict) -> float:
    """Compute minimum distance from fixture edge to wall face."""
    fix_bbox = fixture['bbox']
    wall_bbox = wall['bbox']
    
    wall_size = wall['size']
    thin_axis = 0 if wall_size[0] < wall_size[2] else 2
    
    if thin_axis == 0:
        dist = min(
            abs(fix_bbox[3] - wall_bbox[0]),
            abs(fix_bbox[0] - wall_bbox[3])
        )
    else:
        dist = min(
            abs(fix_bbox[5] - wall_bbox[2]),
            abs(fix_bbox[2] - wall_bbox[5])
        )
    
    return max(0, dist)


def score_samples(
    samples: List[Dict],
    arch_num_points: int = 40,
    total_points: int = 50,
    weights: Optional[ScoringWeights] = None,
) -> np.ndarray:
    """
    Score all samples.
    
    Args:
        samples: List of sample dicts
        arch_num_points: Number of architecture points (first N points, default 40)
        total_points: Total number of points (default 50)
        weights: Not used (kept for API compatibility)
    
    Returns:
        scores: Array of scores for all samples
    """
    all_scores = np.zeros(len(samples), dtype=np.float32)
    
    arch_range = (0, arch_num_points)
    fixture_range = (arch_num_points, total_points)
    
    for i, sample in enumerate(samples):
        trans = sample['translations']
        sizes = sample['sizes']
        angles = sample['angles']
        class_labels = sample['class_labels']
        
        # Score the sample with proper arch/fixture separation
        score, _ = score_single_sample(
            trans, sizes, angles, class_labels,
            arch_range=arch_range, fixture_range=fixture_range
        )
        all_scores[i] = score
    
    return all_scores


def select_and_copy_top_k(
    folder_path: str,
    output_dir: Optional[str] = None,
    top_k: int = 5,
    arch_num_points: int = 40,
    total_points: int = 50,
    create_grid: bool = True,
    verbose: bool = True,
) -> Tuple[List[int], np.ndarray]:
    """
    Score samples, select top-K, and copy SVG files to output directory.
    
    Args:
        folder_path: Path to folder containing boxes_*.npz and boxes_*.svg files
        output_dir: Output directory for top-K SVGs (default: folder_path/top_k)
        top_k: Number of top samples to select
        arch_num_points: Number of architecture points (first N points, default 40)
        total_points: Total number of points per sample (default 50)
        create_grid: Whether to create a comparison grid SVG
        verbose: Whether to print progress
    
    Returns:
        top_indices: Indices of top-K samples (sorted by score, highest first)
        all_scores: Scores for all samples
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = folder / "top_k"
    else:
        output_dir = Path(output_dir)
    
    # Load samples
    samples, bounds, svg_paths = load_samples_from_folder(folder_path)
    
    if verbose:
        print(f"Loaded {len(samples)} samples from {folder_path}")
    
    # Limit top_k to available samples
    top_k = min(top_k, len(samples))
    
    # Score samples
    all_scores = score_samples(samples, arch_num_points, total_points)
    
    # Get top-K indices
    top_indices = np.argsort(all_scores)[-top_k:][::-1]
    
    if verbose:
        print(f"\nScore statistics:")
        print(f"  Min: {all_scores.min():.2f}, Max: {all_scores.max():.2f}")
        print(f"  Mean: {all_scores.mean():.2f}, Std: {all_scores.std():.2f}")
        print(f"\nTop {top_k} samples:")
        for rank, idx in enumerate(top_indices):
            print(f"  #{rank+1}: Sample {idx} (score: {all_scores[idx]:.2f})")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy top-K SVG files
    copied_files = []
    for rank, idx in enumerate(top_indices):
        src_svg = Path(svg_paths[idx])
        if src_svg.exists():
            # Name with rank: top_1_boxes_3.svg means rank 1 is sample 3
            dst_name = f"top_{rank+1}_boxes_{idx}.svg"
            dst_svg = output_dir / dst_name
            shutil.copy2(src_svg, dst_svg)
            copied_files.append(dst_svg)
            if verbose:
                print(f"  Copied: {src_svg.name} -> {dst_name}")
        else:
            if verbose:
                print(f"  Warning: SVG not found: {src_svg}")
    
    # Also copy corresponding NPZ files
    for rank, idx in enumerate(top_indices):
        src_npz = Path(samples[idx]['npz_path'])
        if src_npz.exists():
            dst_name = f"top_{rank+1}_boxes_{idx}.npz"
            dst_npz = output_dir / dst_name
            shutil.copy2(src_npz, dst_npz)
    
    # Create comparison grid SVG
    if create_grid and len(copied_files) > 0:
        # Get scene bounds from JSON if available
        json_path = folder / "simple_design_filtered.json"
        grid_bounds = get_bounds_from_json(str(json_path))
        if grid_bounds is None:
            grid_bounds = bounds
        
        grid_items = []
        
        # Add ground truth label first if available
        label_svg_path = folder / "simple_design_label.svg"
        if label_svg_path.exists():
            with open(label_svg_path, 'r') as f:
                label_svg = f.read()
            grid_items.append(("GT Label", extract_svg_inner(label_svg)))
        
        # Add top-K samples
        for rank, idx in enumerate(top_indices):
            sample = samples[idx]
            svg_content = visualize_scene_to_svg(
                sample['translations'],
                sample['sizes'],
                sample['angles'],
                sample['class_labels'],
                fixed_bounds=grid_bounds
            )
            score = all_scores[idx]
            grid_items.append((f"#{rank+1} (S{idx}, {score:.1f})", extract_svg_inner(svg_content)))
        
        grid_output = output_dir / "comparison_grid.svg"
        create_comparison_grid_svg(grid_items, str(grid_output))
        if verbose:
            print(f"\nSaved comparison grid to: {grid_output}")
    
    # Save scores summary
    scene_id = folder.name
    scores_data = {
        'scene_id': scene_id,
        'folder': str(folder_path),
        'num_samples': len(samples),
        'top_k': top_k,
        'top_indices': top_indices.tolist(),
        'top_scores': all_scores[top_indices].tolist(),
        'all_scores': all_scores.tolist(),
        'score_stats': {
            'min': float(all_scores.min()),
            'max': float(all_scores.max()),
            'mean': float(all_scores.mean()),
            'std': float(all_scores.std()),
        },
        'copied_files': [str(f) for f in copied_files],
    }
    
    scores_json = output_dir / "scores_summary.json"
    with open(scores_json, 'w') as f:
        json.dump(scores_data, f, indent=2)
    
    if verbose:
        print(f"Saved scores summary to: {scores_json}")
        print(f"\nOutput directory: {output_dir}")
        print(f"Total files copied: {len(copied_files)} SVGs + {len(copied_files)} NPZs")
    
    return top_indices.tolist(), all_scores


def process_folder_recursive(
    base_folder: str,
    output_base: Optional[str] = None,
    top_k: int = 5,
    arch_num_points: int = 40,
    total_points: int = 50,
    verbose: bool = True,
) -> Dict[str, Tuple[List[int], np.ndarray]]:
    """
    Process all scene folders recursively under base_folder.
    
    Looks for folders containing boxes_*.npz files.
    
    Args:
        base_folder: Base folder to search
        output_base: Base output folder (default: each scene's own top_k subfolder)
        top_k: Number of top samples per scene
        arch_num_points: Number of architecture points (first N points, default 40)
        total_points: Total number of points per sample (default 50)
        verbose: Whether to print progress
    
    Returns:
        results: Dict mapping folder path to (top_indices, all_scores)
    """
    base = Path(base_folder)
    results = {}
    
    # Find all folders with boxes_*.npz files
    scene_folders = set()
    for npz_file in base.rglob("boxes_*.npz"):
        scene_folders.add(npz_file.parent)
    
    scene_folders = sorted(scene_folders)
    
    if not scene_folders:
        print(f"No scene folders found under {base_folder}")
        return results
    
    print(f"Found {len(scene_folders)} scene folder(s) to process")
    print("=" * 60)
    
    for i, folder in enumerate(scene_folders):
        print(f"\n[{i+1}/{len(scene_folders)}] Processing: {folder.name}")
        
        try:
            if output_base:
                output_dir = Path(output_base) / folder.name / "top_k"
            else:
                output_dir = None  # Use default (folder/top_k)
            
            top_indices, all_scores = select_and_copy_top_k(
                str(folder),
                output_dir=str(output_dir) if output_dir else None,
                top_k=top_k,
                arch_num_points=arch_num_points,
                total_points=total_points,
                create_grid=True,
                verbose=verbose,
            )
            results[str(folder)] = (top_indices, all_scores)
            
        except Exception as e:
            print(f"  Error processing {folder}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Processed {len(results)} / {len(scene_folders)} scene folder(s)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Select top-K SVG samples from generated bathroom scenes based on quality scores"
    )
    
    parser.add_argument(
        "folder",
        help="Path to folder containing boxes_*.npz files (or base folder with --recursive)"
    )
    
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=5,
        help="Number of top samples to select (default: 5)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="Output directory for top-K SVGs (default: <folder>/top_k)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively process all scene folders under the given folder"
    )
    
    parser.add_argument(
        "--arch_num_points",
        type=int,
        default=40,
        help="Number of architecture points (first N points in each sample, default: 40)"
    )
    
    parser.add_argument(
        "--total_points",
        type=int,
        default=50,
        help="Total number of points per sample (default: 50)"
    )
    
    parser.add_argument(
        "--no_grid",
        action="store_true",
        help="Skip creating comparison grid SVG"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder does not exist: {args.folder}")
        sys.exit(1)
    
    verbose = not args.quiet
    
    if args.recursive:
        # Process multiple folders
        process_folder_recursive(
            str(folder),
            output_base=args.output_dir,
            top_k=args.top_k,
            arch_num_points=args.arch_num_points,
            total_points=args.total_points,
            verbose=verbose,
        )
    else:
        # Process single folder
        try:
            select_and_copy_top_k(
                str(folder),
                output_dir=args.output_dir,
                top_k=args.top_k,
                arch_num_points=args.arch_num_points,
                total_points=args.total_points,
                create_grid=not args.no_grid,
                verbose=verbose,
            )
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

