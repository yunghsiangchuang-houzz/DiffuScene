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
from tqdm import tqdm

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


def load_samples_from_folder(folder_path: str, use_postprocess: bool = False) -> Tuple[List[Dict], Optional[np.ndarray], List[str]]:
    """
    Load all boxes_*.npz or boxes_*.postprocess.npz samples from a folder.
    
    Args:
        folder_path: Path to folder containing npz files
        use_postprocess: If True, load boxes_*.postprocess.npz files instead of boxes_*.npz
    
    Returns:
        samples: List of dicts with translations, sizes, angles, class_labels
        bounds: Scene bounds computed from first sample (for consistent visualization)
        svg_paths: Corresponding SVG file paths
    """
    folder = Path(folder_path)
    
    if use_postprocess:
        npz_pattern = "boxes_*.postprocess.npz"
    else:
        npz_pattern = "boxes_*.npz"
    
    # Sort by extracting the numeric index from the filename
    # For boxes_0.npz: stem='boxes_0' -> split('_')[1]='0'
    # For boxes_0.postprocess.npz: stem='boxes_0.postprocess' -> split('_')[1]='0.postprocess' -> split('.')[0]='0'
    def extract_index(p):
        parts = p.stem.split('_')
        if len(parts) >= 2:
            # Take the second part and remove any suffix after '.'
            num_str = parts[1].split('.')[0]
            return int(num_str)
        return 0
    
    npz_files = sorted(folder.glob(npz_pattern), key=extract_index)
    
    # Strictly filter to avoid double-counting
    if use_postprocess:
        # Should already be filtered by glob pattern "boxes_*.postprocess.npz"
        # but let's be safe
        npz_files = [f for f in npz_files if '.postprocess.' in f.name]
    else:
        # Filter out anything that has '.postprocess.' in the name
        # to ensure boxes_0.npz.postprocess.npz is NOT included
        npz_files = [f for f in npz_files if '.postprocess.' not in f.name]
    
    if not npz_files:
        raise ValueError(f"No {npz_pattern} files found in {folder_path}")
    
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
        if use_postprocess:
            # Handle styles like:
            # 1. boxes_0.npz.postprocess.npz -> boxes_0.svg.postprocess.svg
            # 2. boxes_0.postprocess.npz -> boxes_0.postprocess.svg
            name = npz_path.name
            if '.npz.postprocess.npz' in name:
                svg_name = name.replace('.npz.postprocess.npz', '.svg.postprocess.svg')
            elif '.postprocess.npz' in name:
                svg_name = name.replace('.postprocess.npz', '.postprocess.svg')
            else:
                svg_name = name.replace('.npz', '.svg') # Fallback
            
            svg_path = npz_path.parent / svg_name
            
            # Fallback attempts if the direct replacement doesn't exist
            if not svg_path.exists():
                # Try replacing .npz with .svg anywhere
                alt_name = name.replace('.npz', '.svg')
                if (npz_path.parent / alt_name).exists():
                    svg_path = npz_path.parent / alt_name
                else:
                    # Try looking for any SVG with the same index
                    idx = extract_index(npz_path)
                    candidates = list(npz_path.parent.glob(f"boxes_{idx}*.svg"))
                    if candidates:
                        # Prioritize postprocess if available
                        pp_candidates = [c for c in candidates if 'postprocess' in c.name]
                        svg_path = pp_candidates[0] if pp_candidates else candidates[0]
        else:
            # Regular npz -> svg
            # handle boxes_N.npz -> boxes_N.svg
            svg_path = npz_path.with_suffix('.svg')
            if not svg_path.exists():
                name = npz_path.name
                svg_path = npz_path.parent / name.replace('.npz', '.svg')
        
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
) -> Tuple[Dict, Dict]:
    """
    Extract valid (non-empty) architecture and fixture items from sample.
    
    Note: Generated architecture and empty boxes are already removed by filter_generated_architecture.py,
    so we classify items directly by their class labels, not by position.
    
    Returns:
        fixtures: Dict mapping fixture type to list of item dicts
        arch: Dict with 'walls', 'doors', 'floors', 'windows' lists
    """
    # Get class indices
    if class_labels.ndim == 2:
        # Check if it's diffusion format (contains -1 values)
        is_diffusion_format = np.any(class_labels < 0)
        if is_diffusion_format:
            # Diffusion format: empty boxes are all -1, non-empty have one 1 and rest -1
            max_values = np.max(class_labels, axis=-1)
            is_empty = max_values < 0  # All values are -1, so max is -1
            cls_indices = np.argmax(class_labels, axis=-1)
            # Mark empty boxes with a high index (will be filtered by size check)
            cls_indices[is_empty] = 999  # Invalid index, will be filtered
        else:
            # Standard one-hot format (0/1)
            cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels
    
    # Process angles
    angles_flat = angles
    if angles.ndim == 2 and angles.shape[1] == 1:
        angles_flat = angles.flatten()
    
    # Size threshold for valid items (filter out zero-size padding as safety net)
    # Note: Empty boxes should already be removed by filter_generated_architecture.py
    size_norms = np.linalg.norm(sizes, axis=-1)
    valid_mask = size_norms > 0.01
    
    fixtures = {name: [] for name in FIXTURE_TYPES}
    arch = {'walls': [], 'doors': [], 'floors': [], 'windows': []}
    
    # Iterate through all boxes and classify by class label, not position
    for i in range(len(cls_indices)):
        if not valid_mask[i]:
            continue
            
        cls_idx = cls_indices[i]
        if cls_idx >= len(CLASS_LABELS_8):
            continue
            
        label = CLASS_LABELS_8[cls_idx]
        
        item = {
            'center': translations[i],
            'size': sizes[i],
            'angle': angles_flat[i] if i < len(angles_flat) else 0,
            'bbox': _get_aabb(translations[i], sizes[i] / 2),  # sizes are full, convert to half
        }
        
        # Classify by class label
        if label in ARCH_TYPES:
            if label == 'wall':
                arch['walls'].append(item)
            elif label == 'door':
                arch['doors'].append(item)
            elif label == 'floor':
                arch['floors'].append(item)
            elif label == 'window':
                arch['windows'].append(item)
        elif label in FIXTURE_TYPES:
            fixtures[label].append(item)
    
    return fixtures, arch


def _get_aabb(center: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    """Get axis-aligned bounding box as [x1, y1, z1, x2, y2, z2]."""
    return np.concatenate([center - half_size, center + half_size])


def _get_obb_corners_2d(center: np.ndarray, half_size: np.ndarray, angle: float) -> np.ndarray:
    """
    Get oriented bounding box corners in 2D (X-Z plane).
    
    Args:
        center: [x, y, z] center point
        half_size: [hx, hy, hz] half sizes
        angle: Rotation angle in radians (around Y axis, counterclockwise)
    
    Returns:
        corners: 4x2 array of corner coordinates [[x1, z1], [x2, z2], [x3, z3], [x4, z4]]
    """
    # Extract 2D center and half sizes (X and Z axes)
    cx, cz = center[0], center[2]
    hx, hz = half_size[0], half_size[2]
    
    # Local corners before rotation (relative to center)
    local_corners = np.array([
        [-hx, -hz],
        [ hx, -hz],
        [ hx,  hz],
        [-hx,  hz]
    ])
    
    # Rotation matrix for 2D (around Y axis = rotation in X-Z plane)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    # Rotate corners
    rotated_corners = local_corners @ rot_matrix.T
    
    # Translate to world position
    corners = rotated_corners + np.array([cx, cz])
    
    return corners


def _oriented_boxes_overlap_2d(
    center1: np.ndarray, half_size1: np.ndarray, angle1: float,
    center2: np.ndarray, half_size2: np.ndarray, angle2: float
) -> float:
    """
    Compute 2D overlap (floor plan) between two oriented bounding boxes using Separating Axis Theorem.
    Uses X and Z axes (Y is vertical).
    
    Args:
        center1: [x, y, z] center of first box
        half_size1: [hx, hy, hz] half sizes of first box
        angle1: Rotation angle of first box in radians
        center2: [x, y, z] center of second box
        half_size2: [hx, hy, hz] half sizes of second box
        angle2: Rotation angle of second box in radians
    
    Returns:
        overlap_area: Overlapping area in 2D (X-Z plane)
    """
    # Get corners for both boxes
    corners1 = _get_obb_corners_2d(center1, half_size1, angle1)
    corners2 = _get_obb_corners_2d(center2, half_size2, angle2)
    
    # Get edge directions (normalized) for both boxes
    edges1 = [
        corners1[1] - corners1[0],  # Edge 0->1
        corners1[2] - corners1[1],  # Edge 1->2
    ]
    edges2 = [
        corners2[1] - corners2[0],  # Edge 0->1
        corners2[2] - corners2[1],  # Edge 1->2
    ]
    
    # Normalize edges to get axes
    axes = []
    for edge in edges1:
        norm = np.linalg.norm(edge)
        if norm > 1e-6:
            axes.append(edge / norm)
    for edge in edges2:
        norm = np.linalg.norm(edge)
        if norm > 1e-6:
            axes.append(edge / norm)
    
    # Check separation on each axis
    min_overlap = np.inf
    
    for axis in axes:
        # Project corners of box1 onto axis
        proj1 = corners1 @ axis
        min1, max1 = proj1.min(), proj1.max()
        
        # Project corners of box2 onto axis
        proj2 = corners2 @ axis
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            # Separated on this axis, no overlap
            return 0.0
        
        # Calculate overlap on this axis
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    # If we get here, boxes overlap. Use polygon intersection for accurate area.
    if min_overlap < 1e-6:
        return 0.0
    
    # Compute actual polygon intersection area using shapely
    try:
        from shapely.geometry import Polygon
        
        poly1 = Polygon(corners1)
        poly2 = Polygon(corners2)
        
        if not poly1.is_valid or not poly2.is_valid:
            # Fallback to AABB if polygons are invalid
            aabb1 = _get_aabb(center1, half_size1)
            aabb2 = _get_aabb(center2, half_size2)
            return _boxes_overlap_2d_aabb(aabb1, aabb2)
        
        intersection = poly1.intersection(poly2)
        if intersection.is_empty:
            return 0.0
        
        return intersection.area
    except ImportError:
        # Fallback to approximation if shapely is not available
        # Use minimum overlap distance and average box dimensions as approximation
        avg_width = (half_size1[0] + half_size1[2] + half_size2[0] + half_size2[2]) / 2
        return min_overlap * avg_width


def _boxes_overlap_2d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute 2D overlap (floor plan) between two AABBs. Uses X and Z axes.
    DEPRECATED: Use _oriented_boxes_overlap_2d for oriented boxes.
    Kept for backward compatibility with swing_bbox calculations.
    """
    lt = np.maximum(bbox1[[0, 2]], bbox2[[0, 2]])
    rb = np.minimum(bbox1[[3, 5]], bbox2[[3, 5]])
    wh = np.maximum(rb - lt, 0)
    return wh[0] * wh[1]


def _boxes_overlap_2d_aabb(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Helper function for AABB overlap (used as fallback)."""
    lt = np.maximum(bbox1[[0, 2]], bbox2[[0, 2]])
    rb = np.minimum(bbox1[[3, 5]], bbox2[[3, 5]])
    wh = np.maximum(rb - lt, 0)
    return wh[0] * wh[1]


def _oriented_box_aabb_overlap_2d(
    center: np.ndarray, half_size: np.ndarray, angle: float,
    aabb: np.ndarray
) -> float:
    """
    Compute 2D overlap between an oriented box and an AABB.
    
    Args:
        center: [x, y, z] center of oriented box
        half_size: [hx, hy, hz] half sizes of oriented box
        angle: Rotation angle in radians
        aabb: [x1, y1, z1, x2, y2, z2] axis-aligned bounding box
    
    Returns:
        overlap_area: Overlapping area
    """
    # Convert AABB to center + half_size format (axis-aligned, angle=0)
    aabb_min = aabb[[0, 1, 2]]
    aabb_max = aabb[[3, 4, 5]]
    aabb_center = (aabb_min + aabb_max) / 2
    aabb_half_size = (aabb_max - aabb_min) / 2
    
    # Use oriented box overlap with angle=0 for AABB
    return _oriented_boxes_overlap_2d(
        center, half_size, angle,
        aabb_center, aabb_half_size, 0.0
    )


def _distance_2d(center1: np.ndarray, center2: np.ndarray) -> float:
    """2D distance on floor plan (X, Z)."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[2] - center2[2])**2)


def score_single_sample(
    translations: np.ndarray,
    sizes: np.ndarray,
    angles: np.ndarray,
    class_labels: np.ndarray,
) -> Tuple[float, Dict]:
    """
    Score a single sample using bathroom layout quality metrics.
    
    Note: Generated architecture and empty boxes are already removed by filter_generated_architecture.py.
    Items are classified by class labels, not by position.
    
    Returns:
        score: Quality score (higher is better)
        breakdown: Dict with score components
    """
    fixtures, arch = extract_valid_items(
        translations, sizes, angles, class_labels
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
                fix_i = all_fixtures[i]
                fix_j = all_fixtures[j]
                overlap = _oriented_boxes_overlap_2d(
                    fix_i['center'], fix_i['size'] / 2, fix_i['angle'],
                    fix_j['center'], fix_j['size'] / 2, fix_j['angle']
                )
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
            overlap = _oriented_boxes_overlap_2d(
                fix['center'], fix['size'] / 2, fix['angle'],
                wall['center'], wall['size'] / 2, wall['angle']
            )
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
            overlap = _oriented_box_aabb_overlap_2d(
                fix['center'], fix['size'] / 2, fix['angle'],
                swing_bbox
            )
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
    
    # === 8. CIRCULATION (bathroom completeness) ===
    # Required: at least 1 toilet + 1 sink/vanity + 1 shower/tub
    circulation_bonus = 0.0
    has_toilet = len(fixtures['toilet']) >= 1
    has_sink_vanity = len(fixtures['vanity']) >= 1
    has_shower_tub = len(fixtures['shower']) >= 1 or len(fixtures['tub']) >= 1
    
    # Bonus for having required fixtures
    if has_toilet:
        circulation_bonus += 2.0
    if has_sink_vanity:
        circulation_bonus += 2.0
    if has_shower_tub:
        circulation_bonus += 2.0
    
    # Additional bonus for having all required (completeness)
    if has_toilet and has_sink_vanity and has_shower_tub:
        circulation_bonus += 2.0
    
    # Small bonus for having multiple shower/tub types
    if len(fixtures['shower']) >= 1 and len(fixtures['tub']) >= 1:
        completeness_bonus += 1.0
    
    score += completeness_bonus
    breakdown['completeness'] = completeness_bonus
    breakdown['has_required_fixtures'] = has_toilet and has_sink_vanity and has_shower_tub
    
    return max(0.0, score), breakdown


def detect_violations(
    translations: np.ndarray,
    sizes: np.ndarray,
    angles: np.ndarray,
    class_labels: np.ndarray,
) -> Dict[str, bool]:
    """
    Detect violations for all constraint categories.
    
    Returns:
        violations: Dict with keys:
            - 'collision': Fixtures overlapping each other
            - 'fixture_wall_collision': Fixtures penetrating walls
            - 'boundary_violation': Fixtures outside floor boundary
            - 'door_blocking': Fixtures blocking door swing
            - 'completeness_violation': Missing required fixtures (1 toilet + 1 sink/vanity + 1 shower/tub)
            - 'num_violations': Total number of violations
            - 'has_any_violation': True if any violation exists
    """
    fixtures, arch = extract_valid_items(
        translations, sizes, angles, class_labels
    )
    
    walls = arch['walls']
    doors = arch['doors']
    floors = arch['floors']
    
    all_fixtures = []
    for fixture_list in fixtures.values():
        all_fixtures.extend(fixture_list)
    
    violations = {
        'collision': False,
        'fixture_wall_collision': False,
        'boundary_violation': False,
        'door_blocking': False,
        'completeness_violation': False,
    }
    
    # === 1. COLLISION DETECTION ===
    if len(all_fixtures) >= 2:
        for i in range(len(all_fixtures)):
            for j in range(i + 1, len(all_fixtures)):
                fix_i = all_fixtures[i]
                fix_j = all_fixtures[j]
                overlap = _oriented_boxes_overlap_2d(
                    fix_i['center'], fix_i['size'] / 2, fix_i['angle'],
                    fix_j['center'], fix_j['size'] / 2, fix_j['angle']
                )
                if overlap > 0.01:
                    violations['collision'] = True
                    break
            if violations['collision']:
                break
    
    # === 2. FIXTURE-WALL COLLISION ===
    penetration_threshold = 0.15  # Allow 15cm for attachment
    for fix in all_fixtures:
        for wall in walls:
            overlap = _oriented_boxes_overlap_2d(
                fix['center'], fix['size'] / 2, fix['angle'],
                wall['center'], wall['size'] / 2, wall['angle']
            )
            if overlap > 0:
                # Check penetration depth
                wall_sizes_2d = wall['size'][[0, 2]]
                thin_axis = np.argmin(wall_sizes_2d)
                
                fix_center = fix['center'][[0, 2]][thin_axis]
                wall_center = wall['center'][[0, 2]][thin_axis]
                
                penetration = abs(fix_center - wall_center) - wall_sizes_2d[thin_axis]
                if penetration > penetration_threshold:
                    violations['fixture_wall_collision'] = True
                    break
        if violations['fixture_wall_collision']:
            break
    
    # === 3. BOUNDARY VIOLATION ===
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
            total_outside = np.sum(outside_min) + np.sum(outside_max)
            if total_outside > 0.01:  # Small tolerance
                violations['boundary_violation'] = True
                break
    
    # === 4. DOOR BLOCKING ===
    for door in doors:
        door_width = max(door['size'][0], door['size'][2])
        swing_bbox = door['bbox'].copy()
        swing_extension = door_width * 0.8
        
        swing_bbox[0] -= swing_extension * 0.5
        swing_bbox[2] -= swing_extension * 0.5
        swing_bbox[3] += swing_extension * 0.5
        swing_bbox[5] += swing_extension * 0.5
        
        for fix in all_fixtures:
            overlap = _oriented_box_aabb_overlap_2d(
                fix['center'], fix['size'] / 2, fix['angle'],
                swing_bbox
            )
            if overlap > 0.01:
                violations['door_blocking'] = True
                break
        if violations['door_blocking']:
            break
    
    # === 5. COMPLETENESS (bathroom completeness) ===
    # Required: at least 1 toilet + 1 sink/vanity + 1 shower/tub
    has_toilet = len(fixtures['toilet']) >= 1
    has_sink_vanity = len(fixtures['vanity']) >= 1
    has_shower_tub = len(fixtures['shower']) >= 1 or len(fixtures['tub']) >= 1
    
    if not (has_toilet and has_sink_vanity and has_shower_tub):
        violations['completeness_violation'] = True
    
    # Count total violations
    num_violations = sum(1 for v in violations.values() if v)
    violations['num_violations'] = num_violations
    violations['has_any_violation'] = num_violations > 0
    
    return violations


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


def calculate_violation_statistics(samples: List[Dict], top_k: int = 5) -> Dict:
    """
    Calculate violation statistics across all samples and for top-K selection.
    
    Returns:
        stats: Dict with violation percentages and counts
    """
    num_samples = len(samples)
    if num_samples == 0:
        return {
            'num_samples': 0,
            'collision_pass_pct': 0.0,
            'fixture_wall_pass_pct': 0.0,
            'boundary_pass_pct': 0.0,
            'door_blocking_pass_pct': 0.0,
            'completeness_pass_pct': 0.0,
            'all_pass_pct': 0.0,
            'top_k_zero_viol_pct': 0.0,
            'top_k_one_viol_pct': 0.0,
        }
    
    violation_counts = {
        'collision': 0,
        'fixture_wall_collision': 0,
        'boundary_violation': 0,
        'door_blocking': 0,
        'completeness_violation': 0,
        'all_violations': 0,  # Count of samples with at least one violation
    }
    
    sample_metrics = []
    
    for sample in samples:
        # Get violations
        violations = detect_violations(
            sample['translations'],
            sample['sizes'],
            sample['angles'],
            sample['class_labels']
        )
        
        # Get score for top-K calculation
        score, _ = score_single_sample(
            sample['translations'],
            sample['sizes'],
            sample['angles'],
            sample['class_labels']
        )
        
        sample_metrics.append({
            'num_violations': violations['num_violations'],
            'score': score
        })
        
        for key in ['collision', 'fixture_wall_collision', 'boundary_violation', 
                    'door_blocking', 'completeness_violation']:
            if violations[key]:
                violation_counts[key] += 1
        
        if violations['has_any_violation']:
            violation_counts['all_violations'] += 1
    
    # Calculate top-K zero violation percentage
    sample_metrics_sorted = sorted(
        sample_metrics,
        key=lambda x: (x['num_violations'], -x['score'])
    )
    
    actual_top_k = min(top_k, len(sample_metrics_sorted))
    top_k_subset = sample_metrics_sorted[:actual_top_k]
    
    top_k_zero_viol_count = sum(1 for d in top_k_subset if d['num_violations'] == 0)
    top_k_zero_viol_pct = (top_k_zero_viol_count / actual_top_k * 100) if actual_top_k > 0 else 0.0
    
    top_k_one_viol_count = sum(1 for d in top_k_subset if d['num_violations'] <= 1)
    top_k_one_viol_pct = (top_k_one_viol_count / actual_top_k * 100) if actual_top_k > 0 else 0.0
    
    # Calculate pass percentages (non-violation percentages)
    stats = {
        'num_samples': num_samples,
        'collision_pass_pct': (1 - violation_counts['collision'] / num_samples) * 100,
        'fixture_wall_pass_pct': (1 - violation_counts['fixture_wall_collision'] / num_samples) * 100,
        'boundary_pass_pct': (1 - violation_counts['boundary_violation'] / num_samples) * 100,
        'door_blocking_pass_pct': (1 - violation_counts['door_blocking'] / num_samples) * 100,
        'completeness_pass_pct': (1 - violation_counts['completeness_violation'] / num_samples) * 100,
        'all_pass_pct': (1 - violation_counts['all_violations'] / num_samples) * 100,
        'top_k_zero_viol_pct': top_k_zero_viol_pct,
        'top_k_one_viol_pct': top_k_one_viol_pct,
    }
    
    return stats


def score_samples(
    samples: List[Dict],
    weights: Optional[ScoringWeights] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Score all samples.
    
    Note: Generated architecture and empty boxes are already removed by filter_generated_architecture.py.
    Items are classified by class labels, not by position.
    
    Args:
        samples: List of sample dicts
        weights: Not used (kept for API compatibility)
        show_progress: Whether to show progress bar for scoring
    
    Returns:
        scores: Array of scores for all samples
    """
    all_scores = np.zeros(len(samples), dtype=np.float32)
    
    iterator = tqdm(samples, desc="Scoring samples", unit="sample", disable=not show_progress)
    for i, sample in enumerate(iterator):
        trans = sample['translations']
        sizes = sample['sizes']
        angles = sample['angles']
        class_labels = sample['class_labels']
        
        # Score the sample (items classified by class labels)
        score, _ = score_single_sample(
            trans, sizes, angles, class_labels
        )
        all_scores[i] = score
    
    return all_scores


def select_and_copy_top_k(
    folder_path: str,
    output_dir: Optional[str] = None,
    top_k: int = 5,
    create_grid: bool = True,
    verbose: bool = True,
    generate_statistics: bool = True,
) -> Tuple[List[int], np.ndarray, Optional[Dict]]:
    """
    Score samples, select top-K based on violation count + score, and copy files to output directory.
    
    Selection Strategy:
        1. Prioritize samples with fewer violations
        2. Within same violation count, prioritize higher scores
        3. Select from postprocess.npz files (if available, otherwise use regular .npz)
    
    Statistics Generation:
        - Profiles both .npz and .postprocess.npz files
        - Calculates non-violation percentages for each constraint
        - Outputs per-scene statistics
    
    Args:
        folder_path: Path to folder containing boxes_*.npz files
        output_dir: Output directory for top-K SVGs (default: folder_path/top_k)
        top_k: Number of top samples to select
        create_grid: Whether to create a comparison grid SVG
        verbose: Whether to print progress
        generate_statistics: Whether to generate violation statistics
    
    Returns:
        top_indices: Indices of top-K samples
        all_scores: Scores for all samples
        statistics: Dict with violation statistics (if generate_statistics=True)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    scene_id = folder.name
    
    # Set output directory
    if output_dir is None:
        output_dir = folder / "top_k"
    else:
        output_dir = Path(output_dir)
    
    statistics = None
    
    # === STATISTICS GENERATION ===
    if generate_statistics:
        if verbose:
            print(f"\n[{scene_id}] Generating violation statistics...")
        
        stats_results = {}
        
        # Load and profile regular .npz files
        try:
            samples_npz, _, _ = load_samples_from_folder(folder_path, use_postprocess=False)
            if verbose:
                print(f"  Found {len(samples_npz)} regular .npz files")
            stats_npz = calculate_violation_statistics(samples_npz, top_k=top_k)
            stats_results['npz'] = stats_npz
        except ValueError as e:
            if verbose:
                print(f"  Warning: No regular .npz files found: {e}")
            stats_results['npz'] = None
        
        # Load and profile postprocess.npz files
        try:
            samples_postprocess, _, _ = load_samples_from_folder(folder_path, use_postprocess=True)
            if verbose:
                print(f"  Found {len(samples_postprocess)} postprocess.npz files")
            stats_postprocess = calculate_violation_statistics(samples_postprocess, top_k=top_k)
            stats_results['postprocess'] = stats_postprocess
        except ValueError as e:
            if verbose:
                print(f"  Warning: No postprocess.npz files found: {e}")
            stats_results['postprocess'] = None
        
        statistics = {
            'scene_id': scene_id,
            'folder': str(folder_path),
            'npz': stats_results['npz'],
            'postprocess': stats_results['postprocess'],
        }
    
    # === SAMPLE SELECTION ===
    # Try to load postprocess files first, fall back to regular if not available
    try:
        samples, bounds, svg_paths = load_samples_from_folder(folder_path, use_postprocess=True)
        using_postprocess = True
        if verbose:
            print(f"\n[{scene_id}] Using postprocess.npz files for selection ({len(samples)} samples)")
    except ValueError:
        if verbose:
            print(f"\n[{scene_id}] Postprocess files not found, using regular .npz files")
        samples, bounds, svg_paths = load_samples_from_folder(folder_path, use_postprocess=False)
        using_postprocess = False
    
    # Limit top_k to available samples
    top_k = min(top_k, len(samples))
    
    # Score and detect violations for all samples
    if verbose:
        print(f"[{scene_id}] Scoring and analyzing {len(samples)} samples...")
    
    all_scores = np.zeros(len(samples), dtype=np.float32)
    all_violations = []
    
    show_progress = verbose and len(samples) > 10
    iterator = tqdm(samples, desc="Processing", disable=not show_progress)
    
    for i, sample in enumerate(iterator):
        # Score the sample
        score, _ = score_single_sample(
            sample['translations'],
            sample['sizes'],
            sample['angles'],
            sample['class_labels']
        )
        all_scores[i] = score
        
        # Detect violations
        violations = detect_violations(
            sample['translations'],
            sample['sizes'],
            sample['angles'],
            sample['class_labels']
        )
        all_violations.append(violations)
    
    # === TOP-K SELECTION BASED ON VIOLATIONS + SCORE ===
    # Create sorting key: (num_violations, -score)
    # This prioritizes fewer violations first, then higher scores
    sample_data = []
    for i in range(len(samples)):
        sample_data.append({
            'index': i,
            'score': all_scores[i],
            'num_violations': all_violations[i]['num_violations'],
        })
    
    # Sort by: 1) num_violations (ascending), 2) score (descending)
    sample_data_sorted = sorted(
        sample_data,
        key=lambda x: (x['num_violations'], -x['score'])
    )
    
    # Select top-K
    top_k_data = sample_data_sorted[:top_k]
    top_indices = [x['index'] for x in top_k_data]
    
    if verbose:
        print(f"\n[{scene_id}] Top-{top_k} selection:")
        for rank, data in enumerate(top_k_data):
            idx = data['index']
            print(f"  Rank {rank+1}: Sample {idx}, "
                  f"Violations={data['num_violations']}, Score={data['score']:.1f}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy top-K files
    copied_files = []
    for rank, idx in enumerate(top_indices):
        # Copy SVG
        src_svg = Path(svg_paths[idx])
        if src_svg.exists():
            dst_name = f"top_{rank+1}_boxes_{idx}.svg"
            dst_svg = output_dir / dst_name
            shutil.copy2(src_svg, dst_svg)
            copied_files.append(dst_svg)
        else:
            if verbose:
                print(f"  Warning: SVG not found: {src_svg}", file=sys.stderr)
        
        # Copy NPZ
        src_npz = Path(samples[idx]['npz_path'])
        if src_npz.exists():
            # Preserve the original filename pattern (postprocess or not)
            if using_postprocess:
                dst_name = f"top_{rank+1}_boxes_{idx}.postprocess.npz"
            else:
                dst_name = f"top_{rank+1}_boxes_{idx}.npz"
            dst_npz = output_dir / dst_name
            shutil.copy2(src_npz, dst_npz)
    
    # Create comparison grid SVG
    if create_grid:
        json_path = folder / "simple_design_filtered.json"
        grid_bounds = get_bounds_from_json(str(json_path))
        if grid_bounds is None:
            grid_bounds = bounds
        
        grid_items = []
        
        # Add ground truth if available
        label_svg_path = folder / "simple_design_label.svg"
        if label_svg_path.exists():
            with open(label_svg_path, 'r') as f:
                label_svg = f.read()
            grid_items.append(("GT Label", extract_svg_inner(label_svg)))
        
        # Add top-K samples that were successfully copied
        for rank, idx in enumerate(top_indices):
            # Check if we copied this SVG
            dst_name = f"top_{rank+1}_boxes_{idx}.svg"
            dst_svg = output_dir / dst_name
            
            if dst_svg.exists():
                with open(dst_svg, 'r') as f:
                    svg_content = f.read()
                score = all_scores[idx]
                num_viol = all_violations[idx]['num_violations']
                
                # Format violation tags
                violation_tags = []
                v = all_violations[idx]
                if v['collision']: violation_tags.append("Coll")
                if v['fixture_wall_collision']: violation_tags.append("Wall")
                if v['boundary_violation']: violation_tags.append("Bound")
                if v['door_blocking']: violation_tags.append("Door")
                if v['completeness_violation']: violation_tags.append("Compl")
                
                v_str = f"({', '.join(violation_tags)})" if violation_tags else ""
                
                grid_items.append((f"#{rank+1} (S{idx}, V:{num_viol}, S:{score:.1f})\n{v_str}", 
                                 extract_svg_inner(svg_content)))
        
        if len(grid_items) > 0:
            grid_output = output_dir / "comparison_grid.svg"
            create_comparison_grid_svg(grid_items, str(grid_output))
    
    # Save detailed results
    results_data = {
        'scene_id': scene_id,
        'folder': str(folder_path),
        'num_samples': len(samples),
        'top_k': top_k,
        'using_postprocess': using_postprocess,
        'top_indices': top_indices,
        'top_scores': [float(all_scores[i]) for i in top_indices],  # Convert numpy float32 to Python float
        'top_violations': [all_violations[i]['num_violations'] for i in top_indices],
        'statistics': statistics,
    }
    
    results_json = output_dir / "selection_results.json"
    with open(results_json, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return top_indices, all_scores, statistics



def process_folder_recursive(
    base_folder: str,
    output_base: Optional[str] = None,
    top_k: int = 5,
    verbose: bool = True,
    generate_csv: bool = True,
) -> Dict[str, Tuple[List[int], np.ndarray, Optional[Dict]]]:
    """
    Process all scene folders recursively under base_folder and generate CSV statistics.
    
    Looks for folders containing boxes_*.npz or boxes_*.postprocess.npz files.
    
    Args:
        base_folder: Base folder to search
        output_base: Base output folder (default: each scene's own top_k subfolder)
        top_k: Number of top samples per scene
        verbose: Whether to print progress
        generate_csv: Whether to generate CSV statistics file
    
    Returns:
        results: Dict mapping folder path to (top_indices, all_scores, statistics)
    """
    base = Path(base_folder)
    results = {}
    all_statistics = []
    
    # Find all folders with boxes_*.npz or boxes_*.postprocess.npz files
    scene_folders = set()
    for npz_file in base.rglob("boxes_*.npz"):
        scene_folders.add(npz_file.parent)
    for npz_file in base.rglob("boxes_*.postprocess.npz"):
        scene_folders.add(npz_file.parent)
    
    scene_folders = sorted(scene_folders)
    
    if not scene_folders:
        print(f"No scene folders found under {base_folder}", file=sys.stderr)
        return results
    
    # Process folders with tqdm progress bar
    for folder in tqdm(scene_folders, desc="Processing scenes", unit="scene"):
        try:
            if output_base:
                output_dir = Path(output_base) / folder.name / "top_k"
            else:
                output_dir = None  # Use default (folder/top_k)
            
            top_indices, all_scores, statistics = select_and_copy_top_k(
                str(folder),
                output_dir=str(output_dir) if output_dir else None,
                top_k=top_k,
                create_grid=True,
                verbose=False,  # Disable verbose output in recursive mode
                generate_statistics=True,
            )
            results[str(folder)] = (top_indices, all_scores, statistics)
            
            if statistics:
                all_statistics.append(statistics)
            
        except Exception as e:
            tqdm.write(f"Error processing {folder}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    
    if verbose:
        print(f"\nProcessed {len(results)} / {len(scene_folders)} scene folder(s)")
    
    # Generate CSV statistics
    if generate_csv and all_statistics:
        csv_path = base / "violation_statistics.csv"
        write_statistics_csv(all_statistics, str(csv_path))
        if verbose:
            print(f"\nStatistics CSV saved to: {csv_path}")
    
    return results


def write_statistics_csv(all_statistics: List[Dict], output_path: str):
    """
    Write violation statistics to a CSV file.
    
    The CSV will have per-scene rows for both .npz and .postprocess.npz files,
    plus an aggregated summary row for each file type.
    
    Args:
        all_statistics: List of statistics dicts from select_and_copy_top_k
        output_path: Path to output CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Scene ID',
            'Type',
            'Num Samples',
            'Collision Pass %',
            'Fixture-Wall Pass %',
            'Boundary Pass %',
            'Door Blocking Pass %',
            'Completeness Pass %',
            'All Pass %',
            'Top-K 0-Viol %',
            'Top-K 1-Viol %',
        ])
        
        # Accumulators for aggregated statistics
        agg_npz = {
            'num_samples': 0,
            'collision': 0,
            'fixture_wall': 0,
            'boundary': 0,
            'door_blocking': 0,
            'completeness': 0,
            'all_pass': 0,
            'top_k_zero_sum_pct': 0.0,
            'top_k_one_sum_pct': 0.0,
            'top_k_count': 0,
        }
        agg_postprocess = {
            'num_samples': 0,
            'collision': 0,
            'fixture_wall': 0,
            'boundary': 0,
            'door_blocking': 0,
            'completeness': 0,
            'all_pass': 0,
            'top_k_zero_sum_pct': 0.0,
            'top_k_one_sum_pct': 0.0,
            'top_k_count': 0,
        }
        
        # Per-scene rows
        for stats in all_statistics:
            scene_id = stats['scene_id']
            
            # NPZ row
            if stats['npz'] is not None:
                npz_stats = stats['npz']
                writer.writerow([
                    scene_id,
                    'npz',
                    npz_stats['num_samples'],
                    f"{npz_stats['collision_pass_pct']:.2f}",
                    f"{npz_stats['fixture_wall_pass_pct']:.2f}",
                    f"{npz_stats['boundary_pass_pct']:.2f}",
                    f"{npz_stats['door_blocking_pass_pct']:.2f}",
                    f"{npz_stats['completeness_pass_pct']:.2f}",
                    f"{npz_stats['all_pass_pct']:.2f}",
                    f"{npz_stats.get('top_k_zero_viol_pct', 0.0):.2f}",
                    f"{npz_stats.get('top_k_one_viol_pct', 0.0):.2f}",
                ])
                
                # Accumulate for aggregated stats
                n = npz_stats['num_samples']
                agg_npz['num_samples'] += n
                agg_npz['collision'] += n * npz_stats['collision_pass_pct'] / 100
                agg_npz['fixture_wall'] += n * npz_stats['fixture_wall_pass_pct'] / 100
                agg_npz['boundary'] += n * npz_stats['boundary_pass_pct'] / 100
                agg_npz['door_blocking'] += n * npz_stats['door_blocking_pass_pct'] / 100
                agg_npz['completeness'] += n * npz_stats['completeness_pass_pct'] / 100
                agg_npz['all_pass'] += n * npz_stats['all_pass_pct'] / 100
                agg_npz['top_k_zero_sum_pct'] += npz_stats.get('top_k_zero_viol_pct', 0.0)
                agg_npz['top_k_one_sum_pct'] += npz_stats.get('top_k_one_viol_pct', 0.0)
                agg_npz['top_k_count'] += 1
            
            # Postprocess row
            if stats['postprocess'] is not None:
                post_stats = stats['postprocess']
                writer.writerow([
                    scene_id,
                    'postprocess',
                    post_stats['num_samples'],
                    f"{post_stats['collision_pass_pct']:.2f}",
                    f"{post_stats['fixture_wall_pass_pct']:.2f}",
                    f"{post_stats['boundary_pass_pct']:.2f}",
                    f"{post_stats['door_blocking_pass_pct']:.2f}",
                    f"{post_stats['completeness_pass_pct']:.2f}",
                    f"{post_stats['all_pass_pct']:.2f}",
                    f"{post_stats.get('top_k_zero_viol_pct', 0.0):.2f}",
                    f"{post_stats.get('top_k_one_viol_pct', 0.0):.2f}",
                ])
                
                # Accumulate for aggregated stats
                n = post_stats['num_samples']
                agg_postprocess['num_samples'] += n
                agg_postprocess['collision'] += n * post_stats['collision_pass_pct'] / 100
                agg_postprocess['fixture_wall'] += n * post_stats['fixture_wall_pass_pct'] / 100
                agg_postprocess['boundary'] += n * post_stats['boundary_pass_pct'] / 100
                agg_postprocess['door_blocking'] += n * post_stats['door_blocking_pass_pct'] / 100
                agg_postprocess['completeness'] += n * post_stats['completeness_pass_pct'] / 100
                agg_postprocess['all_pass'] += n * post_stats['all_pass_pct'] / 100
                agg_postprocess['top_k_zero_sum_pct'] += post_stats.get('top_k_zero_viol_pct', 0.0)
                agg_postprocess['top_k_one_sum_pct'] += post_stats.get('top_k_one_viol_pct', 0.0)
                agg_postprocess['top_k_count'] += 1
        
        # Aggregated rows
        writer.writerow([])  # Empty row separator
        
        if agg_npz['num_samples'] > 0:
            n = agg_npz['num_samples']
            scenes_n = agg_npz['top_k_count']
            writer.writerow([
                'AGGREGATE',
                'npz',
                n,
                f"{agg_npz['collision'] / n * 100:.2f}",
                f"{agg_npz['fixture_wall'] / n * 100:.2f}",
                f"{agg_npz['boundary'] / n * 100:.2f}",
                f"{agg_npz['door_blocking'] / n * 100:.2f}",
                f"{agg_npz['completeness'] / n * 100:.2f}",
                f"{agg_npz['all_pass'] / n * 100:.2f}",
                f"{agg_npz['top_k_zero_sum_pct'] / scenes_n if scenes_n > 0 else 0.0:.2f}",
                f"{agg_npz['top_k_one_sum_pct'] / scenes_n if scenes_n > 0 else 0.0:.2f}",
            ])
        
        if agg_postprocess['num_samples'] > 0:
            n = agg_postprocess['num_samples']
            scenes_n = agg_postprocess['top_k_count']
            writer.writerow([
                'AGGREGATE',
                'postprocess',
                n,
                f"{agg_postprocess['collision'] / n * 100:.2f}",
                f"{agg_postprocess['fixture_wall'] / n * 100:.2f}",
                f"{agg_postprocess['boundary'] / n * 100:.2f}",
                f"{agg_postprocess['door_blocking'] / n * 100:.2f}",
                f"{agg_postprocess['completeness'] / n * 100:.2f}",
                f"{agg_postprocess['all_pass'] / n * 100:.2f}",
                f"{agg_postprocess['top_k_zero_sum_pct'] / scenes_n if scenes_n > 0 else 0.0:.2f}",
                f"{agg_postprocess['top_k_one_sum_pct'] / scenes_n if scenes_n > 0 else 0.0:.2f}",
            ])



def main():
    parser = argparse.ArgumentParser(
        description="Select top-K SVG samples from generated bathroom scenes based on quality scores and violations"
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
        "--no_grid",
        action="store_true",
        help="Skip creating comparison grid SVG"
    )
    
    parser.add_argument(
        "--no_statistics",
        action="store_true",
        help="Skip generating violation statistics"
    )
    
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Skip generating CSV statistics file (only for --recursive mode)"
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
    generate_statistics = not args.no_statistics
    
    if args.recursive:
        # Process multiple folders
        process_folder_recursive(
            str(folder),
            output_base=args.output_dir,
            top_k=args.top_k,
            verbose=verbose,
            generate_csv=not args.no_csv,
        )
    else:
        # Process single folder
        try:
            top_indices, all_scores, statistics = select_and_copy_top_k(
                str(folder),
                output_dir=args.output_dir,
                top_k=args.top_k,
                create_grid=not args.no_grid,
                verbose=verbose,
                generate_statistics=generate_statistics,
            )
            
            # For single folder, also write a CSV if statistics were generated
            if statistics and generate_statistics:
                csv_path = (Path(args.output_dir) if args.output_dir else (folder / "top_k")) / "violation_statistics.csv"
                write_statistics_csv([statistics], str(csv_path))
                if verbose:
                    print(f"\nStatistics CSV saved to: {csv_path}")
                    
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

