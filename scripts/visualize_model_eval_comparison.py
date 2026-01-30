#!/usr/bin/env python
"""
Create comparison SVG visualizations for multiple model settings in model_eval_2.

Generates a comparison grid with one row per model setting, showing:
- Row 1: v1.2_no_aug (Arch + samples)
- Row 2: v1.3_no_aug (samples)
- Row 3: v1.3_no_aug_fixed_iou (samples)
- Row 4: v1.2_no_aug_fixed_iou (Label + samples)

Usage:
    python scripts/visualize_model_eval_comparison.py --input_dir output/model_eval_2
"""

import argparse
import os
import numpy as np
import json
import math
import re
from pathlib import Path
from tqdm import tqdm

# Color palette
COLOR_PALETTE = {
    "void": [255, 255, 255],
    "floor": [211, 211, 211],
    "wall": [0, 0, 153],
    "door": [153, 0, 0],
    "window": [255, 153, 153],
    "toilet": [152, 223, 138],
    "vanity": [105, 183, 100],
    "tub": [44, 160, 44],
    "shower": [197, 176, 213],
    "clearance": [255, 240, 200],
    "empty": [255, 255, 255]
}

CLASS_LABELS = ["vanity", "toilet", "shower", "tub", "floor", "wall", "door", "window", "empty"]

# Model settings to compare (order matters for display)
# Default: compare only v1.2_no_aug and v1.2_no_aug_fixed_iou
MODEL_SETTINGS = [
    "v1.2_no_aug",
    "v1.3_no_aug",
    "v1.3_aug_fixed_iou"
]


def rotate_point(x, z, yaw_rad):
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    return x * cos_yaw - z * sin_yaw, x * sin_yaw + z * cos_yaw


def get_bounding_box_corners(center, size, angle):
    cx, _, cz = center
    sx, _, sz = size
    half_sx, half_sz = sx / 2, sz / 2
    
    local_corners = [
        (-half_sx, -half_sz),
        (half_sx, -half_sz),
        (half_sx, half_sz),
        (-half_sx, half_sz)
    ]
    
    corners = []
    for lx, lz in local_corners:
        rx, rz = rotate_point(lx, lz, -angle)
        corners.append((cx + rx, cz + rz))
    return corners


def world_to_pixel(world_coords, bounds, img_size, unit_length=10):
    min_x, max_x, min_z, max_z = bounds
    width, height = img_size
    
    world_width = max_x - min_x
    world_height = max_z - min_z
    
    scale = unit_length
    offset_x = (width - world_width * scale) / 2
    offset_y = (height - world_height * scale) / 2
    
    pixel_coords = []
    for x, z in world_coords:
        px = (x - min_x) * scale + offset_x
        py = (z - min_z) * scale + offset_y
        pixel_coords.append((px, py))
    
    return pixel_coords, scale


def compute_scene_bounds(boxes, valid_mask):
    all_points = []
    translations = boxes['translations']
    sizes = boxes['sizes']
    angles = boxes['angles']
    
    for i in range(len(translations)):
        if not valid_mask[i]: 
            continue
        t = translations[i]
        s = sizes[i]
        a = angles[i]
        if hasattr(a, "__len__") and len(a) > 0:
            angle_val = a[0]
        else:
            angle_val = a
        corners = get_bounding_box_corners(t, s, angle_val)
        all_points.extend(corners)
        
    if not all_points:
        return (-1, 1, -1, 1)
        
    xs = [p[0] for p in all_points]
    zs = [p[1] for p in all_points]
    return (min(xs), max(xs), min(zs), max(zs))


def visualize_scene_to_svg_content(translations, sizes, angles, class_labels, fixed_bounds=None, num_partial_boxes=None):
    """Create SVG inner content (without svg wrapper) from scene parameters."""
    # Handle class labels
    if len(class_labels.shape) > 1 and class_labels.shape[1] > 1:
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()
    
    # Handle angles
    if len(angles.shape) > 1:
        if angles.shape[1] == 1:
            angles = angles.flatten()
        elif angles.shape[1] == 2:
            angles = np.arctan2(angles[:, 1], angles[:, 0])
    
    n_objs = len(cls_indices)
    
    # Filter empty
    empty_idx = CLASS_LABELS.index('empty')
    valid_mask = cls_indices != empty_idx
    
    # Filter generated architecture
    if num_partial_boxes is not None:
        for i in range(n_objs):
            cls_idx = cls_indices[i]
            is_arch = cls_idx >= 4 and cls_idx <= 7
            is_generated = i >= num_partial_boxes
            if is_arch and is_generated:
                valid_mask[i] = False
    
    # Filter zero-size boxes
    size_magnitude = np.sqrt(np.sum(sizes ** 2, axis=1))
    has_size = size_magnitude > 0.01
    valid_mask = valid_mask & has_size
    
    if fixed_bounds is not None:
        bounds = fixed_bounds
    else:
        bounds = compute_scene_bounds({
            'translations': translations,
            'sizes': sizes,
            'angles': angles
        }, valid_mask)
    
    W, H = 120, 120
    svg_lines = [f'<rect width="{W}" height="{H}" fill="white"/>']
    
    indices = [i for i in range(n_objs) if valid_mask[i]]
    
    def get_priority(cls_idx):
        name = CLASS_LABELS[cls_idx]
        if name == 'floor': return 0
        if name in ['wall', 'window', 'door']: return 1
        return 2
    
    indices.sort(key=lambda i: get_priority(cls_indices[i]))
    
    for i in indices:
        cls_idx = cls_indices[i]
        label = CLASS_LABELS[cls_idx]
        color = COLOR_PALETTE.get(label, [0,0,0])
        rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"
        
        t = translations[i]
        s = sizes[i]
        a = angles[i]
        if isinstance(a, np.ndarray) and a.size > 0: 
            a = a.item()
            
        corners_world = get_bounding_box_corners(t, s, a)
        corners_px, scale = world_to_pixel(corners_world, bounds, (W, H), unit_length=10)
        pts_str = " ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in corners_px])
        
        svg_lines.append(f'<polygon points="{pts_str}" fill="{rgb_str}" stroke="{rgb_str}" data-type="{label}" />')
    
    return "\n".join(svg_lines)


def get_arch_svg_content(scene_dir, bounds=None):
    """Get architecture-only SVG content from a scene directory."""
    label_svg_path = scene_dir / "simple_design_label.svg"
    if label_svg_path.exists():
        with open(label_svg_path, 'r') as f:
            label_svg = f.read()
        match = re.search(r'<svg[^>]*>(.*)</svg>', label_svg, re.DOTALL)
        if match:
            arch_colors = [
                "rgb(211,211,211)",  # floor
                "rgb(0,0,153)",      # wall
                "rgb(153,0,0)",      # door
                "rgb(255,153,153)",  # window
            ]
            filtered_elements = []
            for element in re.findall(r'<(?:rect|polygon|path)[^>]*/?>', match.group(1)):
                for color in arch_colors:
                    if color in element:
                        filtered_elements.append(element)
                        break
            return '\n'.join(filtered_elements)
    return None


def get_label_svg_content(scene_dir):
    """Get full label SVG content from a scene directory."""
    label_svg_path = scene_dir / "simple_design_label.svg"
    if label_svg_path.exists():
        with open(label_svg_path, 'r') as f:
            label_svg = f.read()
        match = re.search(r'<svg[^>]*>(.*)</svg>', label_svg, re.DOTALL)
        if match:
            return match.group(1)
    return None


def get_scene_bounds_from_json(json_path):
    """Compute scene bounds from JSON file."""
    if not json_path.exists():
        return None, None
    
    try:
        with open(json_path, 'r') as f:
            scene_data = json.load(f)
        json_items = scene_data['items'] if isinstance(scene_data, dict) else scene_data
        all_points = []
        
        arch_types = {'floor', 'wall', 'door', 'window'}
        num_partial_boxes = sum(1 for item in json_items if item.get('diffusion_type', '') in arch_types)
        
        for item in json_items:
            cx, cz = item.get('center_x', 0), item.get('center_z', 0)
            sx, sz = item.get('size_x', 0), item.get('size_z', 0)
            y_rad = math.radians(item.get('yaw', 0))
            h_sx, h_sz = sx/2, sz/2
            for lx, lz in [(-h_sx,-h_sz),(h_sx,-h_sz),(h_sx,h_sz),(-h_sx,h_sz)]:
                rx = lx * math.cos(y_rad) - lz * math.sin(y_rad)
                rz = lx * math.sin(y_rad) + lz * math.cos(y_rad)
                all_points.append((cx + rx, cz + rz))
        
        if all_points:
            xs = [p[0] for p in all_points]
            zs = [p[1] for p in all_points]
            return (min(xs), max(xs), min(zs), max(zs)), num_partial_boxes
    except Exception:
        pass
    return None, None


def create_comparison_svg(scene_id, model_dirs, output_path, n_samples=6):
    """
    Create a comparison SVG for a single scene across all model settings.
    
    Args:
        scene_id: Scene identifier (e.g., "191106600_Bath_US_simple_design_filtered")
        model_dirs: Dict mapping model_name -> model_dir path
        output_path: Output path for the comparison SVG
        n_samples: Number of samples to show per model
    """
    cell_w, cell_h = 120, 120
    margin = 20
    label_w = 60  # Width for row labels on left
    
    # Find which models have this scene
    available_models = []
    for model_name in MODEL_SETTINGS:
        if model_name in model_dirs:
            scene_dir = model_dirs[model_name] / scene_id
            if scene_dir.exists():
                npz_files = list(scene_dir.glob('boxes_*.npz'))
                if npz_files:
                    available_models.append(model_name)
    
    if not available_models:
        return False
    
    # Get bounds from first available model's JSON
    bounds = None
    num_partial_boxes = None
    for model_name in available_models:
        scene_dir = model_dirs[model_name] / scene_id
        json_path = scene_dir / "simple_design_filtered.json"
        bounds, num_partial_boxes = get_scene_bounds_from_json(json_path)
        if bounds:
            break
    
    # Number of columns: Arch + n_samples (samples)
    n_cols = 1 + n_samples
    n_rows = len(available_models)
    
    total_w = label_w + n_cols * (cell_w + margin) + margin
    total_h = n_rows * (cell_h + margin) + margin + 40  # 40 for headers
    
    svg_lines = [
        f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{total_w}" height="{total_h}" fill="white"/>'
    ]
    
    for row_idx, model_name in enumerate(available_models):
        scene_dir = model_dirs[model_name] / scene_id
        
        y_offset = 40 + row_idx * (cell_h + margin)
        
        # Row label on left
        row_label_y = y_offset + cell_h / 2
        svg_lines.append(f'<text x="{label_w/2}" y="{row_label_y}" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" transform="rotate(-90, {label_w/2}, {row_label_y})">{model_name}</text>')
        
        # First column: Arch (for first row) or Label (for last row) or just first sample's architecture
        col_x = label_w + margin
        
        # Get Arch or Label content
        if row_idx == 0:
            # First row shows "Arch"
            arch_content = get_arch_svg_content(scene_dir, bounds)
            header = "Arch"
            content = arch_content if arch_content else '<rect width="120" height="120" fill="white"/>'
        elif row_idx == len(available_models) - 1:
            # Last row shows "Label"
            label_content = get_label_svg_content(scene_dir)
            header = "Label"
            content = label_content if label_content else '<rect width="120" height="120" fill="white"/>'
        else:
            # Middle rows show "Arch" as well
            arch_content = get_arch_svg_content(scene_dir, bounds)
            header = "Arch"
            content = arch_content if arch_content else '<rect width="120" height="120" fill="white"/>'
        
        # Add header only for first row
        if row_idx == 0:
            svg_lines.append(f'<text x="{col_x + cell_w/2}" y="25" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">{header}</text>')
        
        svg_lines.append(f'<g transform="translate({col_x}, {y_offset})">')
        svg_lines.append(f'    <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
        svg_lines.append(f'    {content}')
        svg_lines.append('</g>')
        
        # Sample columns
        npz_files = sorted(scene_dir.glob('boxes_*.npz'), key=lambda x: int(x.stem.split('_')[1]))[:n_samples]
        
        for sample_idx, npz_path in enumerate(npz_files):
            col_x = label_w + margin + (1 + sample_idx) * (cell_w + margin)
            
            # Header for samples (only on first row)
            if row_idx == 0:
                svg_lines.append(f'<text x="{col_x + cell_w/2}" y="25" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Sample {sample_idx}</text>')
            
            # Load and visualize sample
            data = np.load(npz_path, allow_pickle=True)
            sample_content = visualize_scene_to_svg_content(
                translations=data['translations'],
                sizes=data['sizes'],
                angles=data['angles'],
                class_labels=data['class_labels'],
                fixed_bounds=bounds,
                num_partial_boxes=num_partial_boxes
            )
            
            svg_lines.append(f'<g transform="translate({col_x}, {y_offset})">')
            svg_lines.append(f'    {sample_content}')
            svg_lines.append('</g>')
    
    svg_lines.append('</svg>')
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Create comparison SVGs for multiple model settings")
    parser.add_argument('--input_dir', required=True, help="Directory containing model subdirectories (e.g., output/model_eval_2)")
    parser.add_argument('--output_dir', default=None, help="Output directory for comparison SVGs (default: input_dir/comparisons)")
    parser.add_argument('--n_samples', type=int, default=6, help="Number of samples to show per model")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "comparisons"
    
    # Find available model directories
    model_dirs = {}
    for model_name in MODEL_SETTINGS:
        model_path = input_dir / model_name
        if model_path.exists():
            model_dirs[model_name] = model_path
            print(f"Found model: {model_name}")
    
    if not model_dirs:
        print(f"No model directories found in {input_dir}")
        return
    
    # Find all scene IDs across all models
    all_scene_ids = set()
    for model_name, model_path in model_dirs.items():
        for scene_dir in model_path.iterdir():
            if scene_dir.is_dir():
                all_scene_ids.add(scene_dir.name)
    
    print(f"Found {len(all_scene_ids)} unique scenes across {len(model_dirs)} models")
    
    # Create comparison SVGs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for scene_id in tqdm(sorted(all_scene_ids), desc="Creating comparisons"):
        output_path = output_dir / f"{scene_id}_compare.svg"
        if create_comparison_svg(scene_id, model_dirs, output_path, n_samples=args.n_samples):
            success_count += 1
    
    print(f"\nCreated {success_count} comparison SVGs in {output_dir}")


if __name__ == "__main__":
    main()

