import argparse
import os
import numpy as np
import json
import math
from pathlib import Path
from tqdm import tqdm

# Color palette from floorplan2dataset_v3.py
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
    "empty": [255, 255, 255] # Explicit empty class
}

# Class labels matching compute_houzz_stats.py
CLASS_LABELS = ["vanity", "toilet", "shower", "tub", "floor", "wall", "door", "window", "empty"]

def rotate_point(x, z, yaw_rad):
    """Rotate a point around origin by yaw angle (in radians)"""
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    
    x_rot = x * cos_yaw - z * sin_yaw
    z_rot = x * sin_yaw + z * cos_yaw
    
    return x_rot, z_rot

def get_bounding_box_corners(center, size, angle):
    """
    center: [x, y, z]
    size: [w, h, d] -> [x_size, y_size, z_size]
    angle: yaw in radians
    """
    cx, _, cz = center
    sx, _, sz = size
    
    # Half dimensions
    half_sx = sx / 2
    half_sz = sz / 2
    
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
    """
    Convert world coordinates to pixel coordinates.
    Matches the logic from floorplan2dataset_v3.py
    """
    min_x, max_x, min_z, max_z = bounds
    width, height = img_size
    
    # Calculate scale based on unit_length (matching original)
    world_width = max_x - min_x
    world_height = max_z - min_z
    
    scale = unit_length
    
    # Center the floor plan
    offset_x = (width - world_width * scale) / 2
    offset_y = (height - world_height * scale) / 2
    
    pixel_coords = []
    for x, z in world_coords:
        px = (x - min_x) * scale + offset_x
        py = (z - min_z) * scale + offset_y
        pixel_coords.append((px, py))
    
    return pixel_coords, scale

def compute_scene_bounds(boxes, valid_mask):
    """
    Compute bounds of the entire scene to frame it exactly like the original.
    """
    all_points = []
    
    translations = boxes['translations']
    sizes = boxes['sizes']
    angles = boxes['angles']
    
    for i in range(len(translations)):
        if not valid_mask[i]: continue
        
        t = translations[i]
        s = sizes[i]
        a = angles[i]
        # Handle different angle shapes [1] or scalar
        if hasattr(a, "__len__") and len(a) > 0:
            angle_val = a[0]
        else:
            angle_val = a
            
        # Get exact corners using rotation
        corners = get_bounding_box_corners(t, s, angle_val)
        all_points.extend(corners)
        
    if not all_points:
        return (-1, 1, -1, 1)
        
    xs = [p[0] for p in all_points]
    zs = [p[1] for p in all_points]
    
    return (min(xs), max(xs), min(zs), max(zs))

def visualize_scene_to_svg(translations, sizes, angles, class_labels, fixed_bounds=None, num_partial_boxes=None):
    """
    Create SVG visualization from scene parameters.
    
    Args:
        translations: (N, 3) array of object positions
        sizes: (N, 3) array of object sizes  
        angles: (N, 1) or (N, 2) array of object rotations
        class_labels: (N, C) one-hot or (N,) index array of class labels
        fixed_bounds: Optional (min_x, max_x, min_z, max_z) to use for alignment
        num_partial_boxes: Number of partial boxes (architecture). Architecture beyond this is filtered.
    
    Returns:
        str: SVG content
    """
    # Handle class labels
    if len(class_labels.shape) > 1 and class_labels.shape[1] > 1:
        # One-hot
        cls_indices = np.argmax(class_labels, axis=-1)
    else:
        cls_indices = class_labels.flatten()
    
    # Handle angles
    # If angles are (N, 1), flatten to (N,)
    # If angles are (N, 2) (cos, sin), convert to (N,) radians
    if len(angles.shape) > 1:
        if angles.shape[1] == 1:
            angles = angles.flatten()
        elif angles.shape[1] == 2:
            angles = np.arctan2(angles[:, 1], angles[:, 0])
    
    n_objs = len(cls_indices)
    
    # Filter empty
    empty_idx = CLASS_LABELS.index('empty')
    valid_mask = cls_indices != empty_idx
    
    # Filter generated architecture (only show partial architecture + furniture)
    # Architecture classes: floor=4, wall=5, door=6, window=7
    # Furniture classes: vanity=0, toilet=1, shower=2, tub=3
    if num_partial_boxes is not None:
        for i in range(n_objs):
            cls_idx = cls_indices[i]
            is_arch = cls_idx >= 4 and cls_idx <= 7  # floor, wall, door, window
            is_generated = i >= num_partial_boxes
            if is_arch and is_generated:
                valid_mask[i] = False  # Filter out generated architecture
    
    # Also filter zero-size boxes (degenerate)
    size_magnitude = np.sqrt(np.sum(sizes ** 2, axis=1))
    has_size = size_magnitude > 0.01
    valid_mask = valid_mask & has_size
    
    # Use fixed bounds if provided, otherwise compute from objects
    if fixed_bounds is not None:
        bounds = fixed_bounds
    else:
        bounds = compute_scene_bounds({
            'translations': translations,
            'sizes': sizes,
            'angles': angles
        }, valid_mask)
    
    # Setup SVG - Use same dimensions as floorplan2dataset_v3.py (120x120)
    W, H = 120, 120
    svg_lines = [
        f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{W}" height="{H}" fill="white"/>'
    ]
    
    indices = [i for i in range(n_objs) if valid_mask[i]]
    
    # Define priorities (higher draws later)
    def get_priority(cls_idx):
        name = CLASS_LABELS[cls_idx]
        if name == 'floor': return 0
        if name in ['wall', 'window', 'door']: return 1
        return 2
        
    indices.sort(key=lambda i: get_priority(cls_indices[i]))
    
    for i in indices:
        cls_idx = cls_indices[i]
        label = CLASS_LABELS[cls_idx]
        
        # Get styling
        color = COLOR_PALETTE.get(label, [0,0,0])
        rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"
        
        # Get corners in world space
        t = translations[i]
        s = sizes[i]
        a = angles[i]
        if isinstance(a, np.ndarray) and a.size > 0: a = a.item()
            
        corners_world = get_bounding_box_corners(t, s, a)
        
        # To pixel (use unit_length=10 to match floorplan2dataset_v3.py)
        corners_px, scale = world_to_pixel(corners_world, bounds, (W, H), unit_length=10)
        
        # Points string
        pts_str = " ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in corners_px])
        
        svg_lines.append(
            f'<polygon points="{pts_str}" fill="{rgb_str}" stroke="{rgb_str}" data-type="{label}" />'
        )
        
    svg_lines.append('</svg>')
    
    return "\n".join(svg_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Directory containing scene subdirectories with boxes_*.npz files")
    args = parser.parse_args()
    
    # 1. Group files by directory
    scene_dirs = {}
    for root, dirs, files in os.walk(args.input_dir):
        npz_files = [f for f in files if f.startswith('boxes') and f.endswith('.npz')]
        if npz_files:
            # Sort by the number in boxes_N.npz
            scene_dirs[Path(root)] = sorted(npz_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            
    print(f"Found {len(scene_dirs)} scenes with generations.")
    
    pbar = tqdm(scene_dirs.items(), desc="Processing scenes", unit="scene")
    for scene_dir, npz_filenames in pbar:
        house_id = scene_dir.name
        pbar.set_description(f"Processing {house_id}")
        
        # Determine bounds and count partial boxes from original JSON if possible
        original_json_path = scene_dir / "simple_design_filtered.json"
        bounds = None
        num_partial_boxes = None  # Number of architecture elements in partial scene
        if original_json_path.exists():
            try:
                with open(original_json_path, 'r') as f_json:
                    scene_data = json.load(f_json)
                json_items = scene_data['items'] if isinstance(scene_data, dict) else scene_data
                all_json_points = []
                
                # Count architecture elements (floor, wall, door, window)
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
                        all_json_points.append((cx + rx, cz + rz))
                if all_json_points:
                    xs = [p[0] for p in all_json_points]
                    zs = [p[1] for p in all_json_points]
                    bounds = (min(xs), max(xs), min(zs), max(zs))
            except Exception:
                pass

        # 2. Generate individual SVGs
        gen_contents = []
        for npz_filename in npz_filenames:
            npz_path = scene_dir / npz_filename
            data = np.load(npz_path, allow_pickle=True)
            
            translations = data['translations']
            sizes = data['sizes']
            angles = data['angles']
            class_labels = data['class_labels']
            
            svg_content = visualize_scene_to_svg(
                translations=translations,
                sizes=sizes,
                angles=angles,
                class_labels=class_labels,
                fixed_bounds=bounds,
                num_partial_boxes=num_partial_boxes
            )
            
            svg_path = scene_dir / npz_filename.replace('.npz', '.svg')
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            
            # Keep inner content for grid
            import re
            match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
            if match:
                gen_contents.append((npz_filename.replace('boxes_', '').replace('.npz', ''), match.group(1)))

        # 3. Create Multi-Comparison Grid
        grid_items = []
        
        # Add Arch first
        arch_svg_path = scene_dir / "simple_design_arch.svg"
        arch_content = None
        if arch_svg_path.exists():
            with open(arch_svg_path, 'r') as f_arch:
                arch_svg = f_arch.read()
            import re
            arch_match = re.search(r'<svg[^>]*>(.*)</svg>', arch_svg, re.DOTALL)
            if arch_match:
                arch_content = arch_match.group(1)
        else:
            # Fallback: use simple_design_label.svg but filter to only arch elements
            label_svg_path = scene_dir / "simple_design_label.svg"
            if label_svg_path.exists():
                with open(label_svg_path, 'r') as f_label:
                    label_svg = f_label.read()
                import re
                label_match = re.search(r'<svg[^>]*>(.*)</svg>', label_svg, re.DOTALL)
                if label_match:
                    # Filter to only keep floor, wall, door, window colors
                    arch_colors = [
                        "rgb(211,211,211)",  # floor
                        "rgb(0,0,153)",      # wall
                        "rgb(153,0,0)",      # door
                        "rgb(255,153,153)",  # window
                    ]
                    filtered_elements = []
                    # Parse each element and keep only arch elements
                    for element in re.findall(r'<(?:rect|polygon|path)[^>]*/?>', label_match.group(1)):
                        for color in arch_colors:
                            if color in element:
                                filtered_elements.append(element)
                                break
                    arch_content = '\n'.join(filtered_elements)
        
        if arch_content:
            grid_items.append(("Arch", arch_content))
        
        # Add Generations
        for name, content in gen_contents:
            grid_items.append((f"Sample {name}", content))
            
        # Add Original last
        original_svg_path = scene_dir / "simple_design_label.svg"
        if original_svg_path.exists():
            with open(original_svg_path, 'r') as f_orig:
                original_svg = f_orig.read()
            import re
            orig_match = re.search(r'<svg[^>]*>(.*)</svg>', original_svg, re.DOTALL)
            if orig_match:
                grid_items.append(("Label", orig_match.group(1)))
        
        if grid_items:
            # Setup Grid
            cols = 4
            cell_w, cell_h = 120, 120
            margin = 20
            label_h = 20
            
            num_items = len(grid_items)
            num_rows = math.ceil(num_items / cols)
            
            # Legend height
            legend_h = 60
            
            total_w = cols * (cell_w + margin) + margin
            total_h = num_rows * (cell_h + margin + label_h) + margin + legend_h
            
            grid_svg = [f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">',
                       f'<rect width="{total_w}" height="{total_h}" fill="white"/>']
            
            for i, (label, content) in enumerate(grid_items):
                r = i // cols
                c = i % cols
                
                x = c * (cell_w + margin) + margin
                y = r * (cell_h + margin + label_h) + margin + label_h
                
                grid_svg.append(f'''<g transform="translate({x}, {y})">
                    <text x="{cell_w/2}" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">{label}</text>
                    {content}
                </g>''')
            
            # Add Legend
            legend_y = num_rows * (cell_h + margin + label_h) + margin
            svg_legend = [f'<g transform="translate({margin}, {legend_y})">']
            svg_legend.append('<text x="0" y="0" font-family="Arial" font-size="10" font-weight="bold">Legend:</text>')
            
            item_per_row = 4
            rect_w, rect_h = 15, 10
            spacing_x, spacing_y = 120, 15
            
            legend_data = [ (k, v) for k, v in COLOR_PALETTE.items() if k not in ["void", "empty"] ]
            for i, (name, color) in enumerate(legend_data):
                row_idx = i // item_per_row
                col_idx = i % item_per_row
                lx = col_idx * spacing_x
                ly = 10 + row_idx * spacing_y
                rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"
                svg_legend.append(f'''
                    <rect x="{lx}" y="{ly}" width="{rect_w}" height="{rect_h}" fill="{rgb_str}" stroke="black" stroke-width="0.5"/>
                    <text x="{lx + rect_w + 5}" y="{ly + 9}" font-family="Arial" font-size="9">{name}</text>
                ''')
            svg_legend.append('</g>')
            grid_svg.extend(svg_legend)
            
            grid_svg.append('</svg>')
            
            comparison_path = scene_dir / "comparison_grid.svg"
            with open(comparison_path, 'w') as f_grid:
                f_grid.write('\n'.join(grid_svg))

    print("\nDone visualization.")

if __name__ == "__main__":
    main()
