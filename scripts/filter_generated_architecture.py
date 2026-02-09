import argparse
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Class labels matching visualize_houzz_results.py
CLASS_LABELS = ["vanity", "toilet", "shower", "tub", "floor", "wall", "door", "window", "empty"]
EMPTY_CLASS_IDX = CLASS_LABELS.index('empty')

def get_num_partial_boxes(json_path):
    """
    Count architecture elements (floor, wall, door, window) from JSON file.
    Returns the number of partial boxes (architecture elements).
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r') as f_json:
            scene_data = json.load(f_json)
        json_items = scene_data['items'] if isinstance(scene_data, dict) else scene_data
        
        # Count architecture elements (floor, wall, door, window)
        arch_types = {'floor', 'wall', 'door', 'window'}
        num_partial_boxes = sum(1 for item in json_items if item.get('diffusion_type', '') in arch_types)
        return num_partial_boxes
    except Exception as e:
        print(f"Warning: Could not read {json_path}: {e}")
        return None

def remove_generated_architecture_and_empty(npz_data, num_partial_boxes):
    """
    Remove generated architecture (classes 4-7) beyond num_partial_boxes and all empty boxes.
    
    In the diffusion model format, empty boxes are represented by all -1 values.
    Non-empty boxes have one value as 1 and the rest as -1.
    
    Args:
        npz_data: Dictionary containing 'class_labels', 'translations', 'sizes', 'angles', etc.
        num_partial_boxes: Number of partial boxes (architecture). Architecture beyond this is removed.
    
    Returns:
        Modified npz_data dictionary with empty boxes removed
    """
    if num_partial_boxes is None:
        return npz_data
    
    class_labels = npz_data['class_labels'].copy()  # Make a copy to avoid modifying original
    
    # Build a mask for boxes to keep
    n_objs = class_labels.shape[0]
    keep_mask = np.ones(n_objs, dtype=bool)
    
    # Handle different formats
    if len(class_labels.shape) > 1 and class_labels.shape[1] > 1:
        # Multi-class format [N, C] - could be one-hot (0/1) or diffusion format (-1/1)
        # Check if it's diffusion format (contains -1 values)
        is_diffusion_format = np.any(class_labels < 0)
        
        if is_diffusion_format:
            # Diffusion format: empty = all -1, non-empty = one 1, rest -1
            for i in range(n_objs):
                # Check if already empty (all -1)
                if np.all(class_labels[i] == -1):
                    keep_mask[i] = False
                    continue
                
                # Get class index (argmax of the row)
                cls_idx = np.argmax(class_labels[i])
                is_arch = cls_idx >= 4 and cls_idx <= 7  # floor, wall, door, window
                is_generated = i >= num_partial_boxes
                
                # Remove generated architecture
                if is_arch and is_generated:
                    keep_mask[i] = False
        else:
            # Standard one-hot format (0/1)
            cls_indices = np.argmax(class_labels, axis=-1)
            empty_idx = CLASS_LABELS.index('empty')
            
            for i in range(n_objs):
                cls_idx = cls_indices[i]
                # Remove empty boxes
                if cls_idx == empty_idx:
                    keep_mask[i] = False
                    continue
                
                # Remove generated architecture
                is_arch = cls_idx >= 4 and cls_idx <= 7
                is_generated = i >= num_partial_boxes
                if is_arch and is_generated:
                    keep_mask[i] = False
    else:
        # Index format [N]
        cls_indices = class_labels.flatten()
        empty_idx = CLASS_LABELS.index('empty')
        
        for i in range(n_objs):
            cls_idx = cls_indices[i]
            # Remove empty boxes
            if cls_idx == empty_idx:
                keep_mask[i] = False
                continue
            
            # Remove generated architecture
            is_arch = cls_idx >= 4 and cls_idx <= 7
            is_generated = i >= num_partial_boxes
            if is_arch and is_generated:
                keep_mask[i] = False
    
    # Remove boxes from all arrays
    for key in ['class_labels', 'translations', 'sizes', 'angles']:
        if key in npz_data:
            npz_data[key] = npz_data[key][keep_mask]
    
    # Also remove from other fields if they exist (like objfeats, etc.)
    for key in npz_data.keys():
        if key not in ['class_labels', 'translations', 'sizes', 'angles', 'scene_id', 'room_layout']:
            arr = npz_data[key]
            if isinstance(arr, np.ndarray) and len(arr.shape) > 0 and arr.shape[0] == n_objs:
                npz_data[key] = arr[keep_mask]
    
    return npz_data

def main():
    parser = argparse.ArgumentParser(
        description="Remove generated architecture and empty boxes from npz files"
    )
    parser.add_argument('--input_dir', required=True, 
                       help="Directory containing scene subdirectories with boxes_*.npz files")
    parser.add_argument('--dry_run', action='store_true',
                       help="Print what would be changed without modifying files")
    args = parser.parse_args()
    
    # 1. Group files by directory
    scene_dirs = {}
    for root, dirs, files in os.walk(args.input_dir):
        npz_files = [f for f in files if f.startswith('boxes') and f.endswith('.npz')]
        if npz_files:
            scene_dirs[Path(root)] = sorted(npz_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(f"Found {len(scene_dirs)} scenes with npz files.")
    
    total_modified = 0
    pbar = tqdm(scene_dirs.items(), desc="Processing scenes", unit="scene")
    for scene_dir, npz_filenames in pbar:
        house_id = scene_dir.name
        pbar.set_description(f"Processing {house_id}")
        
        # Get num_partial_boxes from JSON
        original_json_path = scene_dir / "simple_design_filtered.json"
        num_partial_boxes = get_num_partial_boxes(original_json_path)
        
        if num_partial_boxes is None:
            tqdm.write(f"  Warning: Could not determine num_partial_boxes for {house_id}, skipping")
            continue
        
        # Process each npz file
        for npz_filename in npz_filenames:
            npz_path = scene_dir / npz_filename
            
            # Load npz file
            npz_data = dict(np.load(npz_path, allow_pickle=True))
            
            # Count how many boxes would be removed
            class_labels = npz_data['class_labels']
            n_objs = class_labels.shape[0]
            remove_count = 0
            
            if len(class_labels.shape) > 1 and class_labels.shape[1] > 1:
                # Check if it's diffusion format
                is_diffusion_format = np.any(class_labels < 0)
                if is_diffusion_format:
                    # Count empty boxes (all -1) and generated architecture
                    for i in range(n_objs):
                        if np.all(class_labels[i] == -1):
                            remove_count += 1
                            continue
                        cls_idx = np.argmax(class_labels[i])
                        is_arch = cls_idx >= 4 and cls_idx <= 7
                        is_generated = i >= num_partial_boxes
                        if is_arch and is_generated:
                            remove_count += 1
                else:
                    # Standard one-hot format
                    cls_indices = np.argmax(class_labels, axis=-1)
                    empty_idx = CLASS_LABELS.index('empty')
                    for i in range(n_objs):
                        cls_idx = cls_indices[i]
                        if cls_idx == empty_idx:
                            remove_count += 1
                            continue
                        is_arch = cls_idx >= 4 and cls_idx <= 7
                        is_generated = i >= num_partial_boxes
                        if is_arch and is_generated:
                            remove_count += 1
            else:
                # Index format
                cls_indices = class_labels.flatten()
                empty_idx = CLASS_LABELS.index('empty')
                for i in range(n_objs):
                    cls_idx = cls_indices[i]
                    if cls_idx == empty_idx:
                        remove_count += 1
                        continue
                    is_arch = cls_idx >= 4 and cls_idx <= 7
                    is_generated = i >= num_partial_boxes
                    if is_arch and is_generated:
                        remove_count += 1
            
            if remove_count > 0:
                total_modified += remove_count
                if args.dry_run:
                    tqdm.write(f"  Would remove {remove_count} boxes (generated arch + empty) from {npz_filename}")
                else:
                    # Remove generated architecture and empty boxes
                    npz_data = remove_generated_architecture_and_empty(npz_data, num_partial_boxes)
                    # Count how many boxes were removed
                    original_count = n_objs
                    new_count = npz_data['class_labels'].shape[0]
                    removed_count = original_count - new_count
                    # Save back to npz (overwrite original)
                    # save_path = npz_path.with_suffix('.filtered.npz')
                    np.savez(npz_path, **npz_data)
                    tqdm.write(f"  Removed {removed_count} boxes (generated arch + empty) from {npz_filename}")
    
    print(f"\nDone. Total boxes removed: {total_modified}")

if __name__ == "__main__":
    main()

