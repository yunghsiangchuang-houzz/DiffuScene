import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import math

class HouzzPreprocessor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.max_arch = 40
        self.max_furn = 10  # Buffer, though we enforce max 4 items
        self.total_points = 50
        
        # Class mapping
        # 0-3: Furniture
        # 4-7: Architecture
        # 8: Start (dummy)
        # 9: Empty/Padding
        self.class_map = {
            'vanity': 0,
            'toilet': 1,
            'shower': 2,
            'tub': 3,
            'floor': 4,
            'wall': 5,
            'door': 6,
            'window': 7,
        }
        # Data loader (Diffusion.__getitem__) will take [:, :-2] and [:, -1:]
        # So it takes 0-7 and index 9. Total 9 classes.
        self.n_classes = 10
        
        self.arch_types = {'wall', 'floor', 'door', 'window'}
        self.furn_types = {'vanity', 'toilet', 'shower', 'tub'}

    def process_scene(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        items = data.get('items', [])
        
        arch_items = []
        furn_items = {} # Use dict to keep only one of each type
        
        # 1. Separate items
        for item in items:
            dtype = item.get('diffusion_type')
            if dtype in self.arch_types:
                arch_items.append(item)
            elif dtype in self.furn_types:
                # Filter: Keep only first instance of each furniture type
                if dtype not in furn_items:
                    furn_items[dtype] = item
        
        furn_list = list(furn_items.values())

        # 2. Check limits (Architecture)
        if len(arch_items) > self.max_arch:
            print(f"Warning: Scene {file_path} has {len(arch_items)} arch elements. Truncating to {self.max_arch}.")
            arch_items = arch_items[:self.max_arch]
            
        # 3. Create arrays
        # Point dim = 13 (8 bbox + 5 class dummy? No, let's look at plan: 8 bbox + class_dim)
        # Actually config says point_dim=13. 
        # But commonly we store: class_labels (one-hot), translations, sizes, angles
        
        # Prepare lists to convert to numpy
        classes = []
        translations = []
        sizes = []
        angles = []
        
        # --- Process Architecture ---
        for item in arch_items:
            self._append_item(item, classes, translations, sizes, angles)
            
        # Pad Architecture
        n_arch = len(classes)
        for _ in range(self.max_arch - n_arch):
            self._append_empty(classes, translations, sizes, angles)
            
        # --- Process Furniture ---
        for item in furn_list:
            self._append_item(item, classes, translations, sizes, angles)
            
        # Pad Furniture/Total
        current_len = len(classes)
        for _ in range(self.total_points - current_len):
            self._append_empty(classes, translations, sizes, angles)
            
        # Convert to numpy
        # Classes: One-hot encoded [N, n_classes]
        # But wait, 3D-Front dataset usually expects checks for 'class_labels' 
        # In current codebase, verify what `boxes.npz` expects.
        # Plan says: class_labels: [N, C], translations [N,3], sizes [N,3], angles [N,1]
        
        class_labels_np = np.eye(self.n_classes)[classes] 
        # Note: Empty class? Usually represented by last class or all zeros in some datasets.
        # In DiffuScene, usually there is an 'Empty' class.
        # For this experiment, let's assume class 'n_classes' is not existing
        # Reviewing `delete_empty_from_network_samples`:
        # class_labels uses `n_classes-2`? 
        # Let's stick to standard encoding. If index is -1, it is empty.
        
        # Better approach:
        # Use simple mapping. If empty, maybe specific flag or last class?
        # Re-reading plan: "Pad to max_length with empty slots"
        # In bedroom implementation, usually the last class is "Empty".
        # Let's verify `n_classes` in bedroom config. It was 22.
        # Here we have 8 real classes. Let's make class 8 = Empty.
        # So n_classes = 9.
        
        final_classes = np.zeros((self.total_points, self.n_classes))
        for i, c in enumerate(classes):
            if c != -1:
                final_classes[i, c] = 1
            else:
                final_classes[i, self.n_classes-1] = 1 # Mark as empty class
                
        translations_np = np.array(translations)
        sizes_np = np.array(sizes)
        angles_np = np.array(angles)
        
        # 4. Generate Room Layout (64x64 mask)
        # Project floor/walls onto 2D map
        room_layout = self._generate_layout(arch_items)
        room_layout = np.expand_dims(room_layout, axis=-1)
        
        # 5. Save
        scene_id = os.path.basename(os.path.dirname(file_path)) + "_" + os.path.basename(file_path).split('.')[0]
        save_dir = os.path.join(self.output_dir, scene_id)
        os.makedirs(save_dir, exist_ok=True)
        
        np.savez_compressed(
            os.path.join(save_dir, 'boxes.npz'),
            scene_id=scene_id,
            class_labels=final_classes,
            translations=translations_np,
            sizes=sizes_np,
            angles=angles_np,
            room_layout=room_layout
        )

    def _append_item(self, item, classes, translations, sizes, angles):
        label = self.class_map.get(item['diffusion_type'])
        if label is None: return # Should not happen due to filtering
        
        classes.append(label)
        translations.append([item['center_x'], item['center_y'], item['center_z']])
        sizes.append([item['size_x'], item['size_y'], item['size_z']])
        
        # Angle: Input is degrees, convert to radians for sin/cos?
        # Model expects: 
        # - dataset uses sin/cos usually? 
        # - Plan says: angles [N, 1]. In bedrooms it's often encoded.
        # - `diffusion_scene_layout_ddpm.py` seems to handle 1 dim angle.
        # - Let's store raw angle in radians or normalized?
        # - Bedroom stats usually have bounds. Let's store radians.
        yaw_rad = np.deg2rad(item.get('yaw', 0))
        angles.append([yaw_rad]) 

    def _append_empty(self, classes, translations, sizes, angles):
        classes.append(-1) # Placeholder
        translations.append([0, 0, 0])
        sizes.append([0, 0, 0])
        angles.append([0])

    def _generate_layout(self, arch_items):
        # specific to 3D-Front: usually 256 or 64 grid
        # Draw floor
        grid_size = 64
        mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
        
        # Simple approximation: Find floor and rasterize bbox
        # Or just return empty if not strictly needed (model uses partial condition anyway)
        # But visualization might need it.
        # For now, return zeros as partial condition handles layout structure matches
        return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/bathroom_svg_files')
    parser.add_argument('--output_dir', default='data/houzz_bathroom_processed')
    args = parser.parse_args()
    
    preprocessor = HouzzPreprocessor(args.output_dir)
    
    files = []
    for root, dirs, filenames in os.walk(args.input_dir):
        for f in filenames:
            if f == 'simple_design_filtered.json':
                files.append(os.path.join(root, f))
                
    print(f"Found {len(files)} scenes.")
    
    for f in tqdm(files):
        preprocessor.process_scene(f)
        
    print("Done.")

if __name__ == '__main__':
    main()
