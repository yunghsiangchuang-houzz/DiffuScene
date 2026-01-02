import os
import numpy as np
import json
import argparse
from collections import Counter

def compute_stats(data_dir, output_file):
    print(f"Scanning {data_dir}...")
    
    all_translations = []
    all_sizes = []
    all_angles = []
    class_counts = Counter()
    
    scene_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for d in scene_dirs:
        npz_path = os.path.join(data_dir, d, 'boxes.npz')
        if not os.path.exists(npz_path): continue
        
        data = np.load(npz_path)
        
        # Filter out empty classes (assumed last index)
        classes = data['class_labels']
        translations = data['translations']
        sizes = data['sizes']
        angles = data['angles']
        
        n_points, n_classes = classes.shape
        empty_idx = n_classes - 1 
        
        for i in range(n_points):
            cls_idx = np.argmax(classes[i])
            if cls_idx == empty_idx: continue
            
            all_translations.append(translations[i])
            all_sizes.append(sizes[i])
            all_angles.append(angles[i])
            class_counts[cls_idx] += 1
            
    # Compute bounds
    all_translations = np.array(all_translations)
    all_sizes = np.array(all_sizes)
    all_angles = np.array(all_angles)
    
    min_t = np.min(all_translations, axis=0) if len(all_translations) else np.zeros(3)
    max_t = np.max(all_translations, axis=0) if len(all_translations) else np.zeros(3)
    
    min_s = np.min(all_sizes, axis=0) if len(all_sizes) else np.zeros(3)
    max_s = np.max(all_sizes, axis=0) if len(all_sizes) else np.zeros(3)
    
    min_a = np.min(all_angles) if len(all_angles) else 0
    max_a = np.max(all_angles) if len(all_angles) else 0
    
    
    # Prepare additional stats
    raw_class_names = ["vanity", "toilet", "shower", "tub", "floor", "wall", "door", "window"]
    # Ensure all classes are present in counts
    final_counts = {}
    for i, name in enumerate(raw_class_names):
        final_counts[name] = class_counts.get(i, 0) # Use string key maybe? No, class_counts uses int index keys currently
    
    # Re-map class_counts (int keys) to string keys matching raw_class_names
    # class_counts has keys 0..8 (where 8 is empty).
    # ThreedFront expects count_furniture to be frequencies or counts of object types.
    
    # Construct strictly valid response
    class_labels = raw_class_names + ["empty"]
    object_types = raw_class_names # Exclude empty from object types? Or include?
                                  # Usually object_types are the 'furniture' types. 
                                  # Let's include all for safety or just furniture.
                                  # Given Config class_dim=9, let's keep consistency.
    
    stats = {
        "bounds_translations": list(np.concatenate([min_t, max_t])),
        "bounds_sizes": list(np.concatenate([min_s, max_s])),
        "bounds_angles": [float(min_a), float(max_a)],
        "class_labels": class_labels,
        "object_types": object_types,
        "class_frequencies": {str(k): v for k, v in class_counts.items()},
        "count_furniture": {str(name): class_counts.get(i, 0) for i, name in enumerate(raw_class_names)},
        "class_order": {name: i for i, name in enumerate(class_labels)}
    }
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Stats saved to {output_file}")
    print("Class counts:", stats["class_frequencies"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/houzz_bathroom_processed')
    parser.add_argument('--output_file', default='data/houzz_bathroom_processed/dataset_stats.txt')
    args = parser.parse_args()
    
    compute_stats(args.data_dir, args.output_file)
