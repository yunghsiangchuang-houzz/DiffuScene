
import os
import sys
import json
import glob
import argparse
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Tuple, Any

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autolayout.AutoLayoutLogic import Room, FurnitureItem, ObjectType, WallSegment
from autolayout.AutoLayoutGame import AutoLayoutGame

def _determine_wall_dimensions(size_x: float, size_z: float, yaw: float) -> Tuple[float, float]:
    """Helper to determine wall length/thickness (copied from AutoLayoutLogic)."""
    if size_x < size_z:
        return size_z, size_x
    else:
        return size_x, size_z

def get_object_type_from_item(item_dict: Dict) -> Tuple[ObjectType, str]:
    """
    Map item dict to ObjectType and return specific type string.
    Returns (ObjectType, specific_type_str)
    """
    otype = item_dict.get('object_type', '').lower()
    diff_type = item_dict.get('diffusion_type', '').lower()
    model_name = item_dict.get('model_name', '').lower()
    
    combined = f"{otype} {diff_type} {model_name}"
    
    specific_type = "unknown"
    mapped_type = None

    # Check for specific types in order of precedence
    if 'vanity' in combined:
        specific_type = 'vanity'
        mapped_type = ObjectType.SINK # Vanity counts as Sink for scoring
    elif 'toilet' in combined:
        specific_type = 'toilet'
        mapped_type = ObjectType.TOILET
    elif 'sink' in combined:
        specific_type = 'sink'
        mapped_type = ObjectType.SINK
    elif 'bathtub' in combined:
        specific_type = 'bathtub'
        mapped_type = ObjectType.BATHTUB
    elif 'shower' in combined:
        specific_type = 'shower'
        mapped_type = ObjectType.SHOWER
    elif 'door' in combined or otype == 'door':
        specific_type = 'door'
        mapped_type = ObjectType.DOOR
    elif 'window' in combined or otype == 'window':
        specific_type = 'window'
        mapped_type = ObjectType.WINDOW
    
    return mapped_type, specific_type

def analyze_file(file_path: str, game: AutoLayoutGame) -> Dict[str, Any]:
    """
    Analyze a single JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    
    wall_items = [item for item in items if item.get('object_type') == 'wall']
    
    # Calculate bounds
    all_x, all_z = [], []
    for wall_data in wall_items:
        length, thickness = _determine_wall_dimensions(wall_data['size_x'], wall_data['size_z'], wall_data.get('yaw', 0))
        wall = WallSegment(
            uuid=wall_data.get('uuid', ''), center_x=wall_data['center_x'], center_z=wall_data['center_z'],
            length=length, thickness=thickness, yaw=wall_data.get('yaw', 0)
        )
        corners = wall.get_corners_2d()
        all_x.extend([c[0] for c in corners])
        all_z.extend([c[1] for c in corners])
        
    if not all_x:
        return None

    shift_x, shift_z = min(all_x), min(all_z)
    max_x, max_z = max(all_x) - shift_x, max(all_z) - shift_z
    
    # Initialize Room
    room = Room(max_x, max_z, 2.5, grid_resolution=128)
    room.min_x, room.min_z, room.max_x, room.max_z = 0.0, 0.0, max_x, max_z
    
    # Add Walls
    for wall_data in wall_items:
        length, thickness = _determine_wall_dimensions(wall_data['size_x'], wall_data['size_z'], wall_data.get('yaw', 0))
        wall = WallSegment(
            uuid=wall_data.get('uuid', ''),
            center_x=wall_data['center_x'] - shift_x,
            center_z=wall_data['center_z'] - shift_z,
            length=length, thickness=thickness, yaw=wall_data.get('yaw', 0)
        )
        room.add_wall_segment(wall)
        
    # Counters
    counts = {
        'door': 0, 'window': 0, 'vanity': 0, 
        'toilet': 0, 'bathtub': 0, 'shower': 0
    }
    
    # Process items
    for item_data in items:
        # Skip walls/floors/ceilings for item processing
        if item_data.get('object_type') in ['wall', 'floor', 'ceiling']:
            continue
            
        mapped_type, specific_type = get_object_type_from_item(item_data)
        
        if specific_type in counts:
            counts[specific_type] += 1
            
        if mapped_type:
            # Add to Room
            if mapped_type == ObjectType.DOOR:
                room.add_door(
                    uuid=item_data.get('uuid', ''),
                    center_x=item_data['center_x'] - shift_x,
                    center_z=item_data['center_z'] - shift_z,
                    size_x=item_data['size_x'], size_z=item_data['size_z'],
                    yaw=item_data.get('yaw', 0)
                )
            elif mapped_type == ObjectType.WINDOW:
                room.add_window(
                    uuid=item_data.get('uuid', ''),
                    center_x=item_data['center_x'] - shift_x,
                    center_z=item_data['center_z'] - shift_z,
                    size_x=item_data['size_x'], size_z=item_data['size_z'],
                    yaw=item_data.get('yaw', 0)
                )
            else:
                # Furniture
                fi = FurnitureItem(
                    obj_type=mapped_type,
                    footprint=(item_data['size_x'], item_data['size_z']),
                    uuid=item_data.get('uuid', '')
                )
                fi.yaw = item_data.get('yaw', 0)
                
                room.place_item(
                    fi, 
                    item_data['center_x'] - shift_x, 
                    item_data['center_z'] - shift_z, 
                    force=True
                )

    
    # Calculate Area
    # Room.room_mask is lazy loaded. Force build if needed.
    if room.room_mask is None:
        try:
            # logic.py handles _build_room_mask_from_contours
            room._build_room_mask_from_contours()
        except:
            pass

    area_sqm = 0.0
    if hasattr(room, 'room_mask') and room.room_mask is not None:
         cell_area = ((room.max_x - room.min_x) / room.n) * ((room.max_z - room.min_z) / room.n)
         area_sqm = np.sum(room.room_mask) * cell_area
    
    # Fallback: If area is effectively zero (e.g. broken walls), use Floor items
    if area_sqm < 0.1:
        floor_area_sum = 0.0
        for item in items:
            otype = item.get('object_type', '').lower()
            dtype = item.get('diffusion_type', '').lower()
            if otype == 'floor' or 'floor' in dtype:
                floor_area_sum += item['size_x'] * item['size_z']
        
        if floor_area_sum > 0:
            area_sqm = floor_area_sum

    # Calculate Score
    try:
        score, breakdown = game._calculate_score_with_breakdown(room)
    except Exception as e:
        # print(f"Error scoring {file_path}: {e}")
        score = -999
        breakdown = []

    return {
        'file_id': os.path.basename(os.path.dirname(file_path)),
        'file_path': file_path,
        'area_sqm': round(area_sqm, 2),
        'score': round(score, 2),
        'breakdown': str(breakdown),
        **counts
    }

def check_data(path="data/houzz_takeoff/layout_analysis.csv"):
    data_folder = path.split("/")[-2]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    print(f"Loaded {path}\n")

    output_dir = os.path.dirname(path)
    if output_dir == "":
        output_dir = "."

    for column in ["area_sqm", "score"]:
        holder = df[column].tolist()
        plot_distribution(holder, column, data_folder, output_dir=output_dir)

def plot_distribution(holder, column, data_folder, output_dir=".", precision=3):
    holder = [round(x, precision) for x in holder]
    counter = Counter(holder)
    keys = sorted(counter.keys())
    counts = [counter[k] for k in keys]
    
    plt.figure(figsize=(10, 6))
    plt.barh(keys, counts)
    plt.xlabel('Frequency')
    plt.ylabel('Value')
    plt.title(f'Data Distribution - {column}')
    
    # Save figure instead of showing
    output_path = os.path.join(output_dir, f"distribution_{column}_{data_folder}.png")
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def conduct_count(holder):
    counter = Counter(holder)
    # Print top 10 most common just to be concise, or ask user? User snippet implies all.
    # But for float values (area), this might be huge.
    # User code: for key in sorted(list(counter.keys()), reverse=True): print...
    # I will stick to user request.
    count = 0
    for key in sorted(list(counter.keys()), reverse=True):   
        print(f"{key}: {counter[key]}")
        count += 1
        if count > 20: # Limit output for console readability
            print("... (truncated)")
            break

def main():
    parser = argparse.ArgumentParser(description="Analyze Houzz Layout Data")
    parser.add_argument('--input_data', type=str, help="Path to input data folder (e.g. data/houzz_takeoff)")
    parser.add_argument('--output', type=str, default='layout_analysis.csv', help="Output CSV file path")
    parser.add_argument('--plot', action='store_true', help="Generate plots from existing CSV")
    args = parser.parse_args()

    args = parser.parse_args()

    # 1. Run Analysis if input_data is provided
    if args.input_data:
        search_pattern = os.path.join(args.input_data, '**', 'simple_design_filtered.json')
        files = glob.glob(search_pattern, recursive=True)
        
        print(f"Found {len(files)} files to analyze in {args.input_data}")
        
        game = AutoLayoutGame()
        results = []
        
        for i, file_path in enumerate(files):
            if i % 100 == 0:
                print(f"Processing {i}/{len(files)}...")
                
            try:
                res = analyze_file(file_path, game)
                if res:
                    results.append(res)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                continue
    
        if not results:
            print("No results generated.")
            return
    
        # Save to CSV
        cols = ['file_id', 'area_sqm', 'score', 'door', 'window', 'vanity', 'toilet', 'bathtub', 'shower', 'file_path', 'breakdown']
        
        print(f"Saving analysis of {len(results)} layouts to {args.output}")
        
        # Ensure directory exists
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        
        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cols)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, 0) for k in cols})
                
        # Calculate Summary Stats
        print("\nSummary Statistics:")
        
        numeric_cols = ['area_sqm', 'score', 'door', 'window', 'vanity', 'toilet', 'bathtub', 'shower']
        stats = {}
        
        for col in numeric_cols:
            values = [r[col] for r in results if r[col] != -999]
            if values:
                stats[col] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
                
        # Print formatted stats
        print(f"{'Metric':<15} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
        print("-" * 65)
        for col, stat in stats.items():
            print(f"{col:<15} {stat['mean']:.2f}{'':<6} {stat['median']:.2f}{'':<6} {stat['min']:.2f}{'':<6} {stat['max']:.2f}")

    elif args.plot:
         # No input data, but plot requested. Just proceed to plotting logic.
         pass
    else:
        parser.print_help()
        print("\nError: --input_data is required for analysis (unless --plot is used standalone)")
        return

    # 2. Run Plotting if requested
    if args.plot:
         check_data(args.output)


if __name__ == '__main__':
    main()
