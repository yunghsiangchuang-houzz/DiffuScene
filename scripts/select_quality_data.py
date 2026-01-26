
import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Select quality data based on analysis CSV")
    parser.add_argument('--data_folder', default="./data/houzz_takeoff", type=str, help="Path to input data folder (e.g. ./data/houzz_takeoff)")
    parser.add_argument('--output_folder', default="./data/high_quality_data", type=str, help="Path to output folder for selected data")
    parser.add_argument('--score_threshold', default=0.5, type=float, help="Minimum score for selected data")
    parser.add_argument('--high_quality', default=True, type=bool, help="get low or high quality data")
    args = parser.parse_args()

    input_csv = os.path.join(args.data_folder, 'layout_analysis.csv')
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Please run analyze_houzz_data.py first.")
        return

    print(f"Loading analysis from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Total files in CSV: {len(df)}")

    # Ensure output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # --- Filtering Logic ---
    # Global Criteria
    # 1. Contains at least one door.
    # 2. Not allowed to put two sink (sink + vanity <= 1)
    # 3. Score > -0.9 (User req 4)
    
    # Note: 'sink' in CSV counts "sink", 'vanity' counts "vanity".
    # analyze_houzz_data adds vanity to SINK object type for scoring, but counts are separate.
    sinks = df['sink'] if 'sink' in df.columns else 0
    df['total_sinks'] = sinks + df['vanity']
    df['total_bath_shower'] = df['bathtub'] + df['shower']

    # Apply Filters
    mask = (df['door'] >= 1) & (df['toilet'] <= 1)
    # Exclude rows where bathtub and shower collide
    collision_mask = ~df['breakdown'].astype(str).str.contains('BATHTUB ↔ SHOWER')
    mask &= collision_mask
    bathtub_dup_mask = ~df['breakdown'].astype(str).str.contains('BATHTUB ↔ BATHTUB')
    mask &= bathtub_dup_mask
    sink_dup_mask = ~df['breakdown'].astype(str).str.contains('SINK ↔ SINK')
    mask &= sink_dup_mask
    if args.high_quality:
        mask &= (df['score'] > args.score_threshold)
    else:
        mask &= (df['score'] <= args.score_threshold)
    selected_df = df[mask].copy()
    
    # Area-based Criteria
    # 2. 3 <= area <= 5, sink >= 1, toilet >= 1
    mask_medium = (
        (selected_df['area_sqm'] >= 2.0) & 
        (selected_df['area_sqm'] <= 5.0) &
        (selected_df['total_sinks'] >= 1) &
        (selected_df['toilet'] == 1)
    )
    
    # 3. area > 5, sink >= 1, toilet >= 1, (bath or shower) >= 1
    mask_large = (
        (selected_df['area_sqm'] > 5.0) &
        (selected_df['area_sqm'] < 25.0) &
        (selected_df['total_sinks'] >= 1) &
        (selected_df['toilet'] == 1) &
        (selected_df['total_bath_shower'] >= 1)
    )
    
    # Combine masks
    final_selection = selected_df[mask_medium | mask_large]
    
    print(f"Selected {len(final_selection)} files matching criteria.")
    
    if len(final_selection) == 0:
        print("No files matched the criteria.")
        return

    # --- Copying Data ---
    print(f"Copying data to {args.output_folder}...")
    
    count = 0
    for _, row in tqdm(final_selection.iterrows(), total=len(final_selection)):
        src_file = row['file_path']
        
        # Source is like: .../12345_takeoff_1/simple_design_filtered.json
        # Check if src exists (CSV might be stale)
        if not os.path.exists(src_file):
            print(f"Warning: Source file not found: {src_file}")
            continue
            
        src_dir = os.path.dirname(src_file)
        dir_name = os.path.basename(src_dir)
        
        dst_dir = os.path.join(args.output_folder, dir_name)
        
        if os.path.exists(dst_dir):
            # User check: do not remove existing data
            # print(f"Skipping {dir_name} (already exists)")
            continue
            
        try:
             shutil.copytree(src_dir, dst_dir)
             count += 1
        except Exception as e:
            print(f"Error copying {src_dir} to {dst_dir}: {e}")
            
    print(f"Successfully copied {count} datasets.")
    
    # Optionally, verify distribution of selected data?
    print("\nSelection Distribution (by Area):")
    print(final_selection['area_sqm'].describe())

if __name__ == "__main__":
    main()
