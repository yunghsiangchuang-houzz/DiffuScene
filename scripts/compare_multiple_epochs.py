#!/usr/bin/env python3
"""
Script to create comparison grids between multiple epoch results.
Layout: Arch and Label in left column, each epoch gets one row with 6 samples.
"""
import argparse
import os
import re
from pathlib import Path


def extract_svg_content(svg_path):
    """Extract the inner content of an SVG file (everything between <svg> tags)."""
    if not svg_path.exists():
        return None
    
    with open(svg_path, 'r') as f:
        svg_content = f.read()
    
    # Extract content between <svg> tags
    match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
    if match:
        return match.group(1)
    return None


def extract_arch_content(label_content):
    """Extract architectural elements (floor, wall, door, window) from label content."""
    if not label_content:
        return None
    
    # Architectural colors
    arch_colors = [
        "rgb(211,211,211)",  # floor
        "rgb(0,0,153)",      # wall
        "rgb(153,0,0)",      # door
        "rgb(255,153,153)",  # window
    ]
    
    filtered_elements = []
    # Parse each element and keep only arch elements
    for element in re.findall(r'<(?:rect|polygon|path)[^>]*/?>', label_content):
        for color in arch_colors:
            if color in element:
                filtered_elements.append(element)
                break
    
    return '\n'.join(filtered_elements) if filtered_elements else None


def create_multi_epoch_comparison(epoch_dirs, epoch_labels, output_dir, house_id):
    """Create a comparison grid SVG for a specific house_id across multiple epochs."""
    # Verify all epoch directories exist for this house_id
    house_dirs = []
    for epoch_dir in epoch_dirs:
        house_dir = epoch_dir / house_id
        if not house_dir.exists():
            print(f"Warning: Missing directory {house_dir}")
            return False
        house_dirs.append(house_dir)
    
    # Load label SVG (try from first directory)
    label_svg_path = house_dirs[0] / "simple_design_label.svg"
    if not label_svg_path.exists():
        # Try other directories
        for house_dir in house_dirs[1:]:
            label_svg_path = house_dir / "simple_design_label.svg"
            if label_svg_path.exists():
                break
        else:
            print(f"Warning: Missing label SVG for {house_id}")
            return False
    
    label_content = extract_svg_content(label_svg_path)
    if not label_content:
        print(f"Warning: Could not extract label content for {house_id}")
        return False
    
    # Extract arch content from label
    arch_content = extract_arch_content(label_content)
    
    # Collect SVG contents for samples 0-5 from each epoch
    epoch_samples = []
    for house_dir in house_dirs:
        samples = []
        for i in range(6):
            svg_path = house_dir / f"boxes_{i}.svg"
            content = extract_svg_content(svg_path)
            if content:
                samples.append(content)
        if len(samples) < 6:
            print(f"Warning: Only found {len(samples)} samples for {house_id} in {house_dir}")
        epoch_samples.append(samples[:6])  # Take first 6
    
    if not any(epoch_samples):
        print(f"Warning: No samples found for {house_id}")
        return False
    
    # Setup grid dimensions
    num_epochs = len(epoch_dirs)
    num_samples = 6
    cell_w, cell_h = 120, 120
    margin = 20
    label_h = 20
    left_label_w = 80
    
    # Grid layout: 1 reference column (Arch/Label stacked) + 6 sample columns
    cols = num_samples + 1  # Samples + 1 reference column
    rows = num_epochs  # One row per epoch
    
    # Calculate total height: rows * (cell_h + margin + label_h) + margin
    total_h = rows * (cell_h + margin + label_h) + margin
    
    total_w = left_label_w + cols * (cell_w + margin) + margin
    
    # Start building SVG
    svg_lines = [
        f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{total_w}" height="{total_h}" fill="white"/>'
    ]
    
    # Calculate positions
    ref_x_start = left_label_w + margin
    samples_x_start = ref_x_start + (cell_w + margin)
    
    # Reference column: Arch and Label aligned with epoch rows
    ref_start_y = margin + label_h
    row_spacing = cell_h + margin + label_h
    
    # Arch in reference column (aligned with first epoch row)
    arch_y = ref_start_y
    svg_lines.append('<!-- Arch in reference column (aligned with row 1) -->')
    if arch_content:
        svg_lines.append(f'<g transform="translate({ref_x_start}, {arch_y})">')
        svg_lines.append('    <text x="60.0" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Arch</text>')
        svg_lines.append(f'    <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
        arch_lines = arch_content.strip().split('\n')
        for line in arch_lines:
            svg_lines.append(f'    {line}')
        svg_lines.append('</g>')
        svg_lines.append('')
    
    # Label in reference column (aligned with second epoch row)
    label_y = ref_start_y + row_spacing
    svg_lines.append('<!-- Label in reference column (aligned with row 2) -->')
    svg_lines.append(f'<g transform="translate({ref_x_start}, {label_y})">')
    svg_lines.append('    <text x="60.0" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Label</text>')
    svg_lines.append(f'    <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
    label_lines = label_content.strip().split('\n')
    for line in label_lines:
        # Skip the outer rect and svg tags if present
        if ('<rect' in line and 'width="120"' in line and 'height="120"' in line) or '<svg' in line or '</svg>' in line:
            continue
        svg_lines.append(f'    {line}')
    svg_lines.append('</g>')
    svg_lines.append('')
    
    # Rows: Each epoch's samples
    for epoch_idx, (samples, epoch_label) in enumerate(zip(epoch_samples, epoch_labels)):
        row_y = ref_start_y + epoch_idx * row_spacing
        svg_lines.append(f'<!-- Row {epoch_idx + 1}: {epoch_label} -->')
        
        # Samples for this epoch
        svg_lines.append(f'<g transform="translate({samples_x_start}, {row_y})">')
        
        for i in range(min(len(samples), num_samples)):
            x_offset = i * (cell_w + margin)
            svg_lines.append(f'    <!-- Sample {i} -->')
            svg_lines.append(f'    <g transform="translate({x_offset}, 0)">')
            svg_lines.append(f'        <text x="{cell_w/2}" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Sample {i}</text>')
            svg_lines.append(f'        <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
            # Add the SVG content (skip the outer rect if present)
            content_lines = samples[i].strip().split('\n')
            for line in content_lines:
                # Skip the outer rect and svg tags if present
                if '<rect width="120" height="120" fill="white"/>' in line or '<svg' in line or '</svg>' in line:
                    continue
                svg_lines.append(f'        {line}')
            svg_lines.append('    </g>')
            svg_lines.append('')
        
        svg_lines.append('</g>')
        svg_lines.append('')
    
    # Labels on the left for each row
    svg_lines.append('<!-- Labels on the left -->')
    svg_lines.append('<g transform="translate(0, 0)">')
    
    # Epoch row labels
    for epoch_idx, epoch_label in enumerate(epoch_labels):
        row_y = ref_start_y + epoch_idx * row_spacing
        row_center_y = row_y + cell_h / 2
        svg_lines.append(f'    <text x="{left_label_w/2}" y="{row_center_y}" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" transform="rotate(-90, {left_label_w/2}, {row_center_y})">{epoch_label}</text>')
    
    svg_lines.append('</g>')
    svg_lines.append('')
    
    svg_lines.append('</svg>')
    
    # Write to file
    output_dir.mkdir(parents=True, exist_ok=True)
    house_id_short = house_id.split('_')[0]  # Extract just the numeric ID
    output_path = output_dir / f"{house_id_short}_compare.svg"
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))
    
    print(f"Created comparison grid: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create comparison grids between multiple epoch results")
    parser.add_argument('--epoch_dirs', nargs='+', required=True, help="List of epoch directories")
    parser.add_argument('--epoch_labels', nargs='+', required=True, help="List of labels for each epoch")
    parser.add_argument('--output_dir', required=True, help="Output directory for comparison grids")
    parser.add_argument('--house_id', help="Specific house_id to process (if not provided, processes all)")
    
    args = parser.parse_args()
    
    if len(args.epoch_dirs) != len(args.epoch_labels):
        print(f"Error: Number of epoch directories ({len(args.epoch_dirs)}) must match number of labels ({len(args.epoch_labels)})")
        return
    
    epoch_dirs = [Path(d) for d in args.epoch_dirs]
    output_dir = Path(args.output_dir)
    
    # Verify all directories exist
    for epoch_dir in epoch_dirs:
        if not epoch_dir.exists():
            print(f"Error: Directory does not exist: {epoch_dir}")
            return
    
    # Get list of house_ids
    if args.house_id:
        house_ids = [args.house_id]
    else:
        # Get intersection of all house_ids from all directories
        house_ids_sets = []
        for epoch_dir in epoch_dirs:
            house_ids_set = {d.name for d in epoch_dir.iterdir() if d.is_dir()}
            house_ids_sets.append(house_ids_set)
        
        # Find common house_ids across all directories
        house_ids = sorted(set.intersection(*house_ids_sets))
    
    print(f"Processing {len(house_ids)} house_ids...")
    
    for house_id in house_ids:
        create_multi_epoch_comparison(epoch_dirs, args.epoch_labels, output_dir, house_id)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

