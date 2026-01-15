#!/usr/bin/env python3
"""
Script to create comparison grids between different epoch results.
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


def create_comparison_grid(dir1, dir2, output_dir, house_id, label1="epoch 40k", label2="epoch 60k"):
    """Create a comparison grid SVG for a specific house_id."""
    house_dir1 = dir1 / house_id
    house_dir2 = dir2 / house_id
    
    if not house_dir1.exists() or not house_dir2.exists():
        print(f"Warning: Missing directories for {house_id}")
        return False
    
    # Load label SVG (try from either directory)
    label_svg_path = house_dir1 / "simple_design_label.svg"
    if not label_svg_path.exists():
        label_svg_path = house_dir2 / "simple_design_label.svg"
    
    label_content = extract_svg_content(label_svg_path)
    if not label_content:
        print(f"Warning: Missing label SVG for {house_id}")
        return False
    
    # Extract arch content from label
    arch_content = extract_arch_content(label_content)
    
    # Collect SVG contents for samples 0-5
    samples1 = []
    samples2 = []
    
    for i in range(6):
        svg1 = house_dir1 / f"boxes_{i}.svg"
        svg2 = house_dir2 / f"boxes_{i}.svg"
        
        content1 = extract_svg_content(svg1)
        content2 = extract_svg_content(svg2)
        
        if content1:
            samples1.append(content1)
        if content2:
            samples2.append(content2)
    
    if not samples1 or not samples2:
        print(f"Warning: Missing samples for {house_id}")
        return False
    
    # Setup grid dimensions
    num_samples = min(len(samples1), len(samples2), 6)
    cell_w, cell_h = 120, 120
    margin = 20
    label_h = 20
    left_label_w = 80
    
    # Calculate number of reference columns (just 1 for arch or label)
    num_ref_cols = 1
    
    cols = num_samples + num_ref_cols  # Samples + 1 reference column
    rows = 2  # Two rows for comparison
    
    total_w = left_label_w + cols * (cell_w + margin) + margin
    total_h = rows * (cell_h + margin + label_h) + margin
    
    # Start building SVG
    svg_lines = [
        f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{total_w}" height="{total_h}" fill="white"/>'
    ]
    
    # Place Arch and Label in rows (not columns)
    ref_x_start = left_label_w + margin
    row1_y = margin + label_h
    row2_y = row1_y + cell_h + margin + label_h
    
    # Row 1: Arch (if exists) + First directory samples
    samples_x_start = ref_x_start + (cell_w + margin)
    svg_lines.append(f'<!-- Row 1: {label1} -->')
    
    # Arch in row 1 (if exists)
    if arch_content:
        svg_lines.append('<!-- Arch in row 1 -->')
        svg_lines.append(f'<g transform="translate({ref_x_start}, {row1_y})">')
        svg_lines.append('    <text x="60.0" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Arch</text>')
        svg_lines.append(f'    <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
        arch_lines = arch_content.strip().split('\n')
        for line in arch_lines:
            svg_lines.append(f'    {line}')
        svg_lines.append('</g>')
        svg_lines.append('')
    
    # First directory samples in row 1
    svg_lines.append(f'<g transform="translate({samples_x_start}, {row1_y})">')
    
    for i in range(num_samples):
        x_offset = i * (cell_w + margin)
        svg_lines.append(f'    <!-- Sample {i} -->')
        svg_lines.append(f'    <g transform="translate({x_offset}, 0)">')
        svg_lines.append(f'        <text x="{cell_w/2}" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Sample {i}</text>')
        svg_lines.append(f'        <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
        # Add the SVG content (skip the outer rect if present)
        content_lines = samples1[i].strip().split('\n')
        for line in content_lines:
            # Skip the outer rect and svg tags if present
            if '<rect width="120" height="120" fill="white"/>' in line or '<svg' in line or '</svg>' in line:
                continue
            svg_lines.append(f'        {line}')
        svg_lines.append('    </g>')
        svg_lines.append('')
    
    svg_lines.append('</g>')
    svg_lines.append('')
    
    # Row 2: Label + Second directory samples
    svg_lines.append(f'<!-- Row 2: {label2} -->')
    
    # Label in row 2
    svg_lines.append('<!-- Label in row 2 -->')
    svg_lines.append(f'<g transform="translate({ref_x_start}, {row2_y})">')
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
    
    # Second directory samples in row 2
    svg_lines.append(f'<g transform="translate({samples_x_start}, {row2_y})">')
    
    for i in range(num_samples):
        x_offset = i * (cell_w + margin)
        svg_lines.append(f'    <!-- Sample {i} -->')
        svg_lines.append(f'    <g transform="translate({x_offset}, 0)">')
        svg_lines.append(f'        <text x="{cell_w/2}" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Sample {i}</text>')
        svg_lines.append(f'        <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
        # Add the SVG content (skip the outer rect if present)
        content_lines = samples2[i].strip().split('\n')
        for line in content_lines:
            # Skip the outer rect and svg tags if present
            if '<rect width="120" height="120" fill="white"/>' in line or '<svg' in line or '</svg>' in line:
                continue
            svg_lines.append(f'        {line}')
        svg_lines.append('    </g>')
        svg_lines.append('')
    
    svg_lines.append('</g>')
    svg_lines.append('')
    
    # Labels on the left
    svg_lines.append('<!-- Labels on the left -->')
    svg_lines.append('<g transform="translate(0, 0)">')
    row1_center_y = row1_y + cell_h / 2
    row2_center_y = row2_y + cell_h / 2
    svg_lines.append(f'    <text x="{left_label_w/2}" y="{row1_center_y}" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" transform="rotate(-90, {left_label_w/2}, {row1_center_y})">{label1}</text>')
    svg_lines.append(f'    <text x="{left_label_w/2}" y="{row2_center_y}" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" transform="rotate(-90, {left_label_w/2}, {row2_center_y})">{label2}</text>')
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
    parser = argparse.ArgumentParser(description="Create comparison grids between different epoch results")
    parser.add_argument('--dir1', required=True, help="First directory (epoch 40k batch inference)")
    parser.add_argument('--dir2', required=True, help="Second directory (epoch 60k)")
    parser.add_argument('--output_dir', required=True, help="Output directory for comparison grids")
    parser.add_argument('--label1', default="epoch 40k", help="Label for first directory")
    parser.add_argument('--label2', default="epoch 60k", help="Label for second directory")
    parser.add_argument('--house_id', help="Specific house_id to process (if not provided, processes all)")
    
    args = parser.parse_args()
    
    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    output_dir = Path(args.output_dir)
    
    if not dir1.exists():
        print(f"Error: First directory does not exist: {dir1}")
        return
    
    if not dir2.exists():
        print(f"Error: Second directory does not exist: {dir2}")
        return
    
    # Get list of house_ids
    if args.house_id:
        house_ids = [args.house_id]
    else:
        # Get all house_ids from both directories
        house_ids1 = {d.name for d in dir1.iterdir() if d.is_dir()}
        house_ids2 = {d.name for d in dir2.iterdir() if d.is_dir()}
        house_ids = sorted(house_ids1 & house_ids2)
    
    print(f"Processing {len(house_ids)} house_ids...")
    
    for house_id in house_ids:
        create_comparison_grid(dir1, dir2, output_dir, house_id, args.label1, args.label2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

