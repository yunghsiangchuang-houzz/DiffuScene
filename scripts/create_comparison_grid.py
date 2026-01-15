#!/usr/bin/env python3
"""
Script to create comparison grids between single and batch inference results.
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


def create_comparison_grid(single_dir, batch_dir, output_dir, house_id):
    """Create a comparison grid SVG for a specific house_id."""
    single_house_dir = single_dir / house_id
    batch_house_dir = batch_dir / house_id
    
    if not single_house_dir.exists() or not batch_house_dir.exists():
        print(f"Warning: Missing directories for {house_id}")
        return False
    
    # Load label SVG
    label_svg_path = single_house_dir / "simple_design_label.svg"
    if not label_svg_path.exists():
        label_svg_path = batch_house_dir / "simple_design_label.svg"
    
    label_content = extract_svg_content(label_svg_path)
    if not label_content:
        print(f"Warning: Missing label SVG for {house_id}")
        return False
    
    # Collect SVG contents for samples 0-5
    single_samples = []
    batch_samples = []
    
    for i in range(6):
        single_svg = single_house_dir / f"boxes_{i}.svg"
        batch_svg = batch_house_dir / f"boxes_{i}.svg"
        
        single_content = extract_svg_content(single_svg)
        batch_content = extract_svg_content(batch_svg)
        
        if single_content:
            single_samples.append(single_content)
        if batch_content:
            batch_samples.append(batch_content)
    
    if not single_samples or not batch_samples:
        print(f"Warning: Missing samples for {house_id}")
        return False
    
    # Setup grid dimensions
    num_samples = min(len(single_samples), len(batch_samples), 6)
    cell_w, cell_h = 120, 120
    margin = 20
    label_h = 20
    left_label_w = 80
    
    cols = num_samples + 1  # +1 for label column
    rows = 2  # Single inference row and batch inference row
    
    total_w = left_label_w + cols * (cell_w + margin) + margin
    total_h = rows * (cell_h + margin + label_h) + margin
    
    # Start building SVG
    svg_lines = [
        f'<svg width="{total_w}" height="{total_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{total_w}" height="{total_h}" fill="white"/>'
    ]
    
    # Row 1: Single inference
    row1_y = margin + label_h
    svg_lines.append(f'<!-- Row 1: single inference -->')
    svg_lines.append(f'<g transform="translate({left_label_w + margin}, {row1_y})">')
    
    # Add label SVG as first column
    svg_lines.append('    <!-- Label -->')
    svg_lines.append('    <g transform="translate(0, 0)">')
    svg_lines.append('        <text x="60.0" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Label</text>')
    svg_lines.append(f'        <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
    # Add the label SVG content (skip the outer rect and svg tags if present)
    label_lines = label_content.strip().split('\n')
    for line in label_lines:
        # Skip the outer rect and svg tags if present
        if ('<rect' in line and 'width="120"' in line and 'height="120"' in line) or '<svg' in line or '</svg>' in line:
            continue
        svg_lines.append(f'        {line}')
    svg_lines.append('    </g>')
    svg_lines.append('')
    
    # Add samples
    for i in range(num_samples):
        x_offset = (i + 1) * (cell_w + margin)  # +1 to account for label column
        svg_lines.append(f'    <!-- Sample {i} -->')
        svg_lines.append(f'    <g transform="translate({x_offset}, 0)">')
        svg_lines.append(f'        <text x="{cell_w/2}" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Sample {i}</text>')
        svg_lines.append(f'        <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
        # Add the SVG content (skip the outer rect if present)
        content_lines = single_samples[i].strip().split('\n')
        for line in content_lines:
            # Skip the outer rect and svg tags if present
            if '<rect width="120" height="120" fill="white"/>' in line or '<svg' in line or '</svg>' in line:
                continue
            svg_lines.append(f'        {line}')
        svg_lines.append('    </g>')
        svg_lines.append('')
    
    svg_lines.append('</g>')
    svg_lines.append('')
    
    # Row 2: Batch inference
    row2_y = row1_y + cell_h + margin + label_h
    svg_lines.append(f'<!-- Row 2: batch inference -->')
    svg_lines.append(f'<g transform="translate({left_label_w + margin}, {row2_y})">')
    
    # Add label SVG as first column
    svg_lines.append('    <!-- Label -->')
    svg_lines.append('    <g transform="translate(0, 0)">')
    svg_lines.append('        <text x="60.0" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Label</text>')
    svg_lines.append(f'        <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
    # Add the label SVG content (skip the outer rect and svg tags if present)
    label_lines = label_content.strip().split('\n')
    for line in label_lines:
        # Skip the outer rect and svg tags if present
        if ('<rect' in line and 'width="120"' in line and 'height="120"' in line) or '<svg' in line or '</svg>' in line:
            continue
        svg_lines.append(f'        {line}')
    svg_lines.append('    </g>')
    svg_lines.append('')
    
    # Add samples
    for i in range(num_samples):
        x_offset = (i + 1) * (cell_w + margin)  # +1 to account for label column
        svg_lines.append(f'    <!-- Sample {i} -->')
        svg_lines.append(f'    <g transform="translate({x_offset}, 0)">')
        svg_lines.append(f'        <text x="{cell_w/2}" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Sample {i}</text>')
        svg_lines.append(f'        <rect width="{cell_w}" height="{cell_h}" fill="white"/>')
        # Add the SVG content (skip the outer rect if present)
        content_lines = batch_samples[i].strip().split('\n')
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
    svg_lines.append(f'    <text x="{left_label_w/2}" y="{row1_center_y}" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" transform="rotate(-90, {left_label_w/2}, {row1_center_y})">single inference</text>')
    svg_lines.append(f'    <text x="{left_label_w/2}" y="{row2_center_y}" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" transform="rotate(-90, {left_label_w/2}, {row2_center_y})">batch inference</text>')
    svg_lines.append('</g>')
    svg_lines.append('')
    
    svg_lines.append('</svg>')
    
    # Write to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_grid.svg"
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))
    
    print(f"Created comparison grid: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create comparison grids between single and batch inference")
    parser.add_argument('--single_dir', required=True, help="Directory containing single inference results")
    parser.add_argument('--batch_dir', required=True, help="Directory containing batch inference results")
    parser.add_argument('--output_dir', required=True, help="Output directory for comparison grids")
    parser.add_argument('--house_id', help="Specific house_id to process (if not provided, processes all)")
    
    args = parser.parse_args()
    
    single_dir = Path(args.single_dir)
    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir)
    
    if not single_dir.exists():
        print(f"Error: Single inference directory does not exist: {single_dir}")
        return
    
    if not batch_dir.exists():
        print(f"Error: Batch inference directory does not exist: {batch_dir}")
        return
    
    # Get list of house_ids
    if args.house_id:
        house_ids = [args.house_id]
    else:
        # Get all house_ids from single_dir
        single_house_ids = {d.name for d in single_dir.iterdir() if d.is_dir()}
        batch_house_ids = {d.name for d in batch_dir.iterdir() if d.is_dir()}
        house_ids = sorted(single_house_ids & batch_house_ids)
    
    print(f"Processing {len(house_ids)} house_ids...")
    
    for house_id in house_ids:
        create_comparison_grid(single_dir, batch_dir, output_dir / house_id, house_id)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

