#!/usr/bin/env python3
"""
Script to add a third row to existing comparison SVG files.
Adds samples from fixed_iou_10x directory with label "fixed_iou_10x".
"""
import argparse
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


def add_third_row_to_comparison(comparison_svg_path, new_samples_dir, house_id):
    """Add a third row to an existing comparison SVG."""
    
    # Read the existing comparison SVG
    with open(comparison_svg_path, 'r') as f:
        svg_content = f.read()
    
    # Check if Row 3 already exists
    if '<!-- Row 3: fixed_iou_10x -->' in svg_content:
        print(f"Skipping {comparison_svg_path.name}: Row 3 already exists")
        return False
    
    # Extract the house ID from the directory name
    house_dir = new_samples_dir / house_id
    if not house_dir.exists():
        print(f"Warning: Missing directory {house_dir}")
        return False
    
    # Load samples 0-5 from the new directory
    new_samples = []
    for i in range(6):
        sample_svg_path = house_dir / f"boxes_{i}.svg"
        sample_content = extract_svg_content(sample_svg_path)
        if sample_content:
            new_samples.append(sample_content)
        else:
            print(f"Warning: Missing sample {i} for {house_id}")
            return False
    
    # Extract current SVG dimensions
    width_match = re.search(r'width="(\d+)"', svg_content)
    height_match = re.search(r'height="(\d+)"', svg_content)
    
    if not width_match or not height_match:
        print(f"Warning: Could not extract SVG dimensions from {comparison_svg_path}")
        return False
    
    current_width = int(width_match.group(1))
    current_height = int(height_match.group(1))
    
    # Calculate new height: add one more row (160 pixels: 120 cell + 20 margin + 20 label)
    row_height = 160
    new_height = current_height + row_height
    
    # Update SVG dimensions
    svg_content = re.sub(r'height="\d+"', f'height="{new_height}"', svg_content)
    
    # Find where to insert the third row (before the labels section)
    # Find the position before the closing </svg>
    svg_end_pos = svg_content.rfind('</svg>')
    if svg_end_pos == -1:
        print(f"Warning: Could not find </svg> tag in {comparison_svg_path}")
        return False
    
    # Row 3 starts at y = 360 (Row 2 starts at 200, so Row 3 = 200 + 160)
    row3_y = 360
    cell_w, cell_h = 120, 120
    margin = 20
    
    # Build the third row content
    row3_lines = []
    row3_lines.append('')
    row3_lines.append('<!-- Row 3: fixed_iou_10x -->')
    row3_lines.append('<!-- Arch in row 3 -->')
    row3_lines.append('<g transform="translate(100, 360)">')
    
    # Extract arch content from Row 1 (it's the same for all rows)
    arch_match = re.search(r'<!-- Arch in row 1 -->\s*<g transform="translate\(100, 40\)">(.*?)</g>', svg_content, re.DOTALL)
    if arch_match:
        arch_content = arch_match.group(1)
        # Add arch content, but fix rect height if needed
        arch_lines = arch_content.strip().split('\n')
        for line in arch_lines:
            # Fix rect height to 120 if it's wrong
            if '<rect' in line and 'height=' in line:
                line = re.sub(r'height="\d+"', 'height="120"', line)
            row3_lines.append(f'    {line}')
    else:
        # Fallback: use a simple arch placeholder
        row3_lines.append('    <text x="60.0" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Arch</text>')
        row3_lines.append('    <rect width="120" height="120" fill="white"/>')
    
    row3_lines.append('</g>')
    row3_lines.append('')
    
    # Add samples
    row3_lines.append('<g transform="translate(240, 360)">')
    for i in range(6):
        x_offset = i * (cell_w + margin)
        row3_lines.append(f'    <!-- Sample {i} -->')
        row3_lines.append(f'    <g transform="translate({x_offset}, 0)">')
        row3_lines.append(f'        <text x="60.0" y="-5" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle">Sample {i}</text>')
        row3_lines.append(f'        <rect width="120" height="120" fill="white"/>')
        
        # Add the sample SVG content (skip the outer rect and svg tags if present)
        content_lines = new_samples[i].strip().split('\n')
        for line in content_lines:
            # Skip the outer rect and svg tags if present
            if '<rect width="120" height="120" fill="white"/>' in line or '<svg' in line or '</svg>' in line:
                continue
            row3_lines.append(f'        {line}')
        row3_lines.append('    </g>')
        row3_lines.append('')
    
    row3_lines.append('</g>')
    row3_lines.append('')
    
    # Insert the third row before the labels section
    # Find the position right before "<!-- Labels on the left -->"
    labels_start_pattern = r'<!-- Labels on the left -->'
    labels_start_match = re.search(labels_start_pattern, svg_content)
    
    if labels_start_match:
        insertion_point = labels_start_match.start()
    else:
        # Fallback: insert before </svg>
        insertion_point = svg_end_pos
    
    # Insert the row content
    row3_content = '\n'.join(row3_lines)
    svg_content = svg_content[:insertion_point] + row3_content + '\n' + svg_content[insertion_point:]
    
    # Update the labels section to include the third row label
    # Find the labels section and add the third row label
    labels_pattern = r'(<!-- Labels on the left -->.*?</g>)'
    labels_match = re.search(labels_pattern, svg_content, re.DOTALL)
    
    if labels_match:
        labels_section = labels_match.group(1)
        # Add the third row label before the closing </g>
        # Row 3 center y = 360 + 60 = 420
        row3_label = f'    <text x="40.0" y="420.0" font-family="Arial" font-size="12" font-weight="bold" text-anchor="middle" transform="rotate(-90, 40.0, 420.0)">fixed_iou_10x</text>\n'
        labels_section = labels_section.replace('</g>', row3_label + '</g>')
        svg_content = svg_content.replace(labels_match.group(1), labels_section)
    
    # Write the updated SVG
    with open(comparison_svg_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Updated comparison SVG: {comparison_svg_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Add third row to existing comparison SVG files")
    parser.add_argument('--comparison_dir', required=True, help="Directory containing comparison SVG files")
    parser.add_argument('--new_samples_dir', required=True, help="Directory containing new samples (fixed_iou_10x)")
    
    args = parser.parse_args()
    
    comparison_dir = Path(args.comparison_dir)
    new_samples_dir = Path(args.new_samples_dir)
    
    if not comparison_dir.exists():
        print(f"Error: Comparison directory does not exist: {comparison_dir}")
        return
    
    if not new_samples_dir.exists():
        print(f"Error: New samples directory does not exist: {new_samples_dir}")
        return
    
    # Find all comparison SVG files
    comparison_files = list(comparison_dir.glob("*_compare.svg"))
    
    print(f"Found {len(comparison_files)} comparison files")
    
    for comparison_file in comparison_files:
        # Extract house ID from filename (e.g., "213444965_compare.svg" -> "213444965")
        house_id_short = comparison_file.stem.replace('_compare', '')
        
        # Find the matching house directory in new_samples_dir
        # The directory name format is like "213444965_Bath_US_simple_design_filtered"
        matching_dirs = [d for d in new_samples_dir.iterdir() if d.is_dir() and d.name.startswith(house_id_short)]
        
        if not matching_dirs:
            print(f"Warning: No matching directory found for {house_id_short}")
            continue
        
        house_id = matching_dirs[0].name
        
        print(f"Processing {house_id_short} -> {house_id}")
        add_third_row_to_comparison(comparison_file, new_samples_dir, house_id)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

