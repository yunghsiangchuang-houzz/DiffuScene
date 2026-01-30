#!/usr/bin/env python3
"""
Create a self-contained HTML viewer for comparing box SVGs (original vs postprocessed).

Usage:
    python scripts/create_box_comparison_viewer.py --input_dir output/houzz_bathroom_v1.2_no_aug/generated_results_model_best_batch64 --output output/houzz_bathroom_v1.2_no_aug/generated_results_model_best_batch64/viewer.html
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm
import json


def escape_html(text):
    """Escape HTML special characters."""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


def escape_js_string(text):
    """Escape JavaScript string."""
    return (text.replace('\\', '\\\\')
                .replace("'", "\\'")
                .replace('"', '\\"')
                .replace('\n', '\\n')
                .replace('\r', '\\r'))


def extract_box_number(filename):
    """Extract box number from filename (e.g., 'boxes_42.svg' -> 42)."""
    match = re.search(r'boxes_(\d+)', filename)
    return int(match.group(1)) if match else None


def create_viewer_html(input_dir, output_path):
    """Create a self-contained HTML viewer for box comparison pairs."""
    
    input_dir = Path(input_dir)
    
    # Find all house directories (exclude top_k subdirectories)
    house_dirs = sorted([d for d in input_dir.iterdir() 
                        if d.is_dir() and d.name.endswith('_Bath_US_simple_design_filtered')])
    
    if not house_dirs:
        print(f"No house directories found in {input_dir}")
        return
    
    print(f"Found {len(house_dirs)} house directories")
    
    # Collect all comparison pairs
    all_data = []
    
    for house_dir in tqdm(house_dirs, desc="Processing houses"):
        house_id = house_dir.name.replace('_Bath_US_simple_design_filtered', '')
        
        # Find all boxes_X.svg files (excluding top_k subdirectory)
        original_files = sorted(house_dir.glob('boxes_*.svg'), 
                               key=lambda f: extract_box_number(f.name) or 999)
        
        pairs = []
        for orig_file in original_files:
            # Skip if it's a postprocessed file
            if 'postprocess' in orig_file.name:
                continue
            
            box_num = extract_box_number(orig_file.name)
            if box_num is None:
                continue
            
            # Find corresponding postprocessed file
            post_file = house_dir / f'boxes_{box_num}.npz.postprocess.npz.svg'
            
            if not post_file.exists():
                print(f"Warning: Postprocessed file not found for {orig_file.name}")
                continue
            
            # Read SVG contents
            try:
                with open(orig_file, 'r', encoding='utf-8') as f:
                    orig_svg = f.read()
                
                with open(post_file, 'r', encoding='utf-8') as f:
                    post_svg = f.read()
                
                # Escape script tags in SVG
                orig_svg = orig_svg.replace('</script>', '</scr\' + \'ipt>')
                post_svg = post_svg.replace('</script>', '</scr\' + \'ipt>')
                
                pairs.append({
                    'box_num': box_num,
                    'original_svg': orig_svg,
                    'postprocessed_svg': post_svg
                })
            except Exception as e:
                print(f"Error reading files for {house_id} boxes_{box_num}: {e}")
                continue
        
        if pairs:
            all_data.append({
                'house_id': house_id,
                'pairs': sorted(pairs, key=lambda x: x['box_num'])
            })
    
    print(f"\nTotal houses: {len(all_data)}")
    print(f"Total pairs: {sum(len(d['pairs']) for d in all_data)}")
    
    # Generate HTML
    html_parts = []
    
    # HTML header and styles
    html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Box Comparison Viewer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .header {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .search-box {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
        }
        
        .search-box:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .stats {
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }
        
        .house-section {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        
        .house-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ddd;
        }
        
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .comparison-pair {
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .comparison-pair:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .pair-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 10px;
            text-align: center;
            font-size: 14px;
        }
        
        .svg-wrapper {
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #fff;
            overflow: auto;
        }
        
        .svg-wrapper:last-child {
            margin-bottom: 0;
        }
        
        .svg-label {
            font-size: 12px;
            color: #666;
            padding: 5px 10px;
            background-color: #f0f0f0;
            border-bottom: 1px solid #ccc;
        }
        
        .svg-label.original {
            background-color: #e3f2fd;
        }
        
        .svg-label.postprocessed {
            background-color: #fff3e0;
        }
        
        .svg-container {
            padding: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .svg-container svg {
            display: block;
            max-width: 100%;
            height: auto;
        }
        
        .hidden {
            display: none;
        }
        
        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .pagination button {
            padding: 8px 16px;
            font-size: 14px;
            border: 1px solid #ddd;
            background-color: #fff;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .pagination button:hover:not(:disabled) {
            background-color: #f0f0f0;
        }
        
        .pagination button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .pagination .page-info {
            padding: 8px 16px;
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Box Comparison Viewer</h1>
        <input type="text" id="searchBox" class="search-box" placeholder="Search by house ID...">
        <div class="stats">
            Total Houses: <span id="totalCount">0</span> | 
            Showing: <span id="showingCount">0</span>
        </div>
    </div>
    
    <div id="contentContainer">''')
    
    # Add house sections with comparison pairs
    for house_data in all_data:
        house_id = house_data['house_id']
        pairs = house_data['pairs']
        
        html_parts.append(f'''
    <div class="house-section" data-house-id="{escape_html(house_id)}">
        <div class="house-title">{escape_html(house_id)} ({len(pairs)} pairs)</div>
        <div class="comparison-grid">''')
        
        for pair in pairs:
            box_num = pair['box_num']
            orig_svg = pair['original_svg']
            post_svg = pair['postprocessed_svg']
            
            html_parts.append(f'''
            <div class="comparison-pair">
                <div class="pair-label">boxes_{box_num}</div>
                <div class="svg-wrapper">
                    <div class="svg-label original">Original</div>
                    <div class="svg-container">
                        {orig_svg}
                    </div>
                </div>
                <div class="svg-wrapper">
                    <div class="svg-label postprocessed">Postprocessed</div>
                    <div class="svg-container">
                        {post_svg}
                    </div>
                </div>
            </div>''')
        
        html_parts.append('''
        </div>
    </div>''')
    
    # Close content container and add JavaScript
    html_parts.append('''    </div>
    
    <script>
        const searchBox = document.getElementById('searchBox');
        const contentContainer = document.getElementById('contentContainer');
        const totalCountSpan = document.getElementById('totalCount');
        const showingCountSpan = document.getElementById('showingCount');
        
        const houseSections = Array.from(document.querySelectorAll('.house-section'));
        const totalCount = houseSections.length;
        
        totalCountSpan.textContent = totalCount;
        
        function updateShowingCount() {
            const visible = houseSections.filter(section => !section.classList.contains('hidden')).length;
            showingCountSpan.textContent = visible;
        }
        
        // Search functionality
        searchBox.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            
            houseSections.forEach(section => {
                const houseId = section.getAttribute('data-house-id').toLowerCase();
                if (query === '' || houseId.includes(query)) {
                    section.classList.remove('hidden');
                } else {
                    section.classList.add('hidden');
                }
            });
            
            updateShowingCount();
        });
        
        // Initialize
        updateShowingCount();
    </script>
</body>
</html>''')
    
    # Write HTML file
    html_content = '\n'.join(html_parts)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nCreated self-contained HTML viewer: {output_path}")
    print(f"  Total houses: {len(all_data)}")
    print(f"  Total comparison pairs: {sum(len(d['pairs']) for d in all_data)}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create self-contained HTML viewer for box comparison pairs")
    parser.add_argument('--input_dir', required=True, help="Directory containing house directories with box SVGs")
    parser.add_argument('--output', default=None, help="Output HTML file path (default: input_dir/viewer.html)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = args.output if args.output else input_dir / "viewer.html"
    
    create_viewer_html(input_dir, output_path)



