#!/usr/bin/env python
"""
Create a self-contained HTML viewer for comparison SVGs.

Usage:
    python scripts/create_comparison_viewer_html.py --input_dir output/model_eval_2/comparisons --output output/model_eval_2/comparisons/viewer.html
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm


def escape_html(text):
    """Escape HTML special characters."""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))


def create_viewer_html(comparisons_dir, output_path):
    """Create a self-contained HTML viewer for all comparison SVGs."""
    
    comparisons_dir = Path(comparisons_dir)
    # Search for comparison_grid.svg files recursively
    svg_files = sorted(comparisons_dir.glob('**/comparison_grid.svg'))
    
    # Also check for *_compare.svg files in the root directory (for backward compatibility)
    if not svg_files:
        svg_files = sorted(comparisons_dir.glob('*_compare.svg'))
    
    if not svg_files:
        print(f"No comparison SVG files found in {comparisons_dir}")
        return
    
    print(f"Found {len(svg_files)} comparison SVG files")
    
    # Read all SVG files and extract their content
    svg_data = []
    for svg_file in tqdm(svg_files, desc="Reading SVGs"):
        # Extract scene ID from parent directory name
        if svg_file.name == 'comparison_grid.svg':
            # Get the parent directory name (scene ID)
            scene_id = svg_file.parent.name
        else:
            # For backward compatibility with *_compare.svg files
            scene_id = svg_file.stem.replace('_compare', '')
        
        with open(svg_file, 'r') as f:
            svg_content = f.read()
        
        # Extract inner SVG content (remove outer <svg> tags)
        match = re.search(r'<svg[^>]*>(.*)</svg>', svg_content, re.DOTALL)
        if match:
            inner_content = match.group(1)
            svg_data.append({
                'scene_id': scene_id,
                'content': inner_content,
                'full_svg': svg_content
            })
    
    # Generate HTML
    html_parts = []
    
    # HTML header and styles
    html_parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Eval Comparison Viewer</title>
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
        
        .export-btn {
            margin-top: 10px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        
        .export-btn:hover {
            background-color: #45a049;
        }
        
        .export-btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        
        .selection-info {
            margin-top: 10px;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .grid-container {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        
        .svg-card {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }
        
        .svg-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .svg-card.selected {
            border: 3px solid #4CAF50;
            background-color: #f0fff0;
        }
        
        .svg-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            font-size: 14px;
            word-break: break-all;
        }
        
        .svg-container {
            width: 100%;
            overflow: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #fff;
        }
        
        .svg-container svg {
            display: block;
            max-width: 100%;
            height: auto;
        }
        
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Eval Comparison Viewer</h1>
        <input type="text" id="searchBox" class="search-box" placeholder="Search by scene ID...">
        <div class="stats">
            Total: <span id="totalCount">''' + str(len(svg_data)) + '''</span> | 
            Showing: <span id="showingCount">''' + str(len(svg_data)) + '''</span>
        </div>
        <div class="selection-info">
            Selected: <span id="selectedCount">0</span> scenes
        </div>
        <button class="export-btn" id="exportBtn">Export Selected IDs</button>
    </div>
    
    <div class="grid-container" id="gridContainer">''')
    
    # Add SVG cards
    for item in svg_data:
        scene_id = item['scene_id']
        inner_content = item['content']
        
        # Extract width and height from full SVG if available
        full_svg = item['full_svg']
        width_match = re.search(r'width="(\d+)"', full_svg)
        height_match = re.search(r'height="(\d+)"', full_svg)
        width = width_match.group(1) if width_match else '1620'
        height = height_match.group(1) if height_match else '620'
        
        html_parts.append(f'''
        <div class="svg-card" data-name="{escape_html(scene_id)}">
            <div class="svg-title">{escape_html(scene_id)}</div>
            <div class="svg-container">
                <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
{inner_content}
                </svg>
            </div>
        </div>''')
    
    # Close grid container and add JavaScript
    html_parts.append('''    </div>
    
    <script>
        const searchBox = document.getElementById('searchBox');
        const gridContainer = document.getElementById('gridContainer');
        const totalCountSpan = document.getElementById('totalCount');
        const showingCountSpan = document.getElementById('showingCount');
        const selectedCountSpan = document.getElementById('selectedCount');
        const exportBtn = document.getElementById('exportBtn');
        
        const svgCards = Array.from(document.querySelectorAll('.svg-card'));
        const totalCount = svgCards.length;
        let selectedIds = new Set();
        
        // Update showing count
        function updateShowingCount() {
            const visible = svgCards.filter(card => !card.classList.contains('hidden')).length;
            showingCountSpan.textContent = visible;
        }
        
        // Update selected count
        function updateSelectedCount() {
            selectedCountSpan.textContent = selectedIds.size;
            exportBtn.disabled = selectedIds.size === 0;
        }
        
        // Search functionality
        searchBox.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            
            svgCards.forEach(card => {
                const name = card.getAttribute('data-name').toLowerCase();
                if (query === '' || name.includes(query)) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });
            
            updateShowingCount();
        });
        
        // Selection functionality
        svgCards.forEach(card => {
            card.addEventListener('click', () => {
                const sceneId = card.getAttribute('data-name');
                
                if (selectedIds.has(sceneId)) {
                    selectedIds.delete(sceneId);
                    card.classList.remove('selected');
                } else {
                    selectedIds.add(sceneId);
                    card.classList.add('selected');
                }
                
                updateSelectedCount();
            });
        });
        
        // Export functionality
        exportBtn.addEventListener('click', () => {
            if (selectedIds.size === 0) {
                alert('No scenes selected');
                return;
            }
            
            const selectedArray = Array.from(selectedIds).sort();
            const content = selectedArray.join('\\n');
            
            const blob = new Blob([content], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'selected_scene_ids.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            alert(`Exported ${selectedIds.size} scene IDs to selected_scene_ids.txt`);
        });
        
        // Initialize
        updateShowingCount();
        updateSelectedCount();
    </script>
</body>
</html>''')
    
    # Write HTML file
    html_content = '\n'.join(html_parts)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nCreated self-contained HTML viewer: {output_path}")
    print(f"  Total scenes: {len(svg_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create self-contained HTML viewer for comparison SVGs")
    parser.add_argument('--input_dir', required=True, help="Directory containing comparison SVG files")
    parser.add_argument('--output', default=None, help="Output HTML file path (default: input_dir/viewer.html)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = args.output if args.output else input_dir / "viewer.html"
    
    create_viewer_html(input_dir, output_path)

