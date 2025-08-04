#!/usr/bin/env python3
"""
Process all paths.txt files in experiment folders to create heatmaps.

Usage:
    python3 process_all_heatmaps.py

This script will:
1. Find all exp/run_agents* folders
2. Process paths.txt in each folder
3. Create a heatmap showing tile usage probability
4. Save heatmaps as numpy arrays in exp_heatmaps/
"""

import os
import re
import numpy as np
from pathlib import Path
from create_heatmap import parse_paths_file, create_heatmap

def extract_grid_info(folder_name):
    """Extract grid number and endpoints from folder name."""
    grid_match = re.search(r'grid(\d+)', folder_name)
    endpoints_match = re.search(r'endpoints(\d+)', folder_name)
    
    if not grid_match or not endpoints_match:
        return None, None
    
    grid_num = int(grid_match.group(1))
    endpoints = int(endpoints_match.group(1))
    return grid_num, endpoints

def process_experiment_folder(folder_path, output_dir):
    """Process a single experiment folder and generate its heatmap."""
    # Extract grid info from folder name
    grid_num, endpoints = extract_grid_info(folder_path.name)
    if grid_num is None:
        print(f"Skipping {folder_path.name}: couldn't extract grid info")
        return None
    
    # Check for paths.txt
    paths_file = folder_path / "paths.txt"
    if not paths_file.exists():
        print(f"Skipping grid_{grid_num:05d}: paths.txt not found")
        return None
    
    try:
        # Parse paths and create heatmap
        tile_usage, total_steps = parse_paths_file(paths_file)
        heatmap = create_heatmap(tile_usage, total_steps)
        
        # Save the heatmap array
        output_file = output_dir / f"grid_{grid_num:05d}_heatmap.npy"
        np.save(output_file, heatmap)
        
        print(f"Processed grid_{grid_num:05d}")
        print(f"  - Endpoints: {endpoints}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Unique tiles: {len(tile_usage)}")
        print(f"  - Array sum: {np.sum(heatmap):.6f}")
        print(f"  - Saved to: {output_file}")
        print("-" * 50)
        
        return {
            'grid_num': grid_num,
            'endpoints': endpoints,
            'total_steps': total_steps,
            'unique_tiles': len(tile_usage),
            'array_sum': float(np.sum(heatmap))
        }
        
    except Exception as e:
        print(f"Error processing grid_{grid_num:05d}: {str(e)}")
        return None

def main():
    # Create output directory
    output_dir = Path("exp_heatmaps")
    output_dir.mkdir(exist_ok=True)
    print(f"Saving heatmaps to {output_dir}")
    print("-" * 50)
    
    # Process all experiment folders
    exp_dir = Path("exp")
    if not exp_dir.exists():
        print("Error: exp directory not found")
        return
    
    # Keep track of statistics
    total_folders = 0
    processed_folders = 0
    skipped_folders = 0
    error_folders = 0
    results = []
    
    # Process each folder
    for folder in sorted(exp_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("run_agents"):
            continue
            
        total_folders += 1
        result = process_experiment_folder(folder, output_dir)
        
        if result is None:
            if not (folder / "paths.txt").exists():
                skipped_folders += 1
            else:
                error_folders += 1
        else:
            processed_folders += 1
            results.append(result)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total folders found: {total_folders}")
    print(f"Successfully processed: {processed_folders}")
    print(f"Skipped (no paths.txt): {skipped_folders}")
    print(f"Failed with errors: {error_folders}")
    
    if processed_folders > 0:
        # Calculate statistics across all processed grids
        total_steps = [r['total_steps'] for r in results]
        unique_tiles = [r['unique_tiles'] for r in results]
        endpoints = [r['endpoints'] for r in results]
        
        print("\nGrid Statistics:")
        print(f"Total steps - min: {min(total_steps)}, max: {max(total_steps)}, mean: {np.mean(total_steps):.1f}")
        print(f"Unique tiles - min: {min(unique_tiles)}, max: {max(unique_tiles)}, mean: {np.mean(unique_tiles):.1f}")
        print(f"Endpoints - min: {min(endpoints)}, max: {max(endpoints)}, mean: {np.mean(endpoints):.1f}")
        
        print(f"\nHeatmap arrays saved in {output_dir}")
        print("To load a heatmap in Python:")
        print("import numpy as np")
        print("heatmap = np.load('exp_heatmaps/grid_XXXXX_heatmap.npy')")

if __name__ == "__main__":
    main() 