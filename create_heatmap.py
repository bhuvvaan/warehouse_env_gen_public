#!/usr/bin/env python3
"""
Usage:
    python3 create_heatmap.py exp/run_agents200_endpoints369_grid00590/paths.txt
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Fixed grid dimensions
ROWS = 33
COLS = 36

def parse_paths_file(file_path):
    """Parse the paths.txt file and count tile usage."""
    tile_usage = defaultdict(int)
    total_steps = 0
    skipped_lines = 0
    
    try:
        # First read number of agents (first line should be valid ASCII)
        with open(file_path, 'r') as f:
            num_agents = int(f.readline().strip())
            print(f"Processing paths for {num_agents} agents...")
    except UnicodeDecodeError:
        # If even first line is corrupted, try binary mode
        with open(file_path, 'rb') as f:
            num_agents = int(f.readline().decode('ascii', errors='ignore').strip())
            print(f"Processing paths for {num_agents} agents (read in binary mode)...")
    
    # Now process the paths, handling potential corruption
    with open(file_path, 'rb') as f:
        # Skip the first line since we already read it
        f.readline()
        
        # Process each line
        for line_num, line_bytes in enumerate(f, 1):
            try:
                # Try to decode the line, ignoring invalid bytes
                line = line_bytes.decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                
                # Split into semicolon-separated entries
                entries = line.strip().split(';')
                
                # Process each entry (tile,_,time)
                valid_entries = 0
                for entry in entries:
                    if not entry:
                        continue
                    try:
                        # First number in each entry is the tile number
                        tile = int(entry.split(',')[0])
                        tile_usage[tile] += 1
                        total_steps += 1
                        valid_entries += 1
                    except (ValueError, IndexError):
                        continue
                
                # If we found no valid entries in this line, count it as skipped
                if valid_entries == 0:
                    skipped_lines += 1
                    
            except Exception as e:
                skipped_lines += 1
                continue
    
    if total_steps == 0:
        raise ValueError("No valid paths found in the file")
    
    print(f"Processed {len(tile_usage)} unique tiles across {total_steps} total steps")
    if skipped_lines > 0:
        print(f"Note: Skipped {skipped_lines} corrupted or invalid lines")
    
    return tile_usage, total_steps

def create_heatmap(tile_usage, total_steps):
    """Convert tile usage to a 2D numpy array for heatmap visualization."""
    heatmap = np.zeros((ROWS, COLS))
    
    # Convert tile numbers to grid coordinates and fill the heatmap
    for tile, count in tile_usage.items():
        # Convert tile number to x,y coordinates
        x = tile // COLS
        y = tile % COLS
        if x < ROWS and y < COLS:  # Ensure within bounds
            heatmap[x, y] = count / total_steps
    
    return heatmap

def plot_heatmap(heatmap, output_file=None, show_plot=True):
    """Plot and save the heatmap."""
    plt.figure(figsize=(12, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(heatmap, 
                cmap='YlOrRd',  # Yellow-Orange-Red colormap
                annot=False,    # Don't show numbers in cells
                fmt='.3f',
                cbar_kws={'label': 'Usage Probability'})
    
    plt.title('Tile Usage Heatmap')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved as {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def print_statistics(tile_usage, total_steps, heatmap):
    """Print statistics about tile usage."""
    print("\nTile Usage Statistics:")
    print(f"Total steps across all agents: {total_steps}")
    print(f"Unique tiles used: {len(tile_usage)}")
    
    # Find most and least used tiles
    most_used = max(tile_usage.items(), key=lambda x: x[1])
    print(f"\nMost used tile: {most_used[0]} (used {most_used[1]} times, {most_used[1]/total_steps*100:.2f}% of total)")
    
    # Calculate average usage
    avg_usage = total_steps / len(tile_usage)
    print(f"Average usage per tile: {avg_usage:.2f} times")
    
    # Usage distribution
    usage_counts = list(tile_usage.values())
    print("\nUsage Distribution:")
    print(f"Min usage: {min(usage_counts)}")
    print(f"Max usage: {max(usage_counts)}")
    print(f"Median usage: {np.median(usage_counts):.2f}")
    print(f"Mean usage: {np.mean(usage_counts):.2f}")
    print(f"Std dev: {np.std(usage_counts):.2f}")
    
    # Print tile number ranges
    print("\nTile Number Ranges:")
    print(f"Min tile: {min(tile_usage.keys())}")
    print(f"Max tile: {max(tile_usage.keys())}")
    
    # Print array sum (should be close to 1.0 since these are probabilities)
    array_sum = np.sum(heatmap)
    print(f"\nArray Sum (total probability): {array_sum:.6f}")
    if not np.isclose(array_sum, 1.0, rtol=1e-5):
        print("Note: Sum is not exactly 1.0, which might indicate some steps were outside the grid bounds")

def save_array(heatmap, output_file):
    """Save the numpy array."""
    # Save array in numpy format
    array_file = output_file.rsplit('.', 1)[0] + '_heatmap.npy'
    np.save(array_file, heatmap)
    print(f"\nNumpy array saved to {array_file}")
    
    # Print array preview
    print("\nArray Preview (showing probabilities > 0):")
    # Get non-zero positions
    nonzero_pos = np.argwhere(heatmap > 0)
    if len(nonzero_pos) > 0:
        print("\nFormat: [row, col]: probability")
        # Show first 10 non-zero positions
        for pos in nonzero_pos[:10]:
            print(f"[{pos[0]}, {pos[1]}]: {heatmap[pos[0], pos[1]]:.6f}")
        if len(nonzero_pos) > 10:
            print(f"... and {len(nonzero_pos) - 10} more non-zero positions")
    
    print("\nTo load this array in Python:")
    print(f"import numpy as np")
    print(f"heatmap = np.load('{array_file}')")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create a heatmap from paths.txt file')
    parser.add_argument('paths_file', help='Path to the paths.txt file')
    parser.add_argument('--output', '-o', help='Output file for the heatmap (e.g., heatmap.png)')
    parser.add_argument('--no-show', action='store_true', help='Do not display the plot window')
    
    args = parser.parse_args()
    
    # Parse the paths file
    print(f"Reading paths from {args.paths_file}...")
    tile_usage, total_steps = parse_paths_file(args.paths_file)
    
    # Create the heatmap
    heatmap = create_heatmap(tile_usage, total_steps)
    
    # Print statistics
    print_statistics(tile_usage, total_steps, heatmap)
    
    # Save numpy array
    if args.output:
        save_array(heatmap, args.output)
    
    # Plot the heatmap
    plot_heatmap(heatmap, args.output, not args.no_show)

if __name__ == "__main__":
    main() 