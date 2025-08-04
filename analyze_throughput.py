#!/usr/bin/env python3
import os
import re
import csv
from pathlib import Path

def extract_grid_number(folder_name):
    """Extract grid number from folder name like 'run_agents200_endpoints352_grid00023'"""
    match = re.search(r'grid(\d+)$', folder_name)
    if match:
        return int(match.group(1))
    return None

def extract_endpoints(folder_name):
    """Extract number of endpoints from folder name"""
    match = re.search(r'endpoints(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None

def extract_throughput(output_file):
    """Extract throughput from output.txt file"""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            # Look for throughput in the format "Throughput: X"
            match = re.search(r'Throughput:\s*([\d.]+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
    return None

def main():
    exp_dir = Path("exp")
    results = []

    # Ensure exp directory exists
    if not exp_dir.exists():
        print("Error: exp directory not found")
        return

    # Process each folder in exp directory
    for folder in exp_dir.iterdir():
        if not folder.is_dir():
            continue

        # Check if this is a run folder (contains grid number)
        grid_num = extract_grid_number(folder.name)
        if grid_num is None:
            continue

        # Extract endpoints
        endpoints = extract_endpoints(folder.name)
        if endpoints is None:
            continue

        # Look for output.txt
        output_file = folder / "output.txt"
        if not output_file.exists():
            print(f"Warning: No output.txt found in {folder}")
            continue

        # Extract throughput
        throughput = extract_throughput(output_file)
        if throughput is not None:
            results.append({
                'grid_number': grid_num,
                'endpoints': endpoints,
                'throughput': throughput,
                'folder': folder.name
            })
            print(f"Processed {folder.name}: throughput = {throughput}")

    # Sort results by grid number
    results.sort(key=lambda x: x['grid_number'])

    # Save to CSV
    if results:
        csv_file = "throughput_results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['grid_number', 'endpoints', 'throughput', 'folder'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_file}")
        print(f"Processed {len(results)} folders")
    else:
        print("No results found")

if __name__ == "__main__":
    main() 