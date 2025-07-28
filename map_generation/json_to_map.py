#!/usr/bin/env python3
import json
import os

# Create maps directory if it doesn't exist
os.makedirs("map_generation/maps", exist_ok=True)

def process_json_line(line):
    """Process a single line from the JSON file."""
    line = line.strip()
    if not line or line in "[]":  # Skip empty lines and brackets
        return None
    if line.endswith(","):  # Remove trailing comma
        line = line[:-1]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None

def break_into_rows(grid_str, width):
    """Break a grid string into rows of specified width."""
    return [grid_str[i:i+width] for i in range(0, len(grid_str), width)]

def process_grid(grid):
    """Process a single grid and save it to a file."""
    grid_id = grid["grid_id"]
    grid_str = grid["grid"]
    e_count = grid["e_count"]

    # Get dimensions from first line
    height, width = 33, 36

    # Break grid string into rows
    grid_rows = break_into_rows(grid_str, width)

    # Create the .map file content
    map_content = [
        f"{height},{width}",  # Fixed first line
        str(e_count),         # Number of endpoints from json
        str(e_count),         # Same number repeated
        "1000",               # Fixed fourth line
    ] + grid_rows            # Add each row of the grid

    # Write to file
    output_filename = f"map_generation/maps/kiva_large_w_mode_grid_{grid_id:05d}.map"
    with open(output_filename, "w") as f:
        f.write("\n".join(map_content))

    print(f"Created {output_filename}")
    print(f"Grid ID: {grid_id}")
    print(f"Endpoint count: {e_count}")
    print(f"Grid dimensions: {height}x{width}")
    print(f"Number of rows: {len(grid_rows)}")
    print(f"Row length: {len(grid_rows[0]) if grid_rows else 0}")
    print("-" * 50)

# Read the first 10 grids from the json file
with open("map_generation/warehouse_grids.json", "r") as f:
    # Skip the first line if it's just a [
    first_line = f.readline().strip()
    if first_line != "[":
        f.seek(0)
    
    # Process first 10 valid grids
    grids_processed = 0
    while grids_processed < 10:
        line = f.readline()
        if not line:
            break
        
        grid = process_json_line(line)
        if grid is not None:
            process_grid(grid)
            grids_processed += 1

print(f"\nProcessed {grids_processed} grids successfully.") 