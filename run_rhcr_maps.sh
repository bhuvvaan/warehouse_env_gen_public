#!/bin/bash

# Loop through all map files
for map_file in map_generation/maps/kiva_large_w_mode_grid_*.map; do
    # Extract the grid number from filename (00001, 00002, etc.)
    grid_num=$(basename "$map_file" .map | sed 's/kiva_large_w_mode_grid_//')
    
    # Get number of endpoints from the map file (second line)
    endpoints=$(head -2 "$map_file" | tail -1)
    
    # Create output directory with grid number
    output_dir="exp/run_agents200_endpoints${endpoints}_grid${grid_num}"
    mkdir -p "$output_dir"
    
    echo "Processing map: $map_file"
    echo "Output directory: $output_dir"
    echo "----------------------------------------"
    
    # Run RHCR command
    RHCR/lifelong -m "$map_file" -k 200 \
        --scenario=KIVA \
        --simulation_window=5 \
        --solver=PBS \
        --suboptimal_bound=1 \
        --seed=00 \
        --screen=1 \
        --simulation_time=1000 \
        --planning_window=10 \
        --output="$output_dir" 2>&1 | tee "$output_dir/output.txt"
        
    echo -e "\nFinished processing $map_file\n"
    echo "========================================"
done

echo "All maps processed successfully!" 