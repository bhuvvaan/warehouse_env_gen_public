#!/bin/bash

# Loop through all diffusion map files
for map_file in diffusion_maps_kiva_format/diffusion_map_*.map; do
    # Extract the map number from filename (001, 002, etc.)
    map_num=$(basename "$map_file" .map | sed 's/diffusion_map_//')
    
    # Get number of endpoints from the map file (second line)
    endpoints=$(head -2 "$map_file" | tail -1)
    
    # Create output directory with map number
    output_dir="diffusion_exp/run_agents200_endpoints${endpoints}_map${map_num}"
    mkdir -p "$output_dir"
    
    echo "Processing diffusion map: $map_file"
    echo "Map number: $map_num"
    echo "Endpoints: $endpoints"
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

echo "All diffusion maps processed successfully!"

# Now extract throughput values and create CSV
echo "Extracting throughput values and creating CSV..."

python3 << 'EOF'
import os
import re
import pandas as pd
import glob

def extract_throughput_from_output(output_file):
    """Extract throughput value from RHCR output file."""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Look for throughput pattern in the output
        # Common patterns: "Throughput: X.XXX" or "Average throughput: X.XXX"
        throughput_patterns = [
            r'Throughput:\s*([0-9]+\.?[0-9]*)',
            r'Average throughput:\s*([0-9]+\.?[0-9]*)',
            r'throughput:\s*([0-9]+\.?[0-9]*)',
            r'Total throughput:\s*([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in throughput_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # If no throughput found, return None
        print(f"Warning: No throughput found in {output_file}")
        return None
        
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
        return None

def main():
    # Find all output files
    output_files = glob.glob("diffusion_exp/*/output.txt")
    
    if not output_files:
        print("No output files found in diffusion_exp/")
        return
    
    print(f"Found {len(output_files)} output files")
    
    # Extract data from each file
    data = []
    for output_file in sorted(output_files):
        # Extract map number from path
        # Path format: diffusion_exp/run_agents200_endpointsXXX_mapYYY/output.txt
        path_parts = output_file.split('/')
        if len(path_parts) >= 3:
            dir_name = path_parts[1]  # run_agents200_endpointsXXX_mapYYY
            
            # Extract map number
            map_match = re.search(r'map(\d+)', dir_name)
            if map_match:
                map_num = int(map_match.group(1))
            else:
                continue
            
            # Extract endpoints
            endpoints_match = re.search(r'endpoints(\d+)', dir_name)
            if endpoints_match:
                endpoints = int(endpoints_match.group(1))
            else:
                endpoints = None
            
            # Extract throughput
            throughput = extract_throughput_from_output(output_file)
            
            data.append({
                'map_number': map_num,
                'endpoints': endpoints,
                'throughput': throughput,
                'output_file': output_file
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by map number
    df = df.sort_values('map_number')
    
    # Save to CSV
    output_csv = "diffusion_maps_throughput_results.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Total maps processed: {len(df)}")
    
    # Print summary statistics
    valid_throughputs = df[df['throughput'].notna()]['throughput']
    if len(valid_throughputs) > 0:
        print(f"\nThroughput Statistics:")
        print(f"  Valid results: {len(valid_throughputs)}/{len(df)}")
        print(f"  Mean throughput: {valid_throughputs.mean():.3f}")
        print(f"  Min throughput: {valid_throughputs.min():.3f}")
        print(f"  Max throughput: {valid_throughputs.max():.3f}")
        print(f"  Std throughput: {valid_throughputs.std():.3f}")
        
        # Show first few results
        print(f"\nFirst 10 results:")
        print(df.head(10)[['map_number', 'endpoints', 'throughput']].to_string(index=False))
    else:
        print("No valid throughput values found!")

if __name__ == "__main__":
    main()
EOF

echo "Throughput extraction complete!"
echo "Results saved in: diffusion_maps_throughput_results.csv" 