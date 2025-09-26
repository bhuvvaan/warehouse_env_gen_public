#!/usr/bin/env python3
"""
Unified Data Preparation Script
Ensures consistent data splitting across heatmap prediction and throughput prediction models.
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from sklearn.model_selection import train_test_split
import pickle

# ============================================================================
# UNIFIED SEED CONTROL - ALL SCRIPTS USE THE SAME SEEDS
# ============================================================================
SEED = 42
RANDOM_STATE = 42  # For sklearn functions
TEST_SIZE = 0.2  # 20% test data as requested

def set_seeds(seed=SEED):
    """Set seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds set to {seed} for reproducibility")

def load_warehouse_grids(json_path):
    """Load all grids from warehouse_grids.json."""
    print(f"Loading all grids from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    grids = {}
    for item in data['grids']:
        grid_id = item['grid_id']
        grid_str = item['grid']
        grids[grid_id] = grid_str
    
    print(f"Loaded {len(grids)} grids (limited to first 1000)")
    return grids

def process_grid_string_2channel(grid_str, rows=33, cols=36):
    """Process grid string into 2 separate one-hot encoded channels:
    Channel 0: Shelves (only '@' = 1, everything else = 0)
    Channel 1: Endpoints (only 'e' = 1, everything else = 0)
    """
    
    # Create 2 separate one-hot channels
    shelves_channel = np.zeros((rows, cols))
    endpoints_channel = np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(grid_str):
                char = grid_str[idx]
                if char == '@':  # Shelves - one-hot encoded
                    shelves_channel[i, j] = 1.0
                elif char == 'e':  # Endpoints - one-hot encoded
                    endpoints_channel[i, j] = 1.0
                # Everything else ('.', 'w') stays 0 in both channels
    
    # Remove first 2 and last 2 columns for both channels
    shelves_channel = shelves_channel[:, 2:-2]  # Shape: (33, 32)
    endpoints_channel = endpoints_channel[:, 2:-2]  # Shape: (33, 32)
    
    # Stack into 2-channel array: (2, 33, 32)
    grid_2channel = np.stack([shelves_channel, endpoints_channel], axis=0)
    
    return grid_2channel

def load_heatmap(heatmap_path):
    """Load heatmap from .npy file and remove first/last 2 columns to match grid dimensions."""
    if os.path.exists(heatmap_path):
        heatmap = np.load(heatmap_path)
        # Remove first 2 and last 2 columns to match grid dimensions (33x36 -> 33x32)
        heatmap = heatmap[:, 2:-2]  # Shape: (33, 32)
        
        # Recalculate probabilities by dividing by sum so that probabilities add up to 1
        heatmap_sum = np.sum(heatmap)
        if heatmap_sum > 0:
            heatmap = heatmap / heatmap_sum
        else:
            # If all values are zero, assign uniform probabilities
            heatmap = np.ones_like(heatmap) / heatmap.size
        
        return heatmap
    else:
        print(f"Warning: Heatmap not found at {heatmap_path}")
        return None

def prepare_unified_data_2channel():
    """Prepare 2-channel data with guaranteed consistent splitting."""
    print("Preparing unified 2-channel data with consistent splitting...")
    
    # Set seeds first
    set_seeds(SEED)
    
    # Load grids
    grids = load_warehouse_grids('map_generation/warehouse_grids.json')
    
    # Load throughput data
    print("Loading throughput data for stratification...")
    throughput_data = pd.read_csv('throughput_results.csv')
    throughput_dict = dict(zip(throughput_data['grid_number'], throughput_data['throughput']))
    
    # Prepare data lists
    processed_grids = []
    heatmaps = []
    grid_ids = []
    throughputs = []
    
    # Process each grid in sorted order to ensure consistency
    for grid_id in sorted(grids.keys()):
        grid_str = grids[grid_id]
        
        # Process grid string into 2 channels
        processed_grid_2channel = process_grid_string_2channel(grid_str)
        processed_grids.append(processed_grid_2channel)
        
        # Load corresponding heatmap
        heatmap_path = f'exp_heatmaps/grid_{grid_id:05d}_heatmap.npy'
        heatmap = load_heatmap(heatmap_path)
        
        if heatmap is not None:
            heatmaps.append(heatmap)
            grid_ids.append(grid_id)
            # Get throughput for this grid (default to 0 if not found)
            throughput = throughput_dict.get(grid_id, 0.0)
            throughputs.append(throughput)
        else:
            # Remove the grid if no heatmap exists
            processed_grids.pop()
    
    print(f"Successfully paired {len(grid_ids)} grids with heatmaps")
    
    return np.array(processed_grids), np.array(heatmaps), grid_ids, throughputs

def create_consistent_splits_2channel(processed_grids_2channel, heatmaps, grid_ids, throughputs):
    """Create consistent train/test splits that can be reused across models."""
    print("Creating consistent train/test splits for 2-channel data...")
    
    # Create throughput bins for stratification
    print("Creating throughput bins for stratified splitting...")
    throughputs = np.array(throughputs)
    
    # Sort throughput values and split into 9 bins evenly
    num_bins = 9
    sorted_indices = np.argsort(throughputs)
    samples_per_bin = len(throughputs) // num_bins
    remaining_samples = len(throughputs) % num_bins
    
    # Initialize labels
    throughput_labels = np.zeros(len(throughputs), dtype=int)
    
    # Assign bins
    current_idx = 0
    for bin_idx in range(num_bins):
        # Calculate how many samples this bin should get
        if bin_idx < remaining_samples:
            bin_size = samples_per_bin + 1
        else:
            bin_size = samples_per_bin
        
        # Assign labels for this bin
        end_idx = current_idx + bin_size
        bin_indices = sorted_indices[current_idx:end_idx]
        throughput_labels[bin_indices] = bin_idx + 1
        
        current_idx = end_idx
    
    print(f"Throughput range: {throughputs.min():.3f} to {throughputs.max():.3f}")
    print(f"Created {num_bins} evenly-distributed bins for stratification")
    
    # Count samples in each bin
    for i in range(1, num_bins + 1):
        bin_mask = throughput_labels == i
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_throughputs = throughputs[bin_mask]
            print(f"Bin {i}: {bin_count} samples, throughput range: {bin_throughputs.min():.3f} - {bin_throughputs.max():.3f}, avg: {bin_throughputs.mean():.3f}")
    
    # Create stratified train/test split based on throughput bins
    print(f"Creating stratified train/test split ({100*(1-TEST_SIZE):.0f}%/{100*TEST_SIZE:.0f}%)...")
    
    train_indices, test_indices = train_test_split(
        range(len(processed_grids_2channel)), 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=throughput_labels
    )
    
    # Split data
    train_grids = processed_grids_2channel[train_indices]
    train_heatmaps = heatmaps[train_indices]
    train_throughputs = throughputs[train_indices]
    train_grid_ids = [grid_ids[i] for i in train_indices]
    
    test_grids = processed_grids_2channel[test_indices]
    test_heatmaps = heatmaps[test_indices]
    test_throughputs = throughputs[test_indices]
    test_grid_ids = [grid_ids[i] for i in test_indices]
    
    print(f"Training set: {len(train_grids)} samples ({len(train_grids)/len(processed_grids_2channel)*100:.1f}%)")
    print(f"  Training throughput: min={train_throughputs.min():.3f}, max={train_throughputs.max():.3f}, mean={train_throughputs.mean():.3f}")
    print(f"Test set: {len(test_grids)} samples ({len(test_grids)/len(processed_grids_2channel)*100:.1f}%)")
    print(f"  Test throughput: min={test_throughputs.min():.3f}, max={test_throughputs.max():.3f}, mean={test_throughputs.mean():.3f}")
    
    # Print channel statistics
    print(f"\n2-Channel Data Statistics:")
    print(f"Training grids shape: {train_grids.shape}")  # Should be (N, 2, 33, 32)
    print(f"Test grids shape: {test_grids.shape}")  # Should be (N, 2, 33, 32)
    
    # Analyze each channel
    for ch_idx, ch_name in enumerate(['Shelves', 'Endpoints']):
        train_ch = train_grids[:, ch_idx, :, :]
        test_ch = test_grids[:, ch_idx, :, :]
        print(f"{ch_name} channel:")
        print(f"  Train: {np.sum(train_ch == 1)} ones, range [{train_ch.min():.1f}, {train_ch.max():.1f}]")
        print(f"  Test:  {np.sum(test_ch == 1)} ones, range [{test_ch.min():.1f}, {test_ch.max():.1f}]")
    
    return {
        'train': {
            'grids': train_grids,
            'heatmaps': train_heatmaps,
            'throughputs': train_throughputs,
            'grid_ids': train_grid_ids,
            'indices': train_indices
        },
        'test': {
            'grids': test_grids,
            'heatmaps': test_heatmaps,
            'throughputs': test_throughputs,
            'grid_ids': test_grid_ids,
            'indices': test_indices
        },
        'all_data': {
            'grids': processed_grids_2channel,
            'heatmaps': heatmaps,
            'throughputs': throughputs,
            'grid_ids': grid_ids,
            'throughput_labels': throughput_labels
        }
    }

def save_splits_2channel(splits, output_dir='unified_data_splits_2channel'):
    """Save the consistent 2-channel data splits for reuse across models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits as pickle files
    splits_file = os.path.join(output_dir, 'data_splits_2channel.pkl')
    with open(splits_file, 'wb') as f:
        pickle.dump(splits, f)
    
    # Save as numpy files for easy loading
    np.save(os.path.join(output_dir, 'train_grids_2channel.npy'), splits['train']['grids'])
    np.save(os.path.join(output_dir, 'train_heatmaps.npy'), splits['train']['heatmaps'])
    np.save(os.path.join(output_dir, 'train_throughputs.npy'), splits['train']['throughputs'])
    np.save(os.path.join(output_dir, 'train_grid_ids.npy'), splits['train']['grid_ids'])
    
    np.save(os.path.join(output_dir, 'test_grids_2channel.npy'), splits['test']['grids'])
    np.save(os.path.join(output_dir, 'test_heatmaps.npy'), splits['test']['heatmaps'])
    np.save(os.path.join(output_dir, 'test_throughputs.npy'), splits['test']['throughputs'])
    np.save(os.path.join(output_dir, 'test_grid_ids.npy'), splits['test']['grid_ids'])
    
    # Save metadata
    metadata = {
        'seed': SEED,
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'total_samples': len(splits['all_data']['grids']),
        'train_samples': len(splits['train']['grids']),
        'test_samples': len(splits['test']['grids']),
        'grid_shape': splits['train']['grids'].shape[1:],  # (2, 33, 32)
        'num_channels': 2,
        'channel_names': ['shelves', 'endpoints'],
        'channel_encodings': {
            'shelves': 'one-hot: @ = 1, rest = 0',
            'endpoints': 'one-hot: e = 1, rest = 0'
        },
        'throughput_range': {
            'min': float(splits['all_data']['throughputs'].min()),
            'max': float(splits['all_data']['throughputs'].max()),
            'mean': float(splits['all_data']['throughputs'].mean())
        }
    }
    
    metadata_file = os.path.join(output_dir, 'metadata_2channel.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"2-channel data splits saved to {output_dir}/")
    print(f"Metadata: {metadata}")
    
    return output_dir

def main():
    """Main function to prepare and save unified 2-channel data splits."""
    print("=" * 60)
    print("UNIFIED DATA PREPARATION - 2 CHANNEL VERSION")
    print("=" * 60)
    print("Channels: [0] Shelves (one-hot: @=1), [1] Endpoints (one-hot: e=1)")
    
    # Prepare data
    processed_grids_2channel, heatmaps, grid_ids, throughputs = prepare_unified_data_2channel()
    
    # Create consistent splits
    splits = create_consistent_splits_2channel(processed_grids_2channel, heatmaps, grid_ids, throughputs)
    
    # Save splits for reuse
    output_dir = save_splits_2channel(splits)
    
    print("\n" + "=" * 60)
    print("2-CHANNEL DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Use the saved splits in {output_dir}/ for consistent training across models")
    print("\nTo load the splits in other scripts, use:")
    print("import pickle")
    print("with open('unified_data_splits_2channel/data_splits_2channel.pkl', 'rb') as f:")
    print("    splits = pickle.load(f)")
    print("\nOr load individual numpy files:")
    print("train_grids = np.load('unified_data_splits_2channel/train_grids_2channel.npy')")
    print("train_heatmaps = np.load('unified_data_splits_2channel/train_heatmaps.npy')")
    print("train_throughputs = np.load('unified_data_splits_2channel/train_throughputs.npy')")
    print("\nGrid shape: (N_samples, 2_channels, 33_rows, 32_cols)")
    print("Channel order: [shelves, endpoints] (both one-hot encoded)")

if __name__ == "__main__":
    main() 