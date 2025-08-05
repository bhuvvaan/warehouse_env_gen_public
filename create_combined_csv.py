import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import random
import os

# Global random seed for reproducibility
RANDOM_SEED = 12

# Set seeds for reproducibility
def set_seeds(seed=RANDOM_SEED):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds set to {seed} for reproducibility")

# Set seeds at the start
set_seeds()

def create_onehot_channel(grid_str, target_char, rows=33, cols=36):
    """Create one-hot encoding channel for a specific character."""
    channel = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(grid_str) and grid_str[idx] == target_char:
                channel[i, j] = 1
    return channel[:, 2:-2]  # Remove first/last two columns

def extract_features(grid_channel, heatmap):
    """Extract meaningful features from grid channels and heatmap."""
    features = {}
    
    # Grid statistics
    features['grid_sum'] = np.sum(grid_channel)
    features['grid_mean'] = np.mean(grid_channel)
    features['grid_std'] = np.std(grid_channel)
    
    # Count non-zero elements in different regions
    h, w = grid_channel.shape
    features['grid_top'] = np.sum(grid_channel[:h//3, :])
    features['grid_middle'] = np.sum(grid_channel[h//3:2*h//3, :])
    features['grid_bottom'] = np.sum(grid_channel[2*h//3:, :])
    features['grid_left'] = np.sum(grid_channel[:, :w//3])
    features['grid_center'] = np.sum(grid_channel[:, w//3:2*w//3])
    features['grid_right'] = np.sum(grid_channel[:, 2*w//3:])
    
    # Distance features
    y_indices, x_indices = np.nonzero(grid_channel)
    if len(y_indices) > 0:
        features['mean_x'] = np.mean(x_indices)
        features['mean_y'] = np.mean(y_indices)
        features['std_x'] = np.std(x_indices)
        features['std_y'] = np.std(y_indices)
    else:
        features['mean_x'] = w/2
        features['mean_y'] = h/2
        features['std_x'] = 0
        features['std_y'] = 0
    
    # Heatmap statistics
    features['heatmap_mean'] = np.mean(heatmap)
    features['heatmap_std'] = np.std(heatmap)
    features['heatmap_max'] = np.max(heatmap)
    features['heatmap_min'] = np.min(heatmap)
    
    # Interaction between grid and heatmap
    overlap = grid_channel * heatmap
    features['overlap_mean'] = np.mean(overlap)
    features['overlap_sum'] = np.sum(overlap)
    
    return features

# Define ResidualBlock2D
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out

# Define OccupancyGridResNet2D
class OccupancyGridResNet2D(nn.Module):
    def __init__(self, input_channels=2):
        super(OccupancyGridResNet2D, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_block1 = ResidualBlock2D(64, 128)
        self.res_block2 = ResidualBlock2D(128, 256)
        self.res_block3 = ResidualBlock2D(256, 128)
        self.res_block4 = ResidualBlock2D(128, 64)
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.dropout(x)
        
        x = self.res_block2(x)
        x = self.dropout(x)
        
        x = self.res_block3(x)
        x = self.dropout(x)
        
        x = self.res_block4(x)
        x = self.dropout(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Reshape for global softmax
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Flatten to [batch_size, height*width]
        
        # Apply softmax across all spatial positions to ensure sum of all probabilities is 1
        x_prob = F.softmax(x_flat, dim=1)
        
        # Reshape back to original dimensions
        x_out = x_prob.view(x.shape)
        
        return x_out

def predict_throughput_for_grid(grid_str, occupancy_model, xgb_model, device):
    """Predict throughput for a single grid using the trained models."""
    # Create input channels (strip first 2 and last 2 columns)
    shelf_channel = create_onehot_channel(grid_str, '@')
    endpoint_channel = create_onehot_channel(grid_str, 'e')
    
    # Prepare input for the occupancy model
    grid_input = torch.FloatTensor(np.stack([shelf_channel, endpoint_channel])[None, ...]).to(device)
    
    # Get predicted heatmap
    with torch.no_grad():
        predicted_heatmap = occupancy_model(grid_input)
        predicted_heatmap = predicted_heatmap.squeeze().cpu().numpy()

    # Normalize the heatmap by dividing by sum (just in case)
    predicted_heatmap = predicted_heatmap / predicted_heatmap.sum()
    
    # Extract features using predicted heatmap
    shelf_features = extract_features(shelf_channel, predicted_heatmap)
    endpoint_features = extract_features(endpoint_channel, predicted_heatmap)
    
    # Combine features with appropriate prefixes
    features = {}
    for k, v in shelf_features.items():
        features[f'shelf_{k}'] = v
    for k, v in endpoint_features.items():
        features[f'endpoint_{k}'] = v
    
    # Convert to DataFrame for XGBoost
    features_df = pd.DataFrame([features])
    
    # Create XGBoost dataset
    dtest = xgb.DMatrix(features_df)
    
    # Predict throughput
    predicted_throughput = xgb_model.predict(dtest)[0]
    
    return predicted_throughput

print("Loading models...")

# Load the trained occupancy grid model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the occupancy model
occupancy_model = OccupancyGridResNet2D().to(device)

# Load the trained weights
occupancy_model.load_state_dict(torch.load('best_occupancy_grid_resnet_model.pth'))
occupancy_model.eval()
print("Occupancy grid model loaded successfully")

# Load the trained XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_throughput_predicted_model.json')
print("XGBoost model loaded successfully")

print("\nLoading data files...")

# Load throughput results for grids 1-1000
throughput_df = pd.read_csv('throughput_results.csv')
throughput_dict = dict(zip(throughput_df['grid_number'], throughput_df['throughput']))
print(f"Loaded {len(throughput_dict)} actual throughput values")

# Load predicted throughput for grids 1001-50000
predicted_df = pd.read_csv('predicted_throughput_all_grids.csv')
predicted_dict = dict(zip(predicted_df['grid_id'], predicted_df['predicted_throughput']))
print(f"Loaded {len(predicted_dict)} predicted throughput values")

# Load grids from warehouse_grids.json (first 1000)
print("\nLoading first 1000 grids from warehouse_grids.json...")
with open('./map_generation/warehouse_grids.json', 'r') as f:
    json_data = json.load(f)
    grids_1_1000 = {str(grid['grid_id']): grid for grid in json_data['grids']}
print(f"Loaded {len(grids_1_1000)} grids from warehouse_grids.json")

# Load grids from warehouse_grids_2.json (grids 1001-50000)
print("\nLoading grids 1001-50000 from warehouse_grids_2.json...")
with open('./map_generation/warehouse_grids_2.json', 'r') as f:
    json_data = json.load(f)
    grids_1001_50000 = {str(grid['grid_id']): grid for grid in json_data['grids']}
print(f"Loaded {len(grids_1001_50000)} grids from warehouse_grids_2.json")

print("\nCreating combined CSV...")

# Initialize results list
results = []

# Process grids 1-1000
print("Processing grids 1-1000...")
for grid_id in range(1, 1001):
    grid_id_str = str(grid_id)
    if grid_id_str in grids_1_1000:
        grid_data = grids_1_1000[grid_id_str]
        
        # Check if we have actual throughput data
        if grid_id in throughput_dict:
            throughput = throughput_dict[grid_id]
            source = "actual"
        else:
            # Predict throughput using the models
            throughput = predict_throughput_for_grid(grid_data['grid'], occupancy_model, xgb_model, device)
            source = "predicted"
        
        results.append({
            'grid_number': grid_id,
            'grid': grid_data['grid'],
            'throughput': throughput,
            'source': source
        })
        
        if grid_id % 100 == 0:
            print(f"Processed grid {grid_id}")

# Process grids 1001-50000
print("\nProcessing grids 1001-50000...")
for grid_id in range(1001, 50001):
    grid_id_str = str(grid_id)
    if grid_id_str in grids_1001_50000:
        grid_data = grids_1001_50000[grid_id_str]
        
        # Use predicted throughput from the CSV
        if grid_id in predicted_dict:
            throughput = predicted_dict[grid_id]
            source = "predicted_csv"
        else:
            # Fallback: predict using models
            throughput = predict_throughput_for_grid(grid_data['grid'], occupancy_model, xgb_model, device)
            source = "predicted_model"
        
        results.append({
            'grid_number': grid_id,
            'grid': grid_data['grid'],
            'throughput': throughput,
            'source': source
        })
        
        if grid_id % 5000 == 0:
            print(f"Processed grid {grid_id}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Sort by grid number
results_df = results_df.sort_values('grid_number')

# Save to CSV
output_file = 'combined_grids_throughput.csv'
results_df.to_csv(output_file, index=False)

print(f"\nResults saved to {output_file}")

# Display summary statistics
print("\nSummary:")
print(f"Total grids processed: {len(results_df)}")
print(f"Grids with actual throughput: {len(results_df[results_df['source'] == 'actual'])}")
print(f"Grids with predicted throughput: {len(results_df[results_df['source'] == 'predicted'])}")
print(f"Grids with predicted throughput from CSV: {len(results_df[results_df['source'] == 'predicted_csv'])}")
print(f"Grids with predicted throughput from model: {len(results_df[results_df['source'] == 'predicted_model'])}")

print(f"\nThroughput statistics:")
print(f"Mean: {results_df['throughput'].mean():.3f}")
print(f"Std: {results_df['throughput'].std():.3f}")
print(f"Min: {results_df['throughput'].min():.3f}")
print(f"Max: {results_df['throughput'].max():.3f}")

print("\nCombined CSV creation completed!") 