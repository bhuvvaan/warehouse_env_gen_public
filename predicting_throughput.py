import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
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

print("Loading data...")

# Load grids from JSON file
print("Reading warehouse_grids.json...")
with open('./map_generation/warehouse_grids.json', 'r') as f:
    json_data = json.load(f)
    grids_data = {str(grid['grid_id']): grid for grid in json_data['grids']}
print(f"Found {len(grids_data)} grids in JSON")

# Load throughput data
print("\nLoading throughput data...")
throughput_df = pd.read_csv('throughput_results.csv')
print(f"Found {len(throughput_df)} throughput entries")

# Load the trained occupancy grid model
print("\nLoading trained occupancy grid model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
occupancy_model = OccupancyGridResNet2D().to(device)

# Load the trained weights
occupancy_model.load_state_dict(torch.load('best_occupancy_grid_resnet_model.pth'))
occupancy_model.eval()

print("Model loaded successfully")

# Process data
print("\nProcessing data...")
all_features = []
output_throughputs = []
processed_count = 0

# Get unique grid numbers from throughput data
grid_numbers = throughput_df['grid_number'].unique()

for grid_num in grid_numbers:
    # Get grid data and throughput
    grid_data = grids_data.get(str(grid_num))
    throughput = throughput_df[throughput_df['grid_number'] == grid_num]['throughput'].values
    
    if grid_data is None or len(throughput) == 0:
        continue
    
    # Create input channels
    shelf_channel = create_onehot_channel(grid_data['grid'], '@')
    endpoint_channel = create_onehot_channel(grid_data['grid'], 'e')
    
    # Prepare input for the occupancy model
    grid_input = torch.FloatTensor(np.stack([shelf_channel, endpoint_channel])[None, ...]).to(device)
    
    # Get predicted heatmap
    with torch.no_grad():
        predicted_heatmap = occupancy_model(grid_input)
        predicted_heatmap = predicted_heatmap.squeeze().cpu().numpy()

    # Ensure probabilities sum to 1
    predicted_heatmap = predicted_heatmap / predicted_heatmap.sum()

    # Min-max normalize the heatmap
    #predicted_heatmap = (predicted_heatmap - predicted_heatmap.min()) / (predicted_heatmap.max() - predicted_heatmap.min())
    
    # Extract features using predicted heatmap
    shelf_features = extract_features(shelf_channel, predicted_heatmap)
    endpoint_features = extract_features(endpoint_channel, predicted_heatmap)
    
    # Combine features with appropriate prefixes
    features = {}
    for k, v in shelf_features.items():
        features[f'shelf_{k}'] = v
    for k, v in endpoint_features.items():
        features[f'endpoint_{k}'] = v
    
    all_features.append(features)
    output_throughputs.append(throughput[0])
    processed_count += 1

print(f"\nSuccessfully processed {processed_count} samples")

# Convert to DataFrame
X = pd.DataFrame(all_features)
y = np.array(output_throughputs)

print(f"\nFeature names:")
print(X.columns.tolist())

# Create bins for throughput values for stratification
print("\nCreating throughput bins for stratification...")
n_bins = 10  # Number of bins for stratification
kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
throughput_bins = kbd.fit_transform(y.reshape(-1, 1)).flatten()

print(f"\nThroughput distribution across bins:")
for bin_idx in range(n_bins):
    count = np.sum(throughput_bins == bin_idx)
    print(f"Bin {bin_idx}: {count} samples ({count/len(throughput_bins)*100:.1f}%)")

# First split into train+val and test using stratification
X_train_val, X_test, y_train_val, y_test, bins_train_val, bins_test = train_test_split(
    X, y, throughput_bins,
    test_size=0.1,
    random_state=RANDOM_SEED,
    stratify=throughput_bins
)

# Then split train+val into train and val, maintaining stratification
X_train, X_val, y_train, y_val, bins_train, bins_val = train_test_split(
    X_train_val, y_train_val, bins_train_val,
    test_size=0.1,  # 0.1 * 0.9 = 0.09 of total data
    random_state=RANDOM_SEED,
    stratify=bins_train_val
)

print("\nSplit sizes:")
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Verify stratification
print("\nThroughput distribution in splits:")
for split_name, split_bins in [("Train", bins_train), ("Val", bins_val), ("Test", bins_test)]:
    print(f"\n{split_name} set distribution:")
    for bin_idx in range(n_bins):
        count = np.sum(split_bins == bin_idx)
        print(f"Bin {bin_idx}: {count} samples ({count/len(split_bins)*100:.1f}%)")

# Create XGBoost datasets
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse', 'mae'],
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED
}

print("\nTraining XGBoost model...")
print("Parameters:", params)

# Train model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=200,
    verbose_eval=100
)

print("\nMaking predictions...")
predictions = model.predict(dtest)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nTest Results:")
print(f"Mean Squared Error: {mse:.6f}")
print(f"Root Mean Square Error: {rmse:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"R² Score: {r2:.6f}")

# Calculate metrics for low and high throughput separately
low_mask = y_test < 4.0
high_mask = ~low_mask

print(f"\nLow Throughput (< 4.0) Metrics:")
print(f"Count: {np.sum(low_mask)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test[low_mask], predictions[low_mask]):.2f}")
print(f"Root Mean Square Error: {np.sqrt(mean_squared_error(y_test[low_mask], predictions[low_mask])):.2f}")

print(f"\nHigh Throughput (>= 4.0) Metrics:")
print(f"Count: {np.sum(high_mask)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test[high_mask], predictions[high_mask]):.2f}")
print(f"Root Mean Square Error: {np.sqrt(mean_squared_error(y_test[high_mask], predictions[high_mask])):.2f}")

# Plot feature importance
importance_scores = model.get_score(importance_type='gain')
importance_df = pd.DataFrame(
    {'Feature': list(importance_scores.keys()),
     'Importance': list(importance_scores.values())}
)
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(range(len(importance_df)), importance_df['Importance'])
plt.xticks(range(len(importance_df)), importance_df['Feature'], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance (Gain)')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_predicted.png')
plt.close()

# Plot predictions vs actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test[low_mask], predictions[low_mask], 
           alpha=0.5, label='Low Throughput (<4.0)', color='red')
plt.scatter(y_test[high_mask], predictions[high_mask], 
           alpha=0.5, label='High Throughput (>=4.0)', color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
plt.xlabel('Actual Throughput')
plt.ylabel('Predicted Throughput')
plt.title('Predictions vs Actuals')
plt.legend()
plt.grid(True)
plt.savefig('throughput_predictions_vs_actuals_predicted.png')
plt.close()

print("\nSaved files:")
print("- feature_importance_predicted.png")
print("- throughput_predictions_vs_actuals_predicted.png")

# Save the model
model.save_model('xgboost_throughput_predicted_model.json')
print("- xgboost_throughput_predicted_model.json")



