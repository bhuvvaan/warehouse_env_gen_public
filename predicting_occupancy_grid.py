import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import random
import os

# Set seeds for reproducibility
def set_seeds(seed=42):
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
set_seeds(42)

def visualize_grid(grid_str, rows=33, cols=36):
    """Convert grid string to a readable format with line breaks."""
    # Replace characters with more readable symbols
    grid_str = (grid_str
                .replace('.', '⋅')  # Empty space
                .replace('@', '█')  # Wall/obstacle
                .replace('e', 'E')  # Endpoint
                .replace('w', 'W')  # Workstation
                )
    # Split into rows
    return '\n'.join(grid_str[i:i+cols] for i in range(0, len(grid_str), cols))

def preprocess_heatmap(heatmap):
    """Remove first/last two columns and rebalance probabilities."""
    # Remove first/last two columns
    trimmed = heatmap[:, 2:-2]
    # Rebalance probabilities to sum to 1
    return trimmed / np.sum(trimmed)

def create_onehot_channel(grid_str, target_char, rows=33, cols=36):
    """Create one-hot encoding channel for a specific character."""
    channel = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(grid_str) and grid_str[idx] == target_char:
                channel[i, j] = 1
    return channel[:, 2:-2]  # Remove first/last two columns

class WarehouseDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):  # Changed padding to 2
        super(ResidualBlock2D, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (if input and output dimensions don't match)
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
        
        # Add residual connection
        out += residual
        out = F.relu(out)
        
        return out

class OccupancyGridResNet2D(nn.Module):
    def __init__(self, input_channels=2):  # 2 channels: shelves and endpoints
        super(OccupancyGridResNet2D, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)  # Changed padding to 2
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_block1 = ResidualBlock2D(64, 128)
        self.res_block2 = ResidualBlock2D(128, 256)
        self.res_block3 = ResidualBlock2D(256, 128)
        self.res_block4 = ResidualBlock2D(128, 64)
        
        # Final convolution to map to output
        self.final_conv = nn.Conv2d(64, 1, kernel_size=5, padding=2)  # Changed padding to 2
        
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
    # Convert list of grids to dictionary keyed by grid_id
    grids_data = {str(grid['grid_id']): grid for grid in json_data['grids']}
print(f"Found {len(grids_data)} grids in JSON")

# Get list of all heatmap files
heatmap_dir = Path("exp_heatmaps")
heatmap_files = sorted(heatmap_dir.glob("grid_*_heatmap.npy"))
print(f"Found {len(heatmap_files)} heatmap files")

# Arrays to store processed data
input_features = []  # Will contain shelf and endpoint channels
output_labels = []   # Will contain heatmaps

# Process data
print("\nProcessing data...")
for heatmap_file in heatmap_files:
    # Extract grid number from filename
    grid_num = int(heatmap_file.stem.split('_')[1])
    
    # Get grid data
    grid_data = grids_data.get(str(grid_num))
    if grid_data is None:
        continue
    
    # Create input channels (shelf and endpoint layouts)
    shelf_channel = create_onehot_channel(grid_data['grid'], '@')
    endpoint_channel = create_onehot_channel(grid_data['grid'], 'e')
    
    # Stack input channels
    features = np.stack([shelf_channel, endpoint_channel])
    
    # Load and process heatmap (label)
    heatmap = np.load(heatmap_file)
    heatmap = preprocess_heatmap(heatmap)
    
    input_features.append(features)
    output_labels.append(heatmap[np.newaxis, :, :])  # Add channel dimension

# Convert to numpy arrays
X = np.array(input_features)
y = np.array(output_labels)

print(f"\nData shapes:")
print(f"Input features (batch, channels, height, width): {X.shape}")
print(f"Output labels (batch, channels, height, width): {y.shape}")

# Load throughput data for stratification
print("\nLoading throughput data for stratified splitting...")
throughput_df = pd.read_csv('throughput_results.csv')

# Create bins for throughput values for stratification
print("Creating throughput bins for stratification...")
n_bins = 10  # Number of bins for stratification
kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
throughput_bins = kbd.fit_transform(throughput_df[['throughput']]).flatten()

# Create a mapping from grid numbers to throughput bins
grid_to_bin = dict(zip(throughput_df['grid_number'], throughput_bins))

# Get grid numbers from heatmap files
grid_numbers = [int(f.stem.split('_')[1]) for f in heatmap_files]

# Create stratification labels for our dataset
strat_labels = np.array([grid_to_bin.get(gnum, 0) for gnum in grid_numbers])

print(f"\nThroughput distribution across bins:")
for bin_idx in range(n_bins):
    count = np.sum(strat_labels == bin_idx)
    print(f"Bin {bin_idx}: {count} samples")

# First split into train+val and test using stratification
indices = np.arange(len(y))
train_val_idx, test_idx = train_test_split(
    indices, 
    test_size=0.1, 
    random_state=42,
    stratify=strat_labels
)

# Then split train+val into train and val, maintaining stratification
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.1,  # 0.1 * 0.9 = 0.09 of total data
    random_state=42,
    stratify=strat_labels[train_val_idx]
)

print("\nInitial split sizes (before augmentation):")
print(f"Training samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")
print(f"Test samples: {len(test_idx)}")

# Verify stratification
print("\nThroughput distribution in splits:")
for split_name, split_idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
    split_bins = strat_labels[split_idx]
    print(f"\n{split_name} set distribution:")
    for bin_idx in range(n_bins):
        count = np.sum(split_bins == bin_idx)
        print(f"Bin {bin_idx}: {count} samples ({count/len(split_idx)*100:.1f}%)")

# Data augmentation - ONLY for training data
print("\nData augmentation (training data only):")
print("Creating horizontally flipped versions of training data...")

# Create flipped versions of training data
X_train_flipped = np.flip(X[train_idx], axis=3).copy()  # Flip horizontally and make a copy
y_train_flipped = np.flip(y[train_idx], axis=3).copy()  # Flip horizontally and make a copy

print(f"Original training samples: {len(train_idx)}")
print(f"After adding flipped samples: {len(train_idx) * 2}")

# Create training dataset with original and flipped data
train_dataset_orig = WarehouseDataset(X[train_idx], y[train_idx])
train_dataset_flipped = WarehouseDataset(X_train_flipped, y_train_flipped)
train_dataset = ConcatDataset([train_dataset_orig, train_dataset_flipped])

# Create validation and test datasets (NO augmentation to prevent data leakage)
val_dataset = WarehouseDataset(X[val_idx], y[val_idx])
test_dataset = WarehouseDataset(X[test_idx], y[test_idx])

# Verify no overlap between sets
assert len(set(train_idx) & set(val_idx)) == 0, "Training and validation sets overlap!"
assert len(set(train_idx) & set(test_idx)) == 0, "Training and test sets overlap!"
assert len(set(val_idx) & set(test_idx)) == 0, "Validation and test sets overlap!"

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

print("\nFinal dataset sizes:")
print(f"Training samples (with augmentation): {len(train_dataset)}")
print(f"Validation samples (no augmentation): {len(val_dataset)}")
print(f"Test samples (no augmentation): {len(test_dataset)}")
print("Note: Augmentation is only applied to training data to prevent data leakage")

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OccupancyGridResNet2D(input_channels=2).to(device)
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(model.parameters(), lr=0.001)  

print(f"\nTraining ResNet model...")
print(f"Using device: {device}")
print(f"Optimizer: Adam with lr=0.001")

def compute_rmse(pred, target):
    """Compute RMSE between predicted and target heatmaps."""
    return torch.sqrt(torch.mean((pred - target) ** 2))

# Training loop
print("\nTraining for 100 epochs...")
num_epochs = 100
best_val_loss = float('inf')
best_model_state = None
val_losses = []
val_rmse = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_rmse = 0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        
        # Compute KL divergence loss
        loss = criterion(torch.log(outputs + 1e-10), batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_rmse += compute_rmse(outputs, batch_labels).item()
    
    # Validation
    model.eval()
    val_loss = 0
    val_rmse_epoch = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_inputs)
            val_loss += criterion(torch.log(outputs + 1e-10), batch_labels).item()
            val_rmse_epoch += compute_rmse(outputs, batch_labels).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_rmse = train_rmse / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_rmse = val_rmse_epoch / len(val_loader)
    
    val_losses.append(avg_val_loss)
    val_rmse.append(avg_val_rmse)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict().copy()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Train RMSE: {avg_train_rmse:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, '
              f'Val RMSE: {avg_val_rmse:.6f}')

print(f"\nBest validation loss: {best_val_loss:.6f}")

# Save the best model
if best_model_state is not None:
    torch.save(best_model_state, "best_occupancy_grid_resnet_model.pth")
    print("Best model saved to 'best_occupancy_grid_resnet_model.pth'")

# Load best model and evaluate on test set
print("\nEvaluating best model on test set...")
model.load_state_dict(best_model_state)
model.eval()

test_loss = 0
test_rmse = 0
with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_inputs)
        test_loss += criterion(torch.log(outputs + 1e-10), batch_labels).item()
        test_rmse += compute_rmse(outputs, batch_labels).item()

avg_test_loss = test_loss / len(test_loader)
avg_test_rmse = test_rmse / len(test_loader)

print(f"\nTest Results:")
print(f"KL Divergence Loss: {avg_test_loss:.6f}")
print(f"RMSE: {avg_test_rmse:.6f}")

# Plot validation metrics history
plt.figure(figsize=(15, 5))

# Plot KL divergence loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), val_losses)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss (KL Divergence)')
plt.title('Validation Loss History')
plt.grid(True)

# Plot RMSE
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_rmse)
plt.xlabel('Epoch')
plt.ylabel('Validation RMSE')
plt.title('Validation RMSE History')
plt.grid(True)

plt.tight_layout()
plt.savefig('validation_metrics.png')
plt.close()

# Visualize predictions
print("\nGenerating visualizations...")
def plot_heatmap(data, title, ax):
    im = ax.imshow(data, cmap='YlOrRd')
    ax.set_title(title)
    return im

model.eval()
with torch.no_grad():
    # Get a batch of test data
    test_features, test_labels = next(iter(test_loader))
    test_features = test_features.to(device)
    predictions = model(test_features)
    
    # Move everything to CPU for visualization
    test_features = test_features.cpu().numpy()
    test_labels = test_labels.numpy()
    predictions = predictions.cpu().numpy()
    
    # Plot first 5 examples
    for i in range(min(5, len(predictions))):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4))
        
        # Plot shelf layout
        im1 = plot_heatmap(test_features[i, 0], 'Shelf Layout', ax1)
        plt.colorbar(im1, ax=ax1)
        
        # Plot endpoint layout
        im2 = plot_heatmap(test_features[i, 1], 'Endpoint Layout', ax2)
        plt.colorbar(im2, ax=ax2)
        
        # Plot predicted heatmap
        im3 = plot_heatmap(predictions[i, 0], 'Predicted Heatmap', ax3)
        plt.colorbar(im3, ax=ax3)
        
        # Plot ground truth heatmap
        im4 = plot_heatmap(test_labels[i, 0], 'Ground Truth Heatmap', ax4)
        plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        plt.savefig(f'heatmap_comparison_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("\nSaved files:")
print("- validation_loss.png")
print("- heatmap_comparison_[0-4].png")
print("- best_occupancy_grid_resnet_model.pth")


