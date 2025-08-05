import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
from pathlib import Path
import pickle

# Global random seed for reproducibility
RANDOM_SEED = 42

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

def convert_grid_string_to_tensor(grid_str, rows=33, cols=36):
    """Convert grid string to tensor, removing first/last 2 columns and converting to -1, 0, 1 format."""
    # Remove first 2 and last 2 columns, so 36 -> 32 columns
    grid_tensor = torch.zeros(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(grid_str):
                char = grid_str[idx]
                if char == '@':  # Shelves
                    grid_tensor[i, j] = 1.0
                elif char == 'e':  # Endpoints
                    grid_tensor[i, j] = -1.0
                elif char == '.' or char == 'w':  # Empty or workstation
                    grid_tensor[i, j] = 0.0
    
    # Remove first 2 and last 2 columns
    grid_tensor = grid_tensor[:, 2:-2]  # Shape: (33, 32)
    
    return grid_tensor

class GridRewardDataset(Dataset):
    """Dataset for grid-reward pairs."""
    
    def __init__(self, csv_file, normalize_rewards=True):
        """
        Args:
            csv_file: Path to the combined CSV file
            normalize_rewards: Whether to normalize reward values to [0, 1]
        """
        self.df = pd.read_csv(csv_file)
        
        # Convert grids to tensors
        self.grids = []
        self.rewards = []
        
        print("Processing grids...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            grid_tensor = convert_grid_string_to_tensor(row['grid'])
            self.grids.append(grid_tensor)
            self.rewards.append(row['throughput'])
        
        self.grids = torch.stack(self.grids)  # Shape: (N, 33, 32)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
        
        # Normalize rewards to [0, 1] range
        if normalize_rewards:
            self.reward_min = self.rewards.min()
            self.reward_max = self.rewards.max()
            self.rewards = (self.rewards - self.reward_min) / (self.reward_max - self.reward_min)
            print(f"Reward range: [{self.reward_min:.3f}, {self.reward_max:.3f}]")
        
        # Normalize grids to [-1, 1] range and add channel dimension
        self.grids = self.grids * 2 - 1  # Convert from [0, 1] to [-1, 1]
        self.grids = self.grids.unsqueeze(1)  # Add channel dimension: (N, 1, 33, 32)
        
        print(f"Dataset loaded: {len(self.grids)} samples")
        print(f"Grid shape: {self.grids.shape}")
        print(f"Reward range: [{self.rewards.min():.3f}, {self.rewards.max():.3f}]")
    
    def __len__(self):
        return len(self.grids)
    
    def __getitem__(self, idx):
        return {
            'grid': self.grids[idx],
            'reward': self.rewards[idx]
        }
    
    def denormalize_reward(self, normalized_reward):
        """Convert normalized reward back to original scale."""
        return normalized_reward * (self.reward_max - self.reward_min) + self.reward_min

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """Basic UNet block."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.up = up
        
        # For upsampling, we need to handle the concatenated skip connection
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x, t, reward_emb=None):
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend time embeddings to 2D
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time embeddings
        h = h + time_emb
        # Add reward embedding if provided (project to correct dimension)
        if reward_emb is not None:
            # Project reward embedding to match feature dimension
            reward_proj = reward_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.shape[2], h.shape[3])
            # Use only the first 'out_ch' dimensions of reward embedding
            if reward_proj.shape[1] > h.shape[1]:
                reward_proj = reward_proj[:, :h.shape[1], :, :]
            elif reward_proj.shape[1] < h.shape[1]:
                # Pad with zeros if needed
                padding = torch.zeros(h.shape[0], h.shape[1] - reward_proj.shape[1], h.shape[2], h.shape[3], 
                                    device=reward_proj.device)
                reward_proj = torch.cat([reward_proj, padding], dim=1)
            h = h + reward_proj
        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or up sample
        return self.transform(h)

class RewardConditionedUNet(nn.Module):
    """UNet model conditioned on reward values."""
    
    def __init__(self, in_channels=1, out_channels=1, time_dim=256, reward_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Reward embeddings
        self.reward_embedding = nn.Sequential(
            nn.Linear(1, reward_dim),
            nn.ReLU(),
            nn.Linear(reward_dim, reward_dim),
            nn.ReLU(),
            nn.Linear(reward_dim, reward_dim)
        )
        
        # Null condition for classifier-free guidance
        self.null_condition = nn.Parameter(torch.randn(reward_dim))
        
        # Initial convolution
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Downsampling
        self.down1 = Block(64, 128, time_dim)
        self.down2 = Block(128, 256, time_dim)
        self.down3 = Block(256, 512, time_dim)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Upsampling with proper skip connections and size maintenance
        self.up1 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Final convolution
        self.output = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, timestep, reward=None, use_null_condition=False):
        # Time embeddings
        t = self.time_mlp(timestep)
        
        # Reward embeddings
        if use_null_condition:
            reward_emb = self.null_condition.unsqueeze(0).expand(x.shape[0], -1)
        elif reward is not None:
            reward_emb = self.reward_embedding(reward.unsqueeze(-1))
        else:
            reward_emb = None
        
        # Initial convolution
        x0 = self.conv0(x)
        
        # Downsampling with skip connections
        x1 = self.down1(x0, t, reward_emb)
        x2 = self.down2(x1, t, reward_emb)
        x3 = self.down3(x2, t, reward_emb)
        
        # Bottleneck
        x4 = self.bottleneck(x3)
        
        # Upsampling with skip connections
        x = self.up1(torch.cat([x4, x3], dim=1))
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up3(torch.cat([x, x1], dim=1))
        
        # Ensure output size matches input
        if x.shape[2:] != x0.shape[2:]:
            x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        
        # Output
        return self.output(x)

class DiffusionModel:
    """Diffusion model for grid generation."""
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Ensure t is on the correct device
        t = t.to(self.device)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def p_losses(self, denoise_model, x_start, t, reward=None, noise=None):
        """Compute loss for training."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy, predicted_noise = self.q_sample(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t, reward)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

def train_reward_conditioned_diffusion(
    model,
    diffusion,
    dataloader,
    optimizer,
    device,
    epochs=100,
    save_every=10,
    guidance_scale=7.5
):
    """Train the reward-conditioned diffusion model."""
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            grids = batch['grid'].to(device)
            rewards = batch['reward'].to(device)
            
            batch_size = grids.shape[0]
            
            # Random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Randomly drop reward conditioning (10% chance) for classifier-free guidance
            drop_conditioning = torch.rand(batch_size, device=device) < 0.1
            rewards_masked = rewards.clone()
            rewards_masked[drop_conditioning] = 0.0  # Use null condition
            
            # Compute loss
            loss = diffusion.p_losses(model, grids, t, rewards_masked)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses
            }
            torch.save(checkpoint, f'diffusion_checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved: diffusion_checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), 'diffusion_model_final.pth')
    print("Final model saved: diffusion_model_final.pth")
    
    return losses

@torch.no_grad()
def sample_from_model(model, diffusion, target_rewards, device, guidance_scale=7.5, num_samples=1):
    """Generate samples conditioned on target rewards."""
    
    model.eval()
    samples = []
    
    for reward in target_rewards:
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
        
        # Start from random noise
        x = torch.randn(num_samples, 1, 33, 32, device=device)
        
        # Reverse diffusion process
        for i in tqdm(reversed(range(0, diffusion.timesteps)), desc=f"Sampling for reward {reward:.2f}"):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Predict noise with and without conditioning
            predicted_noise_cond = model(x, t, reward_tensor)
            predicted_noise_uncond = model(x, t, use_null_condition=True)
            
            # Classifier-free guidance
            predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)
            
            # Denoise step
            alpha = diffusion.alphas[i]
            alpha_prev = diffusion.alphas_cumprod_prev[i]
            beta = diffusion.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - diffusion.alphas_cumprod[i])) * predicted_noise) + torch.sqrt(beta) * noise
        
        samples.append(x.cpu())
    
    return torch.cat(samples, dim=0)

def visualize_samples(samples, target_rewards, dataset, save_path='diffusion_samples.png'):
    """Visualize generated samples."""
    
    num_samples = len(samples)
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, (sample, reward) in enumerate(zip(samples, target_rewards)):
        # Denormalize reward
        denorm_reward = dataset.denormalize_reward(reward)
        
        # Continuous sample
        axes[0, i].imshow(sample.squeeze(), cmap='RdBu', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Continuous\nReward: {denorm_reward:.2f}')
        axes[0, i].axis('off')
        
        # Binarized sample (top 240 values as shelves, bottom 350 as endpoints)
        sample_flat = sample.flatten()
        sorted_indices = torch.argsort(sample_flat, descending=True)
        
        binarized = torch.zeros_like(sample)
        binarized_flat = binarized.flatten()
        
        # Top 240 as shelves (1)
        binarized_flat[sorted_indices[:240]] = 1.0
        # Next 350 as endpoints (-1)
        binarized_flat[sorted_indices[240:590]] = -1.0
        # Rest as empty (0)
        
        axes[1, i].imshow(binarized.squeeze(), cmap='RdBu', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Binarized\nReward: {denorm_reward:.2f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Samples saved to {save_path}")

def main():
    """Main training and sampling function."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = GridRewardDataset('combined_grids_throughput.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize model
    print("Initializing model...")
    model = RewardConditionedUNet().to(device)
    
    # Initialize diffusion
    diffusion = DiffusionModel(device=device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training
    print("Starting training...")
    losses = train_reward_conditioned_diffusion(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        epochs=10,# Reduced for faster training
        save_every=10
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('diffusion_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sampling
    print("Generating samples...")
    target_rewards = [0.2, 0.4, 0.6, 0.8]  # Normalized rewards
    samples = sample_from_model(model, diffusion, target_rewards, device, guidance_scale=7.5)
    
    # Visualize samples
    visualize_samples(samples, target_rewards, dataset)
    
    # Save diffusion model and parameters
    diffusion_state = {
        'model_state_dict': model.state_dict(),
        'diffusion_params': {
            'timesteps': diffusion.timesteps,
            'betas': diffusion.betas,
            'alphas': diffusion.alphas,
            'alphas_cumprod': diffusion.alphas_cumprod,
            'alphas_cumprod_prev': diffusion.alphas_cumprod_prev,
            'sqrt_alphas_cumprod': diffusion.sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': diffusion.sqrt_one_minus_alphas_cumprod,
            'posterior_variance': diffusion.posterior_variance
        },
        'dataset_params': {
            'reward_min': dataset.reward_min,
            'reward_max': dataset.reward_max
        }
    }
    
    torch.save(diffusion_state, 'diffusion_model_complete.pth')
    print("Complete diffusion model saved: diffusion_model_complete.pth")

if __name__ == "__main__":
    main() 