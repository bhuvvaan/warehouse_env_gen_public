import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds set to {seed} for reproducibility")

# Set seeds at the start
set_seeds()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class RewardConditionedUNet(nn.Module):
    """UNet model with optional reward conditioning via cross-attention (CFG-compatible)."""
    
    def __init__(self, embedding_dim=32):
        super().__init__()
        from diffusers import UNet2DConditionModel

        self.unet = UNet2DConditionModel(
            sample_size=(33, 32),  # Updated for your grid dimensions
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 1024),  # Added extra block for larger model
            down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            cross_attention_dim=embedding_dim,
        )

        self.reward_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.null_condition = nn.Parameter(torch.randn(1, 1, embedding_dim))
    
    def forward(self, x, timesteps, reward=None, condition_dropout=False):
        """
        Args:
            x: Noisy input [B, 1, 33, 32]
            timesteps: [B]
            reward: [B] or None
            condition_dropout: bool indicating whether to drop conditioning
        """
        if not condition_dropout and reward is not None:
            reward_emb = self.reward_embedding(reward.view(-1, 1))  # [B, D]
            reward_emb = reward_emb.unsqueeze(1)                    # [B, 1, D]
        else:
            reward_emb = self.null_condition.expand(x.size(0), -1, -1)  # [B, 1, D]

        return self.unet(x, timesteps, encoder_hidden_states=reward_emb, return_dict=False)[0]

class GridRewardDataset(Dataset):
    """Dataset for grid layouts with rewards."""
    
    def __init__(self, csv_file, normalize_rewards=True):
        """
        Args:
            csv_file: Path to the combined CSV file
            normalize_rewards: Whether to normalize reward values using mean and std
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
        
        # Normalize rewards if requested
        if normalize_rewards:
            self.reward_mean = self.rewards.mean()
            self.reward_std = self.rewards.std()
            self.rewards = (self.rewards - self.reward_mean) / self.reward_std
            print(f"Normalized rewards: mean={self.reward_mean:.2f}, std={self.reward_std:.2f}")
        else:
            self.reward_mean = 0
            self.reward_std = 1
        
        # Normalize grids to [-1, 1] range and add channel dimension
        # Note: grids are already in [-1, 0, 1] format from convert_grid_string_to_tensor
        # self.grids = self.grids * 2 - 1  # This line is incorrect - removing it
        self.grids = self.grids.unsqueeze(1)  # Add channel dimension: (N, 1, 33, 32)
        
        print(f"Dataset loaded: {len(self.grids)} samples")
        print(f"Grid shape: {self.grids.shape}")
        print(f"Reward range: [{self.rewards.min():.3f}, {self.rewards.max():.3f}]")
    
    def __len__(self):
        return len(self.grids)
    
    def __getitem__(self, idx):
        return self.grids[idx], self.rewards[idx]
    
    def denormalize_reward(self, normalized_reward):
        """Convert normalized reward back to original scale."""
        return normalized_reward * self.reward_std + self.reward_mean

def train_reward_conditioned_diffusion(
    dataset_path,
    output_dir="reward_conditioned_diffusion_model_v2",
    batch_size=32,
    num_epochs=10,
    learning_rate=1e-4,
    embedding_dim=32,
    save_interval=5
):
    """Train a reward-conditioned diffusion model."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = GridRewardDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create reward-conditioned UNet with cross-attention
    model = RewardConditionedUNet(embedding_dim=embedding_dim)
    model.to(device)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop
    global_step = 0
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            clean_images, rewards = batch
            clean_images = clean_images.to(device)
            rewards = rewards.to(device)

            # Sample noise and timesteps
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (clean_images.shape[0],), device=device
            ).long()

            # Add noise to the inputs
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Classifier-free guidance trick: randomly drop conditioning
            drop_prob = 0.1
            condition_dropout = torch.rand(1).item() < drop_prob

            # Forward pass with or without conditioning
            noise_pred = model(noisy_images, timesteps, rewards, condition_dropout=condition_dropout)

            # Loss and backprop
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})

        
        # Calculate average epoch loss
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Save model checkpoint
        if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save the model
            model.unet.save_pretrained(checkpoint_dir)
            
            # Save the reward embedding separately
            torch.save(model.reward_embedding.state_dict(), 
                      os.path.join(checkpoint_dir, "reward_embedding.pt"))
            
            # Save dataset normalization parameters
            torch.save({
                "reward_mean": dataset.reward_mean,
                "reward_std": dataset.reward_std
            }, os.path.join(checkpoint_dir, "normalization_params.pt"))
            
            print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()
    
    return model, dataset

def sample_from_model(
    model,
    dataset,
    num_samples=4,
    target_rewards=None,
    guidance_scale=3.0,
    num_inference_steps=50
):
    """Generate samples from the trained model with specified target rewards."""
    model.eval()
    
    # Create scheduler for inference
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps)
    
    # If no target rewards provided, use evenly spaced values
    if target_rewards is None:
        # Generate samples with different reward targets
        # from low to high (e.g., -2 to 2 in normalized space)
        target_rewards = torch.linspace(-2, 2, num_samples, device=device)
    else:
        # Normalize provided rewards
        target_rewards = torch.tensor(
            [(r - dataset.reward_mean) / dataset.reward_std for r in target_rewards],
            device=device
        )
    
    # Start with random noise
    samples = torch.randn(num_samples, 1, 33, 32, device=device)
    
    # Show inference progress with tqdm
    for t in tqdm(scheduler.timesteps, desc="Generating samples"):
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            # Conditional prediction (with reward)
            eps_cond = model(samples, timesteps, target_rewards, condition_dropout=False)

            if guidance_scale > 1.0:
                # Unconditional prediction (no reward input)
                eps_uncond = model(samples, timesteps, reward=None, condition_dropout=True)

                # Apply classifier-free guidance
                model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                model_output = eps_cond

            # Update samples
            samples = scheduler.step(model_output, t, samples).prev_sample

    # Print numerical values before trinarization
    print("\nContinuous output values before trinarization:")
    for i in range(num_samples):
        sample_data = samples[i, 0].cpu().numpy()
        print(f"Sample {i+1}: min={sample_data.min():.4f}, max={sample_data.max():.4f}, mean={sample_data.mean():.4f}, std={sample_data.std():.4f}")
    
    # Trinarize the final samples
    trinary_samples = []
    for i in range(num_samples):
        # Convert back from [-1,1] to [0,1] range for thresholding
        normalized_sample = (samples[i] + 1.0) / 2.0
        
        # Take top 240 values as shelves (1), bottom 350 as endpoints (-1)
        flat_sample = normalized_sample.view(-1)
        _, top_indices = torch.topk(flat_sample, k=240)
        _, bottom_indices = torch.topk(flat_sample, k=350, largest=False)
        
        trinary = torch.zeros_like(flat_sample)
        trinary[top_indices] = 1.0  # Shelves
        trinary[bottom_indices] = -1.0  # Endpoints
        trinary_samples.append(trinary.view(1, 33, 32))
    
    trinary_samples = torch.stack(trinary_samples)
    
    return samples, trinary_samples, target_rewards

def visualize_samples(samples, trinary_samples, target_rewards, dataset):
    """Visualize generated samples with their target rewards."""
    num_samples = samples.shape[0]
    
    # Denormalize rewards for display
    denorm_rewards = [dataset.denormalize_reward(r.item()) for r in target_rewards]
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Plot continuous sample
        im1 = axes[0, i].imshow(samples[i, 0].cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
        axes[0, i].set_title(f"Continuous\nTarget: {denorm_rewards[i]:.2f}")
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Plot trinary sample
        im2 = axes[1, i].imshow(trinary_samples[i, 0].cpu().numpy(), cmap='RdBu', vmin=-1, vmax=1)
        axes[1, i].set_title("Trinarized")
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('diffusion_samples_v2.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Path to the dataset with rewards
    dataset_path = "combined_grids_throughput.csv"
    
    # Train the model
    model, dataset = train_reward_conditioned_diffusion(
        dataset_path=dataset_path,
        output_dir="reward_conditioned_diffusion_model_v2",
        batch_size=32,  # Back to 32 to fix CUDA memory issue
        num_epochs=10,
        learning_rate=1e-4,
        embedding_dim=32,
        save_interval=5
    )
    
    # Generate samples with different target rewards
    # These are original scale rewards (will be normalized internally)
    target_rewards = [3.0, 4.0, 5.0, 7.0]
    
    samples, trinary_samples, norm_rewards = sample_from_model(
        model=model,
        dataset=dataset,
        num_samples=len(target_rewards),
        target_rewards=target_rewards,
        guidance_scale=9.0,  # Tripled from 3.0
        num_inference_steps=50
    )
    
    # Visualize the samples
    visualize_samples(samples, trinary_samples, norm_rewards, dataset)

if __name__ == "__main__":
    main() 