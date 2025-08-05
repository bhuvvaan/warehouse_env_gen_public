import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Import the model classes from the training script
from train_diffusion_model import (
    SinusoidalPositionEmbeddings, 
    Block, 
    RewardConditionedUNet, 
    DiffusionModel,
    convert_grid_string_to_tensor
)

class DiffusionInference:
    """Class for inference with the trained diffusion model."""
    
    def __init__(self, model_path='diffusion_model_complete.pth'):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the saved diffusion model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the saved model
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained diffusion model."""
        print(f"Loading model from {model_path}...")
        
        # Load the complete state
        state = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        self.model = RewardConditionedUNet().to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
        
        # Initialize diffusion with saved parameters
        diffusion_params = state['diffusion_params']
        self.diffusion = DiffusionModel()
        self.diffusion.timesteps = diffusion_params['timesteps']
        self.diffusion.betas = diffusion_params['betas'].to(self.device)
        self.diffusion.alphas = diffusion_params['alphas'].to(self.device)
        self.diffusion.alphas_cumprod = diffusion_params['alphas_cumprod'].to(self.device)
        self.diffusion.alphas_cumprod_prev = diffusion_params['alphas_cumprod_prev'].to(self.device)
        self.diffusion.sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod'].to(self.device)
        self.diffusion.sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod'].to(self.device)
        self.diffusion.posterior_variance = diffusion_params['posterior_variance'].to(self.device)
        
        # Load dataset parameters for denormalization
        self.reward_min = state['dataset_params']['reward_min']
        self.reward_max = state['dataset_params']['reward_max']
        
        print("Model loaded successfully!")
        print(f"Reward range: [{self.reward_min:.3f}, {self.reward_max:.3f}]")
    
    def normalize_reward(self, reward):
        """Normalize reward to [0, 1] range."""
        return (reward - self.reward_min) / (self.reward_max - self.reward_min)
    
    def denormalize_reward(self, normalized_reward):
        """Convert normalized reward back to original scale."""
        return normalized_reward * (self.reward_max - self.reward_min) + self.reward_min
    
    @torch.no_grad()
    def generate_grid(self, target_throughput, guidance_scale=7.5, num_samples=1):
        """
        Generate grid layout conditioned on target throughput.
        
        Args:
            target_throughput: Target throughput value (original scale)
            guidance_scale: Classifier-free guidance scale
            num_samples: Number of samples to generate
        
        Returns:
            Generated grid tensors
        """
        # Normalize the target throughput
        normalized_reward = self.normalize_reward(target_throughput)
        reward_tensor = torch.tensor([normalized_reward], device=self.device, dtype=torch.float32)
        
        print(f"Generating grid for throughput: {target_throughput:.3f} (normalized: {normalized_reward:.3f})")
        
        # Start from random noise
        x = torch.randn(num_samples, 1, 33, 32, device=self.device)
        
        # Reverse diffusion process
        for i in tqdm(reversed(range(0, self.diffusion.timesteps)), desc="Generating..."):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            
            # Predict noise with and without conditioning
            predicted_noise_cond = self.model(x, t, reward_tensor)
            predicted_noise_uncond = self.model(x, t, use_null_condition=True)
            
            # Classifier-free guidance
            predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)
            
            # Denoise step
            alpha = self.diffusion.alphas[i]
            alpha_prev = self.diffusion.alphas_cumprod_prev[i]
            beta = self.diffusion.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - self.diffusion.alphas_cumprod[i])) * predicted_noise) + torch.sqrt(beta) * noise
        
        return x.cpu()
    
    def binarize_grid(self, continuous_grid):
        """
        Convert continuous grid to binary grid with proper shelf/endpoint counts.
        
        Args:
            continuous_grid: Continuous grid tensor of shape (1, 33, 32)
        
        Returns:
            Binary grid tensor with shelves=1, endpoints=-1, empty=0
        """
        sample_flat = continuous_grid.flatten()
        sorted_indices = torch.argsort(sample_flat, descending=True)
        
        binarized = torch.zeros_like(continuous_grid)
        binarized_flat = binarized.flatten()
        
        # Top 240 values as shelves (1)
        binarized_flat[sorted_indices[:240]] = 1.0
        # Next 350 values as endpoints (-1) - adjust this number based on your data
        binarized_flat[sorted_indices[len(binarized_flat)-350:len(binarized_flat)]] = -1.0
        # Rest as empty (0)
        
        return binarized
    
    def grid_to_string(self, grid_tensor):
        """
        Convert grid tensor back to string format.
        
        Args:
            grid_tensor: Grid tensor of shape (33, 32)
        
        Returns:
            Grid string with proper padding (first 2 and last 2 columns added back)
        """
        # Add back the first 2 and last 2 columns
        full_grid = torch.zeros(33, 36)
        full_grid[:, 2:-2] = grid_tensor
        
        # Convert to string
        grid_str = ""
        for i in range(33):
            for j in range(36):
                val = full_grid[i, j].item()
                if val == 1.0:  # Shelves
                    grid_str += '@'
                elif val == -1.0:  # Endpoints
                    grid_str += 'e'
                else:  # Empty or workstation
                    grid_str += '.'
        
        return grid_str
    
    def visualize_grid(self, grid_tensor, target_throughput, save_path=None):
        """
        Visualize the generated grid.
        
        Args:
            grid_tensor: Grid tensor of shape (1, 33, 32)
            target_throughput: Target throughput value
            save_path: Path to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Continuous grid
        im1 = ax1.imshow(grid_tensor.squeeze(), cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title(f'Continuous Grid\nTarget Throughput: {target_throughput:.3f}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Binarized grid
        binarized = self.binarize_grid(grid_tensor)
        im2 = ax2.imshow(binarized.squeeze(), cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title(f'Binarized Grid\nShelves: 240, Endpoints: 350')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_multiple_grids(self, target_throughputs, guidance_scale=7.5, save_dir='generated_grids'):
        """
        Generate multiple grids for different target throughputs.
        
        Args:
            target_throughputs: List of target throughput values
            guidance_scale: Classifier-free guidance scale
            save_dir: Directory to save generated grids
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        results = []
        
        for i, throughput in enumerate(target_throughputs):
            print(f"\nGenerating grid {i+1}/{len(target_throughputs)} for throughput: {throughput:.3f}")
            
            # Generate grid
            grid_tensor = self.generate_grid(throughput, guidance_scale)
            
            # Binarize grid
            binarized_grid = self.binarize_grid(grid_tensor)
            
            # Convert to string
            grid_string = self.grid_to_string(binarized_grid.squeeze())
            
            # Save visualization
            viz_path = os.path.join(save_dir, f'grid_throughput_{throughput:.2f}.png')
            self.visualize_grid(grid_tensor, throughput, viz_path)
            
            # Save grid data
            grid_data = {
                'grid_number': i + 1,
                'target_throughput': throughput,
                'grid_string': grid_string,
                'num_shelves': 240,
                'num_endpoints': 350
            }
            
            results.append(grid_data)
            
            # Save individual grid data
            grid_file = os.path.join(save_dir, f'grid_throughput_{throughput:.2f}.json')
            with open(grid_file, 'w') as f:
                json.dump(grid_data, f, indent=2)
        
        # Save all results
        all_results_file = os.path.join(save_dir, 'all_generated_grids.json')
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAll grids saved to {save_dir}")
        return results

def main():
    """Example usage of the diffusion inference."""
    
    # Initialize inference
    inference = DiffusionInference()
    
    # Example target throughputs (in original scale)
    target_throughputs = [3.0, 4.0, 5.0, 7.0]
    
    # Generate grids
    results = inference.generate_multiple_grids(target_throughputs, guidance_scale=7.5)
    
    # Print summary
    print("\nGenerated Grids Summary:")
    for result in results:
        print(f"Grid {result['grid_number']}: Throughput {result['target_throughput']:.2f}")
        print(f"  Grid string length: {len(result['grid_string'])}")
        print(f"  Shelves: {result['num_shelves']}, Endpoints: {result['num_endpoints']}")
        print()

if __name__ == "__main__":
    main() 