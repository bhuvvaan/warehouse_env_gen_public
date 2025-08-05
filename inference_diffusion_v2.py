import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
            block_out_channels=(128, 256, 512),  # Added more channels for larger input
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
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

class DiffusionInferenceV2:
    """Class for inference with the trained diffusion model v2."""
    
    def __init__(self, checkpoint_dir="reward_conditioned_diffusion_model_v2/checkpoint-5"):
        """
        Initialize the inference class.
        
        Args:
            checkpoint_dir: Path to the saved diffusion model checkpoint
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load the saved model
        self.load_model(checkpoint_dir)
    
    def load_model(self, checkpoint_dir):
        """Load the trained diffusion model."""
        print(f"Loading model from {checkpoint_dir}...")
        
        # Initialize model
        self.model = RewardConditionedUNet(embedding_dim=32).to(self.device)
        
        # Load UNet weights
        self.model.unet = UNet2DConditionModel.from_pretrained(checkpoint_dir)
        self.model.unet.to(self.device)
        
        # Load reward embedding
        reward_embedding_path = os.path.join(checkpoint_dir, "reward_embedding.pt")
        if os.path.exists(reward_embedding_path):
            self.model.reward_embedding.load_state_dict(torch.load(reward_embedding_path, map_location=self.device))
        
        # Load normalization parameters
        norm_params_path = os.path.join(checkpoint_dir, "normalization_params.pt")
        if os.path.exists(norm_params_path):
            norm_params = torch.load(norm_params_path, map_location=self.device)
            self.reward_mean = norm_params["reward_mean"]
            self.reward_std = norm_params["reward_std"]
        else:
            # Default values if not found
            self.reward_mean = 4.98
            self.reward_std = 1.12
        
        self.model.eval()
        
        print("Model loaded successfully!")
        print(f"Reward range: mean={self.reward_mean:.3f}, std={self.reward_std:.3f}")
    
    def normalize_reward(self, reward):
        """Normalize reward to standardized scale."""
        return (reward - self.reward_mean) / self.reward_std
    
    def denormalize_reward(self, normalized_reward):
        """Convert normalized reward back to original scale."""
        return normalized_reward * self.reward_std + self.reward_mean
    
    @torch.no_grad()
    def generate_grid(self, target_throughput, guidance_scale=3.0, num_samples=1, num_inference_steps=50):
        """
        Generate grid layout conditioned on target throughput.
        
        Args:
            target_throughput: Target throughput value (original scale)
            guidance_scale: Classifier-free guidance scale
            num_samples: Number of samples to generate
            num_inference_steps: Number of denoising steps
        
        Returns:
            Generated continuous grid tensors
        """
        # Normalize the target throughput
        normalized_reward = self.normalize_reward(target_throughput)
        reward_tensor = torch.tensor([normalized_reward], device=self.device, dtype=torch.float32)
        
        print(f"Generating grid for throughput: {target_throughput:.3f} (normalized: {normalized_reward:.3f})")
        
        # Create scheduler for inference
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps)
        
        # Start from random noise
        samples = torch.randn(num_samples, 1, 33, 32, device=self.device)
        
        # Reverse diffusion process with tqdm progress bar
        for t in tqdm(scheduler.timesteps, desc="Generating samples"):
            timesteps = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            # Conditional prediction (with reward)
            eps_cond = self.model(samples, timesteps, reward_tensor, condition_dropout=False)

            if guidance_scale > 1.0:
                # Unconditional prediction (no reward input)
                eps_uncond = self.model(samples, timesteps, reward=None, condition_dropout=True)

                # Apply classifier-free guidance
                model_output = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                model_output = eps_cond

            # Update samples
            samples = scheduler.step(model_output, t, samples).prev_sample
        
        return samples.cpu()
    
    def print_all_continuous_values(self, continuous_grid, sample_name="Grid"):
        """
        Print all continuous values in the grid.
        
        Args:
            continuous_grid: Continuous grid tensor of shape (1, 33, 32)
            sample_name: Name of the sample for identification
        """
        grid_data = continuous_grid.squeeze().numpy()
        
        print(f"\n{sample_name} - All Continuous Values (33x32):")
        print("=" * 80)
        
        # Print the full grid with row and column indices
        print("     ", end="")
        for j in range(32):
            print(f"{j:6d}", end="")
        print()
        print("-" * 200)
        
        for i in range(33):
            print(f"{i:2d} |", end="")
            for j in range(32):
                print(f"{grid_data[i, j]:6.3f}", end="")
            print()
        
        print("=" * 80)
        print(f"Grid shape: {grid_data.shape}")
        print(f"Total values: {grid_data.size}")
    
    def analyze_continuous_output(self, continuous_grid):
        """
        Analyze the continuous output values.
        
        Args:
            continuous_grid: Continuous grid tensor of shape (1, 33, 32)
        
        Returns:
            Dictionary with analysis results
        """
        grid_data = continuous_grid.squeeze().numpy()
        
        analysis = {
            'min': grid_data.min(),
            'max': grid_data.max(),
            'mean': grid_data.mean(),
            'std': grid_data.std(),
            'shape': grid_data.shape,
            'total_pixels': grid_data.size,
            'positive_pixels': np.sum(grid_data > 0),
            'negative_pixels': np.sum(grid_data < 0),
            'zero_pixels': np.sum(grid_data == 0),
            'positive_ratio': np.sum(grid_data > 0) / grid_data.size,
            'negative_ratio': np.sum(grid_data < 0) / grid_data.size,
            'zero_ratio': np.sum(grid_data == 0) / grid_data.size
        }
        
        return analysis
    
    def analyze_continuous_distribution(self, continuous_grid):
        """
        Analyze the distribution of continuous values to count shelves below -0.9 and above 0.9.
        
        Args:
            continuous_grid: Continuous grid tensor of shape (1, 33, 32)
        
        Returns:
            Dictionary with distribution analysis
        """
        grid_data = continuous_grid.squeeze().numpy()
        
        # Count values in different ranges
        below_minus_09 = np.sum(grid_data < -0.9)
        above_09 = np.sum(grid_data > 0.9)
        between_minus_09_and_09 = np.sum((grid_data >= -0.9) & (grid_data <= 0.9))
        
        # Calculate percentages
        total_values = grid_data.size
        below_minus_09_pct = (below_minus_09 / total_values) * 100
        above_09_pct = (above_09 / total_values) * 100
        between_pct = (between_minus_09_and_09 / total_values) * 100
        
        analysis = {
            'below_minus_09': below_minus_09,
            'above_09': above_09,
            'between_minus_09_and_09': between_minus_09_and_09,
            'total_values': total_values,
            'below_minus_09_pct': below_minus_09_pct,
            'above_09_pct': above_09_pct,
            'between_pct': between_pct,
            'min_value': grid_data.min(),
            'max_value': grid_data.max(),
            'mean_value': grid_data.mean(),
            'std_value': grid_data.std()
        }
        
        return analysis
    
    def print_continuous_distribution(self, analysis):
        """Print the continuous distribution analysis."""
        print(f"\nContinuous Distribution Analysis (Before Post-Processing):")
        print(f"=" * 60)
        print(f"Values below -0.9: {analysis['below_minus_09']} ({analysis['below_minus_09_pct']:.2f}%)")
        print(f"Values above 0.9:  {analysis['above_09']} ({analysis['above_09_pct']:.2f}%)")
        print(f"Values between -0.9 and 0.9: {analysis['between_minus_09_and_09']} ({analysis['between_pct']:.2f}%)")
        print(f"Total values: {analysis['total_values']}")
        print(f"Range: [{analysis['min_value']:.4f}, {analysis['max_value']:.4f}]")
        print(f"Mean: {analysis['mean_value']:.4f}, Std: {analysis['std_value']:.4f}")
        print(f"=" * 60)
    
    def trinarize_grid(self, continuous_grid):
        """
        Convert continuous grid to trinary grid with fixed thresholds at 0.9 and -0.9.
        
        Args:
            continuous_grid: Continuous grid tensor of shape (1, 33, 32)
        
        Returns:
            Trinary grid tensor with shelves=1, endpoints=-1, empty=0
        """
        grid_data = continuous_grid.squeeze().numpy()
        
        # Use fixed thresholds: values > 0.9 become shelves (1), values < -0.9 become endpoints (-1)
        trinary = np.zeros_like(grid_data)
        trinary[grid_data > 0.9] = 1.0   # Shelves
        trinary[grid_data < -0.9] = -1.0  # Endpoints
        # Values between -0.9 and 0.9 remain 0 (empty/workstation)
        
        trinary = torch.from_numpy(trinary).unsqueeze(0)  # Add batch dimension
        
        return trinary
    
    def visualize_grid(self, continuous_grid, trinary_grid, target_throughput, save_path=None):
        """
        Visualize the generated grid.
        
        Args:
            continuous_grid: Continuous grid tensor of shape (1, 33, 32)
            trinary_grid: Trinary grid tensor of shape (1, 33, 32)
            target_throughput: Target throughput value
            save_path: Path to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Continuous grid
        im1 = ax1.imshow(continuous_grid.squeeze(), cmap='RdBu', vmin=-1, vmax=1)
        ax1.set_title(f'Continuous Grid\nTarget Throughput: {target_throughput:.3f}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Trinary grid
        im2 = ax2.imshow(trinary_grid.squeeze(), cmap='RdBu', vmin=-1, vmax=1)
        ax2.set_title(f'Trinary Grid\nShelves: {np.sum(trinary_grid.squeeze().numpy() == 1.0)}, Endpoints: {np.sum(trinary_grid.squeeze().numpy() == -1.0)}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_multiple_grids(self, target_throughputs, guidance_scale=3.0, save_dir='generated_grids_v2'):
        """
        Generate multiple grids for different target throughputs.
        
        Args:
            target_throughputs: List of target throughput values
            guidance_scale: Classifier-free guidance scale
            save_dir: Directory to save generated grids
        """
        os.makedirs(save_dir, exist_ok=True)
        
        results = []
        
        for i, throughput in enumerate(target_throughputs):
            print(f"\n{'='*60}")
            print(f"Generating grid {i+1}/{len(target_throughputs)} for throughput: {throughput:.3f}")
            print(f"{'='*60}")
            
            # Generate continuous grid
            continuous_grid = self.generate_grid(throughput, guidance_scale, num_samples=1)
            
            # Print all continuous values
            self.print_all_continuous_values(continuous_grid, f"Grid {i+1} (Throughput: {throughput:.3f})")
            
            # Analyze continuous output
            analysis = self.analyze_continuous_output(continuous_grid)
            
            print(f"\nContinuous Output Analysis:")
            print(f"  Min value: {analysis['min']:.4f}")
            print(f"  Max value: {analysis['max']:.4f}")
            print(f"  Mean value: {analysis['mean']:.4f}")
            print(f"  Std value: {analysis['std']:.4f}")
            print(f"  Positive pixels: {analysis['positive_pixels']} ({analysis['positive_ratio']:.2%})")
            print(f"  Negative pixels: {analysis['negative_pixels']} ({analysis['negative_ratio']:.2%})")
            print(f"  Zero pixels: {analysis['zero_pixels']} ({analysis['zero_ratio']:.2%})")
            
            # Analyze continuous distribution
            distribution_analysis = self.analyze_continuous_distribution(continuous_grid)
            self.print_continuous_distribution(distribution_analysis)

            # Trinarize grid
            trinary_grid = self.trinarize_grid(continuous_grid)
            
            # Count the actual number of endpoints based on threshold
            trinary_data = trinary_grid.squeeze().numpy()
            num_endpoints = np.sum(trinary_data == -1.0)
            num_shelves = np.sum(trinary_data == 1.0)
            num_empty = np.sum(trinary_data == 0.0)
            
            print(f"\nTrinarization Results:")
            print(f"  Shelves (1.0): {num_shelves}")
            print(f"  Endpoints (-1.0): {num_endpoints} (threshold: < -0.9)")
            print(f"  Empty (0.0): {num_empty}")
            
            # Validate grid connectivity
            validation_results = self.validate_grid_connectivity(trinary_grid)
            self.print_validation_results(validation_results)
            
            # Print original trinarized grid in character format
            self.print_grid_characters(trinary_grid, f"Original Grid {i+1} (Throughput: {throughput:.3f})")
            
            # Post-process the grid to fix connectivity issues
            processed_grid, post_processing_results = self.post_process_grid(trinary_grid)
            self.print_post_processing_results(post_processing_results)
            
            # Print post-processed grid in character format
            self.print_grid_characters(processed_grid, f"Post-Processed Grid {i+1} (Throughput: {throughput:.3f})", expanded=True)
            
            # Save visualization
            viz_path = os.path.join(save_dir, f'grid_throughput_{throughput:.2f}.png')
            self.visualize_grid(continuous_grid, processed_grid, throughput, viz_path)
            
            # Save grid data
            grid_data = {
                'grid_number': i + 1,
                'target_throughput': throughput,
                'continuous_analysis': analysis,
                'num_shelves': int(num_shelves),
                'num_endpoints': int(num_endpoints),
                'num_empty': int(num_empty),
                'endpoint_threshold': -0.9,
                'validation': validation_results,
                'post_processing': post_processing_results
            }
            
            results.append(grid_data)
            
            # Save individual grid data
            grid_file = os.path.join(save_dir, f'grid_throughput_{throughput:.2f}.json')
            with open(grid_file, 'w') as f:
                json.dump(grid_data, f, indent=2, default=str)
        
        # Save all results
        all_results_file = os.path.join(save_dir, 'all_generated_grids_v2.json')
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nAll grids saved to {save_dir}")
        return results

    def validate_grid_connectivity(self, trinary_grid):
        """
        Validate grid connectivity using DFS logic from map_generation/dfs.py.
        
        Args:
            trinary_grid: Trinary grid tensor of shape (1, 33, 32)
        
        Returns:
            Dictionary with validation results
        """
        grid_data = trinary_grid.squeeze().numpy()
        rows, cols = grid_data.shape
        
        # Convert to character representation for easier processing
        char_grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                if grid_data[i, j] == 1.0:  # Shelves (black)
                    row.append('@')
                elif grid_data[i, j] == -1.0:  # Endpoints (blue)
                    row.append('e')
                else:  # Empty/workstation
                    row.append('.')
            char_grid.append(row)
        
        # Check if all endpoints are connected (adapted from validate_blue_connectivity)
        def validate_blue_connectivity(grid):
            """Check that all blue tiles ('e') are connected, treating black tiles ('@') as barriers."""
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            
            def dfs(i, j):
                if i < 0 or i >= rows or j < 0 or j >= cols:
                    return
                if grid[i][j] == '@' or visited[i][j]:
                    return
                visited[i][j] = True
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs(i + dx, j + dy)
            
            # Find the first blue tile to start DFS
            start_found = False
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] == 'e':
                        dfs(i, j)
                        start_found = True
                        break
                if start_found:
                    break

            # If no blue tile is found, connectivity fails
            if not start_found:
                return False

            # Ensure every blue tile was visited by DFS
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] == 'e' and not visited[i][j]:
                        return False
            return True
        
        # Check endpoint connectivity
        endpoints_connected = validate_blue_connectivity(char_grid)
        
        # Check each shelf has at least 2 adjacent endpoints (adapted from place_blue_tiles_adjacent_to_black)
        def count_adjacent_endpoints(shelf_pos):
            """Count how many endpoints are adjacent to a shelf."""
            i, j = shelf_pos
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < rows and 0 <= nj < cols and 
                    char_grid[ni][nj] == 'e'):
                    count += 1
            return count
        
        # Check each shelf has at least 2 adjacent endpoints
        shelves_valid = True
        invalid_shelves = []
        total_shelves = 0
        
        for i in range(rows):
            for j in range(cols):
                if char_grid[i][j] == '@':
                    total_shelves += 1
                    adjacent_endpoints = count_adjacent_endpoints((i, j))
                    if adjacent_endpoints < 2:
                        shelves_valid = False
                        invalid_shelves.append((i, j, adjacent_endpoints))
        
        # Overall validation
        overall_valid = endpoints_connected and shelves_valid
        
        validation_results = {
            'overall_valid': overall_valid,
            'endpoints_connected': endpoints_connected,
            'shelves_valid': shelves_valid,
            'total_shelves': total_shelves,
            'invalid_shelves': invalid_shelves
        }
        
        return validation_results
    
    def print_validation_results(self, validation_results):
        """Print validation results in a readable format."""
        print(f"\nGrid Connectivity Validation:")
        print(f"=" * 50)
        print(f"Overall Valid: {'✅ YES' if validation_results['overall_valid'] else '❌ NO'}")
        print(f"Endpoints Connected: {'✅ YES' if validation_results['endpoints_connected'] else '❌ NO'}")
        print(f"All Shelves Valid: {'✅ YES' if validation_results['shelves_valid'] else '❌ NO'}")
        print(f"Total Shelves: {validation_results['total_shelves']}")
        
        if validation_results['invalid_shelves']:
            print(f"❌ Invalid Shelves ({len(validation_results['invalid_shelves'])}):")
            for i, (row, col, adjacent) in enumerate(validation_results['invalid_shelves'][:10]):  # Show first 10
                print(f"  Shelf {i+1}: Position ({row}, {col}), Adjacent endpoints: {adjacent}")
            if len(validation_results['invalid_shelves']) > 10:
                print(f"  ... and {len(validation_results['invalid_shelves']) - 10} more")
        else:
            print(f"✅ All shelves have at least 2 adjacent endpoints")
        
        print(f"=" * 50)

    def print_grid_characters(self, trinary_grid, title="Grid", expanded=False):
        """
        Print the grid in character format (@, ., e, w).
        
        Args:
            trinary_grid: Trinary grid tensor of shape (1, 33, 32)
            title: Title for the grid display
            expanded: Whether this is an expanded grid with workstations
        """
        if expanded:
            # For expanded grid, we need to reconstruct it from the post-processing
            grid_data = trinary_grid.squeeze().numpy()
            rows, cols = grid_data.shape
            
            # Convert to character representation
            char_grid = []
            for i in range(rows):
                row = []
                for j in range(cols):
                    if grid_data[i, j] == 1.0:  # Shelves (black)
                        row.append('@')
                    elif grid_data[i, j] == -1.0:  # Endpoints (blue)
                        row.append('e')
                    else:  # Empty/workstation
                        row.append('.')
                char_grid.append(row)
            
            # Add 4 columns as specified
            for i in range(rows):
                # Add left blank column
                char_grid[i].insert(0, '.')
                # Add left w column (where row index % 3 == 1)
                if i % 3 == 1:
                    char_grid[i].insert(0, 'w')
                else:
                    char_grid[i].insert(0, '.')
                # Add right blank column
                char_grid[i].append('.')
                # Add right w column (where row index % 3 == 1)
                if i % 3 == 1:
                    char_grid[i].append('w')
                else:
                    char_grid[i].append('.')
            
            new_rows, new_cols = len(char_grid), len(char_grid[0])
        else:
            # Original grid format
            grid_data = trinary_grid.squeeze().numpy()
            char_grid = []
            for i in range(grid_data.shape[0]):
                row = []
                for j in range(grid_data.shape[1]):
                    if grid_data[i, j] == 1.0:  # Shelves (black)
                        row.append('@')
                    elif grid_data[i, j] == -1.0:  # Endpoints (blue)
                        row.append('e')
                    else:  # Empty/workstation
                        row.append('.')
                char_grid.append(row)
            new_rows, new_cols = len(char_grid), len(char_grid[0])
        
        print(f"\n{title} - Character Format ({new_rows}x{new_cols}):")
        print("=" * 100)
        
        # Print column headers
        print("     ", end="")
        for j in range(new_cols):
            print(f"{j:2d}", end="")
        print()
        print("-" * 100)
        
        # Print grid with row numbers
        for i in range(new_rows):
            print(f"{i:2d} |", end="")
            for j in range(new_cols):
                print(f" {char_grid[i][j]}", end="")
            print()
        
        print("=" * 100)
        
        # Count elements
        num_shelves = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == '@')
        num_endpoints = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e')
        num_empty = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == '.')
        num_workstations = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'w')
        
        print(f"Summary: Shelves (@): {num_shelves}, Endpoints (e): {num_endpoints}, Empty (.): {num_empty}, Workstations (w): {num_workstations}")
        print("=" * 100)

    def post_process_grid(self, trinary_grid):
        """
        Post-process the grid to fix connectivity issues using fixed thresholds.
        
        Args:
            trinary_grid: Trinary grid tensor of shape (1, 33, 32)
        
        Returns:
            Post-processed trinary grid tensor
        """
        grid_data = trinary_grid.squeeze().numpy()
        rows, cols = grid_data.shape
        
        # Convert to character representation for easier processing
        char_grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                if grid_data[i, j] == 1.0:  # Shelves (black)
                    row.append('@')
                elif grid_data[i, j] == -1.0:  # Endpoints (blue)
                    row.append('e')
                else:  # Empty/workstation
                    row.append('.')
            char_grid.append(row)
        
        print("\nPost-processing grid with fixed thresholds (0.9, -0.9)...")
        
        # Step 1: Randomly add missing shelves (target: 240 shelves)
        print("Step 1: Randomly adding missing shelves...")
        current_shelves = sum(1 for i in range(rows) for j in range(cols) if char_grid[i][j] == '@')
        target_shelves = 240
        missing_shelves = max(0, target_shelves - current_shelves)
        
        if missing_shelves > 0:
            # Find all empty positions
            empty_positions = []
            for i in range(rows):
                for j in range(cols):
                    if char_grid[i][j] == '.':
                        empty_positions.append((i, j))
            
            # Randomly select positions to add shelves
            import random
            random.seed(42)  # For reproducibility
            selected_positions = random.sample(empty_positions, min(missing_shelves, len(empty_positions)))
            
            for i, j in selected_positions:
                char_grid[i][j] = '@'
            
            print(f"  Added {len(selected_positions)} shelves")
            print(f"  Total shelves now: {current_shelves + len(selected_positions)}")
        else:
            print(f"  No missing shelves to add (current: {current_shelves})")
        
        # Step 2: Check and add blue tiles if <2 anywhere
        print("Step 2: Checking and adding blue tiles if <2 anywhere...")
        
        def count_adjacent_endpoints(shelf_pos):
            """Count how many endpoints are adjacent to a shelf."""
            i, j = shelf_pos
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < rows and 0 <= nj < cols and 
                    char_grid[ni][nj] == 'e'):
                    count += 1
            return count
        
        def find_empty_adjacent_positions(pos):
            """Find empty positions adjacent to a given position."""
            i, j = pos
            empty_positions = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < rows and 0 <= nj < cols and 
                    char_grid[ni][nj] == '.'):
                    empty_positions.append((ni, nj))
            return empty_positions
        
        def find_connected_empty_adjacent_positions(pos, visited):
            """Find empty positions adjacent to a given position that are connected to the main component."""
            i, j = pos
            connected_empty_positions = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < new_rows and 0 <= nj < new_cols and 
                    char_grid[ni][nj] == '.' and visited[ni][nj]):
                    connected_empty_positions.append((ni, nj))
            return connected_empty_positions
        
        shelves_needing_endpoints = []
        added_endpoints = []
        
        for i in range(rows):
            for j in range(cols):
                if char_grid[i][j] == '@':
                    adjacent_endpoints = count_adjacent_endpoints((i, j))
                    if adjacent_endpoints < 2:
                        needed = 2 - adjacent_endpoints
                        shelves_needing_endpoints.append(((i, j), needed))
                        
                        # Try to add missing endpoints
                        empty_adjacent = find_empty_adjacent_positions((i, j))
                        for _ in range(min(needed, len(empty_adjacent))):
                            if empty_adjacent:
                                pos = empty_adjacent.pop(0)
                                char_grid[pos[0]][pos[1]] = 'e'
                                added_endpoints.append(pos)
        
        print(f"  Found {len(shelves_needing_endpoints)} shelves needing endpoints")
        print(f"  Added {len(added_endpoints)} new endpoints")
        
        # Show details of added endpoints
        if added_endpoints:
            print(f"  Freshly added endpoints:")
            for i, (row, col) in enumerate(added_endpoints):
                print(f"    Endpoint {i+1}: Position ({row}, {col})")
        else:
            print(f"  No new endpoints were added")
        
        # Step 3: Add left and right 2x columns
        print("Step 3: Adding left and right 2x columns...")
        for i in range(rows):
            # Add left blank column
            char_grid[i].insert(0, '.')
            # Add left w column (where row index % 3 == 1)
            if i % 3 == 1:
                char_grid[i].insert(0, 'w')
            else:
                char_grid[i].insert(0, '.')
            # Add right blank column
            char_grid[i].append('.')
            # Add right w column (where row index % 3 == 1)
            if i % 3 == 1:
                char_grid[i].append('w')
            else:
                char_grid[i].append('.')
        
        # Update dimensions
        new_rows, new_cols = len(char_grid), len(char_grid[0])
        print(f"  Grid expanded from {rows}x{cols} to {new_rows}x{new_cols}")
        
        # Step 4: Check for connectivity
        print("Step 4: Checking connectivity...")
        
        def count_adjacent_endpoints(shelf_pos):
            """Count how many endpoints are adjacent to a shelf."""
            i, j = shelf_pos
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < new_rows and 0 <= nj < new_cols and 
                    char_grid[ni][nj] == 'e'):
                    count += 1
            return count
        
        def count_adjacent_shelves(endpoint_pos):
            """Count how many shelves are adjacent to an endpoint."""
            i, j = endpoint_pos
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < new_rows and 0 <= nj < new_cols and 
                    char_grid[ni][nj] == '@'):
                    count += 1
            return count
        
        # helper to compute visited grid and unconnected endpoint count
        def compute_connectivity():
            """Return (visited_matrix, unconnected_endpoint_count) after DFS from first endpoint"""
            visited_tmp = [[False for _ in range(new_cols)] for _ in range(new_rows)]
            # locate first endpoint
            start_i = start_j = None
            for ii in range(new_rows):
                found = False
                for jj in range(new_cols):
                    if char_grid[ii][jj] == 'e':
                        start_i, start_j = ii, jj
                        found = True
                        break
                if found:
                    break
            if start_i is not None:
                stack = [(start_i, start_j)]
                while stack:
                    ci, cj = stack.pop()
                    if ci < 0 or ci >= new_rows or cj < 0 or cj >= new_cols:
                        continue
                    if visited_tmp[ci][cj] or char_grid[ci][cj] == '@':
                        continue
                    visited_tmp[ci][cj] = True
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        stack.append((ci+dx, cj+dy))
            # count unconnected endpoints
            unconn = 0
            for ii in range(new_rows):
                for jj in range(new_cols):
                    if char_grid[ii][jj] == 'e' and not visited_tmp[ii][jj]:
                        unconn += 1
            return visited_tmp, unconn
        
        # Repeat isolated endpoint removal and re-checking 3 times
        total_isolated_removed = 0
        total_additional_added = 0
        total_shelves_removed = 0
        
        for iteration in range(3):
            print(f"Step 4.{iteration+1}: Iteration {iteration+1} - Processing isolated endpoints...")
            
            # Step 4a: Remove isolated endpoints (endpoints with no adjacent shelves OR not connected to main component)
            isolated_endpoints_removed = []
            shelves_removed_this_iteration = []
            
            # First, find the main connected component
            visited = [[False for _ in range(new_cols)] for _ in range(new_rows)]
            
            def dfs_find_component(i, j):
                if i < 0 or i >= new_rows or j < 0 or j >= new_cols:
                    return
                if char_grid[i][j] == '@' or visited[i][j]:
                    return
                visited[i][j] = True
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs_find_component(i + dx, j + dy)
            
            # Find the first endpoint to start DFS
            start_found = False
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == 'e':
                        dfs_find_component(i, j)
                        start_found = True
                        break
                if start_found:
                    break
                
            # Now remove endpoints that are either:
            # 1. Not connected to the main component (not visited by DFS)
            # 2. Have no adjacent shelves
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == 'e':
                        should_remove = False
                        reason = ""
                        
                        # Check if not connected to main component
                        if not visited[i][j]:
                            should_remove = True
                            reason = "not connected to main component"
                        else:
                            # Check if has no adjacent shelves
                            adjacent_shelves = count_adjacent_shelves((i, j))
                            if adjacent_shelves == 0:
                                should_remove = True
                                reason = "no adjacent shelves"
                        
                        if should_remove:
                            char_grid[i][j] = '.'  # Remove isolated endpoint
                            isolated_endpoints_removed.append((i, j, reason))
                            
                            # Remove ONE adjacent shelf to this isolated endpoint
                            shelf_found = False
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ni, nj = i + dx, j + dy
                                if (0 <= ni < new_rows and 0 <= nj < new_cols and 
                                    char_grid[ni][nj] == '@' and not shelf_found):
                                    char_grid[ni][nj] = '.'  # Remove adjacent shelf
                                    shelves_removed_this_iteration.append((ni, nj))
                                    shelf_found = True
                                    break  # Only remove ONE adjacent shelf
            
            total_isolated_removed += len(isolated_endpoints_removed)
            total_shelves_removed += len(shelves_removed_this_iteration)
            
            print(f"  Removed {len(isolated_endpoints_removed)} isolated endpoints")
            print(f"  Removed {len(shelves_removed_this_iteration)} adjacent shelves")
            if isolated_endpoints_removed:
                print(f"  Isolated endpoints removed:")
                for i, (row, col, reason) in enumerate(isolated_endpoints_removed):
                    print(f"    Endpoint {i+1}: Position ({row}, {col}) - {reason}")
            
            if shelves_removed_this_iteration:
                print(f"  Adjacent shelves removed:")
                for i, (row, col) in enumerate(shelves_removed_this_iteration):
                    print(f"    Shelf {i+1}: Position ({row}, {col})")
            
            # Step 4b: Immediately add back the removed shelves at random positions
            print(f"  Adding back {len(shelves_removed_this_iteration)} removed shelves at random positions...")
            import random
            random.seed(42 + iteration)  # Different seed for each iteration
            
            # Find all empty positions
            empty_positions = []
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == '.':
                        empty_positions.append((i, j))
            
            # Randomly select positions to add back the removed shelves
            shelves_to_add_back = min(len(shelves_removed_this_iteration), len(empty_positions))
            if shelves_to_add_back > 0:
                selected_positions = random.sample(empty_positions, shelves_to_add_back)
                
                # For bookkeeping, show which removed shelf went where
                print(f"  Mapping of removed shelves to new positions:")
                for i, (new_pos) in enumerate(selected_positions):
                    print(f"    Removed shelf {i+1}: Added back at position ({new_pos[0]}, {new_pos[1]})")
                
                for i, j in selected_positions:
                    char_grid[i][j] = '@'
                
                print(f"  Added back {len(selected_positions)} shelves at random positions")
            else:
                print(f"  No empty positions available to add back shelves")
            
            # Step 4c: Check if any shelves need more endpoints after adding back shelves
            shelves_still_needing_endpoints = []
            additional_endpoints_added = []
            
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == '@':
                        adjacent_endpoints = count_adjacent_endpoints((i, j))
                        if adjacent_endpoints < 2:
                            needed = 2 - adjacent_endpoints
                            shelves_still_needing_endpoints.append(((i, j), needed))
                            
                            # Try to add missing endpoints - prioritize connected positions
                            connected_empty_adjacent = find_connected_empty_adjacent_positions((i, j), visited)
                            regular_empty_adjacent = find_empty_adjacent_positions((i, j))
                            
                            # Use connected positions first, then fall back to any empty positions
                            available_positions = connected_empty_adjacent + [pos for pos in regular_empty_adjacent if pos not in connected_empty_adjacent]
                            
                            for _ in range(min(needed, len(available_positions))):
                                if available_positions:
                                    pos = available_positions.pop(0)
                                    char_grid[pos[0]][pos[1]] = 'e'
                                    additional_endpoints_added.append(pos)
            
            total_additional_added += len(additional_endpoints_added)
            print(f"  Found {len(shelves_still_needing_endpoints)} shelves still needing endpoints")
            print(f"  Added {len(additional_endpoints_added)} additional endpoints")
            
            if additional_endpoints_added:
                print(f"  Additional endpoints added:")
                for i, (row, col) in enumerate(additional_endpoints_added):
                    print(f"    Endpoint {i+1}: Position ({row}, {col})")
            
            # Step 4d: Check connectivity after endpoints are added
            visited, unconnected_after_endpoint = compute_connectivity()
            
            # Count shelves that still need endpoints
            current_invalid_shelves = []
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == '@':
                        adjacent_endpoints = count_adjacent_endpoints((i, j))
                        if adjacent_endpoints < 2:
                            current_invalid_shelves.append((i, j))
            
            print(f"  Connectivity after endpoint addition → unconnected endpoints: {unconnected_after_endpoint}")
            print(f"  Shelves still needing endpoints: {len(current_invalid_shelves)}")
            
            # If everything valid, break early
            if unconnected_after_endpoint == 0 and len(current_invalid_shelves) == 0:
                print(f"  All connectivity and shelf checks satisfied after iteration {iteration+1}. Stopping early.")
                break
            
            # Step 4e: Check if no changes were made - if so, stop early
            if len(isolated_endpoints_removed) == 0 and len(additional_endpoints_added) == 0:
                print(f"  No changes in iteration {iteration+1}, stopping early")
                break
        
        print(f"Total isolated endpoints removed across all iterations: {total_isolated_removed}")
        print(f"Total adjacent shelves removed across all iterations: {total_shelves_removed}")
        print(f"Total additional endpoints added across all iterations: {total_additional_added}")
        
        # Step 4e: Final connectivity check and blue tile placement
        print("Step 4e: Final connectivity check and blue tile placement...")
        
        def place_blue_tiles_adjacent_to_black(grid):
            """
            For each black tile ('@'), ensure at least 2 adjacent blue tiles ('e').
            Adapted from dfs.py logic.
            """
            base_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            endpoints_added = 0
            
            for i in range(new_rows):
                for j in range(new_cols):
                    if grid[i][j] == '@':
                        placed_count = 0
                        
                        # Count existing adjacent blue tiles
                        for dx, dy in base_directions:
                            ni, nj = i + dx, j + dy
                            if (0 <= ni < new_rows and 0 <= nj < new_cols and 
                                grid[ni][nj] == 'e'):
                                placed_count += 1
                        
                        # If not enough, try placing new blue tiles
                        if placed_count < 2:
                            directions = base_directions.copy()
                            random.shuffle(directions)
                            for dx, dy in directions:
                                ni, nj = i + dx, j + dy
                                if (0 <= ni < new_rows and 0 <= nj < new_cols and 
                                    grid[ni][nj] == '.'):
                                    grid[ni][nj] = 'e'
                                    placed_count += 1
                                    endpoints_added += 1
                                    if placed_count >= 2:
                                        break
            
            return endpoints_added
        
        new_endpoints_added = place_blue_tiles_adjacent_to_black(char_grid)
        print(f"  Added {new_endpoints_added} new endpoints using dfs.py logic")
        
        # Final connectivity check
        print("  Final connectivity check...")
        
        # Re-run DFS to find main component
        visited = [[False for _ in range(new_cols)] for _ in range(new_rows)]
        
        def dfs_final(i, j):
            if i < 0 or i >= new_rows or j < 0 or j >= new_cols:
                return
            if char_grid[i][j] == '@' or visited[i][j]:
                return
            visited[i][j] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs_final(i + dx, j + dy)
        
        # Find the first endpoint to start DFS
        start_found = False
        for i in range(new_rows):
            for j in range(new_cols):
                if char_grid[i][j] == 'e':
                    dfs_final(i, j)
                    start_found = True
                    break
            if start_found:
                break
        
        # Count final connectivity
        total_endpoints_final = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e')
        connected_endpoints_final = sum(1 for i in range(new_rows) for j in range(new_cols) 
                                      if char_grid[i][j] == 'e' and visited[i][j])
        unconnected_endpoints_final = total_endpoints_final - connected_endpoints_final
        
        print(f"  Final results:")
        print(f"    Total endpoints: {total_endpoints_final}")
        print(f"    Connected endpoints: {connected_endpoints_final}")
        print(f"    Unconnected endpoints: {unconnected_endpoints_final}")
        print(f"    Connectivity: {(connected_endpoints_final/total_endpoints_final)*100:.1f}%")
        
        # Final validation
        final_shelves_valid = True
        final_invalid_shelves = []
        
        for i in range(new_rows):
            for j in range(new_cols):
                if char_grid[i][j] == '@':
                    adjacent_endpoints = count_adjacent_endpoints((i, j))
                    if adjacent_endpoints < 2:
                        final_shelves_valid = False
                        final_invalid_shelves.append((i, j, adjacent_endpoints))
        
        overall_valid = (unconnected_endpoints_final == 0) and final_shelves_valid
        
        # Convert back to tensor (keep original 33x32 size for compatibility)
        processed_grid = torch.zeros(1, rows, cols)
        for i in range(rows):
            for j in range(cols):
                # Map from expanded grid back to original positions
                # Skip the 2 left columns and 2 right columns
                expanded_i = i
                expanded_j = j + 2  # Skip left blank and left w columns
                if char_grid[expanded_i][expanded_j] == '@':
                    processed_grid[0, i, j] = 1.0
                elif char_grid[expanded_i][expanded_j] == 'e':
                    processed_grid[0, i, j] = -1.0
                else:
                    processed_grid[0, i, j] = 0.0
        
        # Count final elements in the expanded grid (use the same grid that was validated)
        final_shelves = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == '@')
        final_endpoints = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e')
        final_empty = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == '.')
        final_workstations = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'w')
        
        # Also count the actual endpoints that are connected vs unconnected for consistency
        visited = [[False for _ in range(new_cols)] for _ in range(new_rows)]
        
        def dfs_count(i, j):
            if i < 0 or i >= new_rows or j < 0 or j >= new_cols:
                return
            if char_grid[i][j] == '@' or visited[i][j]:
                return
            visited[i][j] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs_count(i + dx, j + dy)
        
        # Find first endpoint and do DFS
        start_found = False
        for i in range(new_rows):
            for j in range(new_cols):
                if char_grid[i][j] == 'e':
                    dfs_count(i, j)
                    start_found = True
                    break
            if start_found:
                break
        
        # Count connected vs unconnected endpoints
        connected_endpoints = sum(1 for i in range(new_rows) for j in range(new_cols) 
                                if char_grid[i][j] == 'e' and visited[i][j])
        unconnected_endpoints = sum(1 for i in range(new_rows) for j in range(new_cols) 
                                  if char_grid[i][j] == 'e' and not visited[i][j])
        
        post_processing_results = {
            'overall_valid': overall_valid,
            'endpoints_connected': (unconnected_endpoints_final == 0),
            'shelves_valid': final_shelves_valid,
            'missing_shelves_added': missing_shelves,
            'endpoints_added': len(added_endpoints),
            'isolated_endpoints_removed': total_isolated_removed,
            'adjacent_shelves_removed': total_shelves_removed,
            'additional_endpoints_added': total_additional_added,
            'shelves_added_back': len(selected_positions) if 'selected_positions' in locals() else 0,
            'new_endpoints_from_dfs': new_endpoints_added,
            'grid_expanded': True,
            'original_size': f"{rows}x{cols}",
            'expanded_size': f"{new_rows}x{new_cols}",
            'final_shelves': final_shelves,
            'final_endpoints': final_endpoints,
            'connected_endpoints': connected_endpoints,
            'unconnected_endpoints': unconnected_endpoints,
            'final_empty': final_empty,
            'final_workstations': final_workstations,
            'final_invalid_shelves': final_invalid_shelves,
            'final_connectivity_percentage': (connected_endpoints_final/total_endpoints_final)*100 if total_endpoints_final > 0 else 0
        }
        
        return processed_grid, post_processing_results

    def print_post_processing_results(self, post_processing_results):
        """Print post-processing results in a readable format."""
        print(f"\nPost-Processing Results:")
        print(f"=" * 50)
        print(f"Overall Valid: {'✅ YES' if post_processing_results['overall_valid'] else '❌ NO'}")
        print(f"Endpoints Connected: {'✅ YES' if post_processing_results['endpoints_connected'] else '❌ NO'}")
        print(f"All Shelves Valid: {'✅ YES' if post_processing_results['shelves_valid'] else '❌ NO'}")
        print(f"Missing Shelves Added: {post_processing_results['missing_shelves_added']}")
        print(f"Endpoints Added: {post_processing_results['endpoints_added']}")
        print(f"Isolated Endpoints Removed: {post_processing_results['isolated_endpoints_removed']}")
        print(f"Adjacent Shelves Removed: {post_processing_results['adjacent_shelves_removed']}")
        print(f"Additional Endpoints Added: {post_processing_results['additional_endpoints_added']}")
        print(f"Shelves Added Back: {post_processing_results['shelves_added_back']}")
        print(f"New Endpoints from dfs.py: {post_processing_results['new_endpoints_from_dfs']}")
        print(f"Final Shelves: {post_processing_results['final_shelves']}")
        print(f"Final Endpoints: {post_processing_results['final_endpoints']}")
        print(f"Connected Endpoints: {post_processing_results['connected_endpoints']}")
        print(f"Unconnected Endpoints: {post_processing_results['unconnected_endpoints']}")
        print(f"Final Empty: {post_processing_results['final_empty']}")
        print(f"Final Workstations: {post_processing_results['final_workstations']}")
        print(f"Final Invalid Shelves: {post_processing_results['final_invalid_shelves']}")
        print(f"Final Connectivity Percentage: {post_processing_results['final_connectivity_percentage']:.1f}%")
        
        print(f"=" * 50)

def main():
    """Example usage of the diffusion inference v2."""
    
    # Initialize inference
    inference = DiffusionInferenceV2()
    
    # Example target throughputs (in original scale)
    target_throughputs = [10]
    
    # Generate grids
    results = inference.generate_multiple_grids(target_throughputs, guidance_scale=9.0)
    
    # Print summary
    print("\n" + "="*60)
    print("Generated Grids Summary:")
    print("="*60)
    for result in results:
        print(f"Grid {result['grid_number']}: Throughput {result['target_throughput']:.2f}")
        analysis = result['continuous_analysis']
        print(f"  Continuous range: [{analysis['min']:.3f}, {analysis['max']:.3f}]")
        print(f"  Mean: {analysis['mean']:.3f}, Std: {analysis['std']:.3f}")
        print(f"  Shelves: {result['num_shelves']}, Endpoints: {result['num_endpoints']}")
        print()

if __name__ == "__main__":
    main() 