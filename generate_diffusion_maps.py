import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import random

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

class DiffusionMapGenerator:
    """Class for generating valid diffusion maps and saving them as .map files."""
    
    def __init__(self, checkpoint_dir="reward_conditioned_diffusion_model_v2/checkpoint-5"):
        """
        Initialize the diffusion map generator.
        
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
    
    @torch.no_grad()
    def generate_grid(self, target_throughput, guidance_scale=7.0, num_samples=1, num_inference_steps=50):
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
    
    def validate_grid_connectivity(self, trinary_grid):
        """
        Validate grid connectivity using DFS logic.
        
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
        
        # Check if all endpoints are connected
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
        
        # Check each shelf has at least 2 adjacent endpoints
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
    
    def save_kiva_map(self, processed_grid, map_filepath):
        """
        Save the expanded grid as a Kiva-format .map file.
        """
        # Convert processed grid back to expanded format with left/right columns
        grid_data = processed_grid.squeeze().numpy()
        rows, cols = grid_data.shape
        
        # Convert to character grid (original 33x32)
        char_grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                if grid_data[i, j] == 1.0:
                    row.append('@')
                elif grid_data[i, j] == -1.0:
                    row.append('e')
                else:
                    row.append('.')
            char_grid.append(row)
        
        # Add left and right 2x columns to match the post-processing
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
        
        # Count endpoints in the expanded grid
        endpoint_count = sum(row.count('e') for row in char_grid)
        
        with open(map_filepath, 'w') as f:
            f.write(f"{len(char_grid)},{len(char_grid[0])}\n")
            f.write(f"{endpoint_count}\n")
            f.write(f"{endpoint_count}\n")
            f.write("1000\n")
            for row in char_grid:
                f.write(''.join(row) + '\n')

    def post_process_grid(self, trinary_grid):
        """
        Post-process the grid to fix connectivity issues using fixed thresholds.
        Args:
            trinary_grid: Trinary grid tensor of shape (1, 33, 32)
        Returns:
            Post-processed trinary grid tensor and validation results
        """
        import random
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
        # Step 1: Randomly add missing shelves (target: 240 shelves)
        current_shelves = sum(1 for i in range(rows) for j in range(cols) if char_grid[i][j] == '@')
        target_shelves = 240
        missing_shelves = max(0, target_shelves - current_shelves)
        if missing_shelves > 0:
            empty_positions = []
            for i in range(rows):
                for j in range(cols):
                    if char_grid[i][j] == '.':
                        empty_positions.append((i, j))
            random.seed(42)
            selected_positions = random.sample(empty_positions, min(missing_shelves, len(empty_positions)))
            for i, j in selected_positions:
                char_grid[i][j] = '@'
        # Step 2: Check and add blue tiles if <2 anywhere
        def count_adjacent_endpoints(shelf_pos):
            i, j = shelf_pos
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < rows and 0 <= nj < cols and char_grid[ni][nj] == 'e'):
                    count += 1
            return count
        def find_empty_adjacent_positions(pos):
            i, j = pos
            empty_positions = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < rows and 0 <= nj < cols and char_grid[ni][nj] == '.'):
                    empty_positions.append((ni, nj))
            return empty_positions
        shelves_needing_endpoints = []
        added_endpoints = []
        for i in range(rows):
            for j in range(cols):
                if char_grid[i][j] == '@':
                    adjacent_endpoints = count_adjacent_endpoints((i, j))
                    if adjacent_endpoints < 2:
                        needed = 2 - adjacent_endpoints
                        shelves_needing_endpoints.append(((i, j), needed))
                        empty_adjacent = find_empty_adjacent_positions((i, j))
                        for _ in range(min(needed, len(empty_adjacent))):
                            if empty_adjacent:
                                pos = empty_adjacent.pop(0)
                                char_grid[pos[0]][pos[1]] = 'e'
                                added_endpoints.append(pos)
        # Step 3: Add left and right 2x columns
        for i in range(rows):
            char_grid[i].insert(0, '.')
            if i % 3 == 1:
                char_grid[i].insert(0, 'w')
            else:
                char_grid[i].insert(0, '.')
            char_grid[i].append('.')
            if i % 3 == 1:
                char_grid[i].append('w')
            else:
                char_grid[i].append('.')
        new_rows, new_cols = len(char_grid), len(char_grid[0])
        def count_adjacent_endpoints_exp(shelf_pos):
            i, j = shelf_pos
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < new_rows and 0 <= nj < new_cols and char_grid[ni][nj] == 'e'):
                    count += 1
            return count
        def count_adjacent_shelves(endpoint_pos):
            i, j = endpoint_pos
            count = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if (0 <= ni < new_rows and 0 <= nj < new_cols and char_grid[ni][nj] == '@'):
                    count += 1
            return count
        def compute_connectivity():
            visited_tmp = [[False for _ in range(new_cols)] for _ in range(new_rows)]
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
            unconn = 0
            for ii in range(new_rows):
                for jj in range(new_cols):
                    if char_grid[ii][jj] == 'e' and not visited_tmp[ii][jj]:
                        unconn += 1
            return visited_tmp, unconn
        total_isolated_removed = 0
        total_additional_added = 0
        total_shelves_removed = 0
        for iteration in range(3):
            isolated_endpoints_removed = []
            shelves_removed_this_iteration = []
            visited = [[False for _ in range(new_cols)] for _ in range(new_rows)]
            def dfs_find_component(i, j):
                if i < 0 or i >= new_rows or j < 0 or j >= new_cols:
                    return
                if char_grid[i][j] == '@' or visited[i][j]:
                    return
                visited[i][j] = True
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    dfs_find_component(i + dx, j + dy)
            start_found = False
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == 'e':
                        dfs_find_component(i, j)
                        start_found = True
                        break
                if start_found:
                    break
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == 'e':
                        should_remove = False
                        if not visited[i][j]:
                            should_remove = True
                        else:
                            adjacent_shelves = count_adjacent_shelves((i, j))
                            if adjacent_shelves == 0:
                                should_remove = True
                        if should_remove:
                            char_grid[i][j] = '.'
                            isolated_endpoints_removed.append((i, j))
                            shelf_found = False
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ni, nj = i + dx, j + dy
                                if (0 <= ni < new_rows and 0 <= nj < new_cols and char_grid[ni][nj] == '@' and not shelf_found):
                                    char_grid[ni][nj] = '.'
                                    shelves_removed_this_iteration.append((ni, nj))
                                    shelf_found = True
                                    break
            total_isolated_removed += len(isolated_endpoints_removed)
            total_shelves_removed += len(shelves_removed_this_iteration)
            empty_positions = []
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == '.':
                        empty_positions.append((i, j))
            shelves_to_add_back = min(len(shelves_removed_this_iteration), len(empty_positions))
            if shelves_to_add_back > 0:
                selected_positions = random.sample(empty_positions, shelves_to_add_back)
                for i, j in selected_positions:
                    char_grid[i][j] = '@'
            shelves_still_needing_endpoints = []
            additional_endpoints_added = []
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == '@':
                        adjacent_endpoints = count_adjacent_endpoints_exp((i, j))
                        if adjacent_endpoints < 2:
                            needed = 2 - adjacent_endpoints
                            shelves_still_needing_endpoints.append(((i, j), needed))
                            empty_adjacent = find_empty_adjacent_positions((i, j))
                            for _ in range(min(needed, len(empty_adjacent))):
                                if empty_adjacent:
                                    pos = empty_adjacent.pop(0)
                                    char_grid[pos[0]][pos[1]] = 'e'
                                    additional_endpoints_added.append(pos)
            total_additional_added += len(additional_endpoints_added)
            visited, unconnected_after_endpoint = compute_connectivity()
            current_invalid_shelves = []
            for i in range(new_rows):
                for j in range(new_cols):
                    if char_grid[i][j] == '@':
                        adjacent_endpoints = count_adjacent_endpoints_exp((i, j))
                        if adjacent_endpoints < 2:
                            current_invalid_shelves.append((i, j))
            if unconnected_after_endpoint == 0 and len(current_invalid_shelves) == 0:
                break
            if len(isolated_endpoints_removed) == 0 and len(additional_endpoints_added) == 0:
                break
        def place_blue_tiles_adjacent_to_black(grid):
            base_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            endpoints_added = 0
            for i in range(new_rows):
                for j in range(new_cols):
                    if grid[i][j] == '@':
                        placed_count = 0
                        for dx, dy in base_directions:
                            ni, nj = i + dx, j + dy
                            if (0 <= ni < new_rows and 0 <= nj < new_cols and grid[ni][nj] == 'e'):
                                placed_count += 1
                        if placed_count < 2:
                            directions = base_directions.copy()
                            random.shuffle(directions)
                            for dx, dy in directions:
                                ni, nj = i + dx, j + dy
                                if (0 <= ni < new_rows and 0 <= nj < new_cols and grid[ni][nj] == '.'):
                                    grid[ni][nj] = 'e'
                                    placed_count += 1
                                    endpoints_added += 1
                                    if placed_count >= 2:
                                        break
            return endpoints_added
        new_endpoints_added = place_blue_tiles_adjacent_to_black(char_grid)
        visited = [[False for _ in range(new_cols)] for _ in range(new_rows)]
        def dfs_final(i, j):
            if i < 0 or i >= new_rows or j < 0 or j >= new_cols:
                return
            if char_grid[i][j] == '@' or visited[i][j]:
                return
            visited[i][j] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs_final(i + dx, j + dy)
        start_found = False
        for i in range(new_rows):
            for j in range(new_cols):
                if char_grid[i][j] == 'e':
                    dfs_final(i, j)
                    start_found = True
                    break
            if start_found:
                break
        total_endpoints_final = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e')
        connected_endpoints_final = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e' and visited[i][j])
        unconnected_endpoints_final = total_endpoints_final - connected_endpoints_final
        final_shelves_valid = True
        final_invalid_shelves = []
        for i in range(new_rows):
            for j in range(new_cols):
                if char_grid[i][j] == '@':
                    adjacent_endpoints = count_adjacent_endpoints_exp((i, j))
                    if adjacent_endpoints < 2:
                        final_shelves_valid = False
                        final_invalid_shelves.append((i, j, adjacent_endpoints))
        overall_valid = (unconnected_endpoints_final == 0) and final_shelves_valid
        processed_grid = torch.zeros(1, rows, cols)
        for i in range(rows):
            for j in range(cols):
                expanded_i = i
                expanded_j = j + 2
                if char_grid[expanded_i][expanded_j] == '@':
                    processed_grid[0, i, j] = 1.0
                elif char_grid[expanded_i][expanded_j] == 'e':
                    processed_grid[0, i, j] = -1.0
                else:
                    processed_grid[0, i, j] = 0.0
        final_shelves = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == '@')
        final_endpoints = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e')
        final_empty = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == '.')
        final_workstations = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'w')
        visited = [[False for _ in range(new_cols)] for _ in range(new_rows)]
        def dfs_count(i, j):
            if i < 0 or i >= new_rows or j < 0 or j >= new_cols:
                return
            if char_grid[i][j] == '@' or visited[i][j]:
                return
            visited[i][j] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs_count(i + dx, j + dy)
        start_found = False
        for i in range(new_rows):
            for j in range(new_cols):
                if char_grid[i][j] == 'e':
                    dfs_count(i, j)
                    start_found = True
                    break
            if start_found:
                break
        connected_endpoints = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e' and visited[i][j])
        unconnected_endpoints = sum(1 for i in range(new_rows) for j in range(new_cols) if char_grid[i][j] == 'e' and not visited[i][j])
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

    def generate_one_map(self, target_throughput=7.0, save_dir='diffusion_maps_kiva_format', map_number=1):
        """
        Generate one map, run full post-processing, check if valid, and save if valid.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Generating map {map_number} with target throughput: {target_throughput}")
        
        # Generate continuous grid
        continuous_grid = self.generate_grid(target_throughput, guidance_scale=3.0, num_samples=1)
        
        # Trinarize grid
        trinary_grid = self.trinarize_grid(continuous_grid)
        
        # Run full post-processing
        processed_grid, post_processing_results = self.post_process_grid(trinary_grid)
        
        print(f"\nPost-Processing Results:")
        print(f"Overall Valid: {'✅ YES' if post_processing_results['overall_valid'] else '❌ NO'}")
        print(f"Endpoints Connected: {'✅ YES' if post_processing_results['endpoints_connected'] else '❌ NO'}")
        print(f"All Shelves Valid: {'✅ YES' if post_processing_results['shelves_valid'] else '❌ NO'}")
        print(f"Final Shelves: {post_processing_results['final_shelves']}")
        print(f"Final Endpoints: {post_processing_results['final_endpoints']}")
        
        if post_processing_results['overall_valid']:
            # Save the map with numbered filename
            map_filename = f"diffusion_map_{map_number:03d}.map"
            map_filepath = os.path.join(save_dir, map_filename)
            self.save_kiva_map(processed_grid, map_filepath)
            
            # Save metadata
            metadata = {
                'map_number': map_number,
                'target_throughput': target_throughput,
                'post_processing_results': post_processing_results
            }
            metadata_filename = f"diffusion_map_{map_number:03d}_metadata.json"
            metadata_filepath = os.path.join(save_dir, metadata_filename)
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"\n✅ Valid map generated and saved!")
            print(f"Map saved to: {map_filepath}")
            print(f"Metadata saved to: {metadata_filepath}")
            return True
        else:
            print(f"\n❌ Invalid map generated - not saving")
            if 'final_invalid_shelves' in post_processing_results and post_processing_results['final_invalid_shelves']:
                print(f"Invalid shelves: {len(post_processing_results['final_invalid_shelves'])}")
            return False

def main():
    """Generate 100 valid diffusion maps."""
    
    # Initialize the generator
    generator = DiffusionMapGenerator()
    
    # Generate 100 valid maps
    num_maps = 10
    valid_maps_generated = 0
    total_attempts = 0
    
    print(f"Generating {num_maps} valid diffusion maps...")
    
    while valid_maps_generated < num_maps:
        total_attempts += 1
        print(f"\n{'='*60}")
        print(f"Attempt {total_attempts} - Generating map {valid_maps_generated + 1}/{num_maps}")
        print(f"{'='*60}")
        
        success = generator.generate_one_map(target_throughput=10.0, map_number=valid_maps_generated + 1)
        
        if success:
            valid_maps_generated += 1
            print(f"✅ Successfully generated map {valid_maps_generated}/{num_maps}")
        else:
            print(f"❌ Failed to generate valid map on attempt {total_attempts}")
    
    print(f"\n🎉 Generation complete!")
    print(f"   Valid maps generated: {valid_maps_generated}/{num_maps}")
    print(f"   Total attempts: {total_attempts}")
    print(f"   Success rate: {(valid_maps_generated/total_attempts)*100:.1f}%")
    print(f"   All maps saved in: diffusion_maps_kiva_format/")

if __name__ == "__main__":
    main()
