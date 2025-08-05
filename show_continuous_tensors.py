import torch
import numpy as np
import matplotlib.pyplot as plt
from inference_diffusion import DiffusionInference

def show_continuous_tensors():
    """Show the actual continuous tensor values from diffusion model outputs."""
    
    # Initialize the inference class
    inference = DiffusionInference()
    
    # Target throughputs to generate
    target_throughputs = [3.0, 4.0, 5.0, 7.0]
    
    # Generate continuous tensors
    continuous_tensors = []
    
    for throughput in target_throughputs:
        print(f"\nGenerating continuous tensor for throughput: {throughput}")
        
        # Generate the continuous grid tensor
        continuous_grid = inference.generate_grid(throughput, guidance_scale=7.5, num_samples=1)
        continuous_tensors.append(continuous_grid.squeeze())  # Remove batch dimension
        
        # Print statistics
        grid_data = continuous_grid.squeeze().numpy()
        print(f"  Min value: {grid_data.min():.4f}")
        print(f"  Max value: {grid_data.max():.4f}")
        print(f"  Mean value: {grid_data.mean():.4f}")
        print(f"  Std value: {grid_data.std():.4f}")
        print(f"  Shape: {grid_data.shape}")
    
    # Visualize all continuous tensors
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (tensor, throughput) in enumerate(zip(continuous_tensors, target_throughputs)):
        # Convert to numpy for visualization
        grid_data = tensor.numpy()
        
        # Create the visualization
        im = axes[i].imshow(grid_data, cmap='RdBu', vmin=-1, vmax=1)
        axes[i].set_title(f'Continuous Output\nThroughput: {throughput}\nRange: [{grid_data.min():.3f}, {grid_data.max():.3f}]')
        axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('continuous_tensors_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show value distribution for each throughput
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (tensor, throughput) in enumerate(zip(continuous_tensors, target_throughputs)):
        grid_data = tensor.numpy().flatten()
        
        axes[i].hist(grid_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[i].set_title(f'Value Distribution\nThroughput: {throughput}')
        axes[i].set_xlabel('Tensor Values')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {grid_data.mean():.3f}\nStd: {grid_data.std():.3f}\nMin: {grid_data.min():.3f}\nMax: {grid_data.max():.3f}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('continuous_tensors_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nContinuous tensor analysis complete!")
    print("Files saved:")
    print("- continuous_tensors_detailed.png: Visual representation of continuous outputs")
    print("- continuous_tensors_distribution.png: Value distribution histograms")

if __name__ == "__main__":
    show_continuous_tensors() 