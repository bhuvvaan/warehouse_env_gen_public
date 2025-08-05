import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def display_continuous_outputs():
    """Display the continuous outputs from the diffusion model."""
    
    # Check if the generated_grids directory exists
    if not os.path.exists('generated_grids'):
        print("Generated grids directory not found. Please run inference_diffusion.py first.")
        return
    
    # Get all PNG files in the generated_grids directory
    png_files = [f for f in os.listdir('generated_grids') if f.endswith('.png')]
    png_files.sort()  # Sort to get consistent order
    
    if not png_files:
        print("No PNG files found in generated_grids directory.")
        return
    
    # Create a figure to display all continuous outputs
    fig, axes = plt.subplots(1, len(png_files), figsize=(4*len(png_files), 4))
    
    if len(png_files) == 1:
        axes = [axes]
    
    for i, png_file in enumerate(png_files):
        # Extract throughput value from filename
        throughput = png_file.replace('grid_throughput_', '').replace('.png', '')
        
        # Load and display the image
        img_path = os.path.join('generated_grids', png_file)
        img = mpimg.imread(img_path)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Throughput: {throughput}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('continuous_outputs_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Displayed {len(png_files)} continuous outputs from diffusion model")
    print("Images saved as 'continuous_outputs_summary.png'")
    
    # Also show the original training samples
    if os.path.exists('diffusion_samples.png'):
        print("\nOriginal training samples (diffusion_samples.png):")
        img = mpimg.imread('diffusion_samples.png')
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Original Diffusion Training Samples\n(Top: Continuous, Bottom: Binarized)')
        plt.tight_layout()
        plt.savefig('original_training_samples.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    display_continuous_outputs() 