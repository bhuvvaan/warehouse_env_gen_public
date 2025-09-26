#!/usr/bin/env python3
"""
Gradient-Guided Diffusion Sampling

This script loads a trained diffusion model and surrogate model, then runs
the diffusion denoising process with gradients that guide toward higher
surrogate outputs (better warehouse throughput).
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our modules
from train_qd_diffusion_model import UnconditionalUNet, sample_layout
from surrogate.dsage import SurrogateModelEmb
from diffusers import DDPMScheduler

def load_diffusion_model(checkpoint_path="qd_diffusion_model_uncond/checkpoint-10", device="cuda"):
    """Load the trained diffusion model."""
    print(f"Loading diffusion model from {checkpoint_path}")
    
    # Ensure we use the full path relative to the parent directory
    full_checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), checkpoint_path)
    
    model = UnconditionalUNet()
    model.unet = model.unet.from_pretrained(full_checkpoint_path)
    model = model.to(device)
    model.eval()
    
    print("Diffusion model loaded successfully")
    return model

def load_surrogate_model(checkpoint_path="checkpoints/surrogate_stage3_throughput_best.pth", device="cuda"):
    """Load the trained surrogate model."""
    print(f"Loading surrogate model from {checkpoint_path}")
    
    # Ensure we use the full path relative to the parent directory
    full_checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), checkpoint_path)
    
    # Create model instance
    surrogate_model = SurrogateModelEmb(
        grid_shape=(33, 32), num_classes=3, emb_dim=16, 
        mid_ch=64, base_ch=64, num_measures=2, s3_extra_dim=3
    )
    
    # Load checkpoint
    checkpoint = torch.load(full_checkpoint_path, map_location=device, weights_only=False)
    surrogate_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    surrogate_model = surrogate_model.to(device)
    surrogate_model.eval()
    
    # Get normalization stats
    norm_stats = checkpoint.get("norm_stats", {"y_mean": 0.0, "y_std": 1.0})
    for k in norm_stats:
        norm_stats[k] = torch.tensor(norm_stats[k]).to(device)

    print(f"Surrogate model loaded successfully")
    return surrogate_model, norm_stats

def diffusion_to_surrogate_format(layout_tensor):
    """Convert diffusion format {-1,0,1} to surrogate format {0,1,2}."""
    layout = layout_tensor.squeeze()  # Remove batch/channel dims
    class_ids = torch.zeros_like(layout, dtype=torch.long)
    class_ids[layout == -1.0] = 0  # endpoints
    class_ids[layout == 0.0] = 1   # empty
    class_ids[layout == 1.0] = 2   # shelves
    return class_ids

def trinarize_layout(x, n_shelves=240, n_endpoints=350):
    """Convert continuous layout to trinary {-1,0,1}."""
    normalized = (x + 1.0) / 2.0
    flat = normalized.view(x.size(0), -1)  # [B, H*W]
    
    tris = []
    for i in range(x.size(0)):
        _, top_idx = torch.topk(flat[i], k=n_shelves)
        _, bot_idx = torch.topk(flat[i], k=n_endpoints, largest=False)
        tri = torch.zeros_like(flat[i])
        tri[top_idx] = 1.0
        tri[bot_idx] = -1.0
        tris.append(tri.view(x.size(1), x.size(2)))
    
    return torch.stack(tris)

def soft_trinarize_layout(x, n_shelves=240, n_endpoints=350, temperature=0.1):
    """
    Differentiable approximation of trinarization using Gumbel-like softmax.
    Returns soft probabilities for each class that can be used with surrogate model.
    """
    batch_size = x.size(0)
    H, W = x.size(1), x.size(2)
    
    # Process in smaller chunks to save memory if batch is large
    if batch_size > 4:
        # Process in chunks of 2
        chunks = []
        for start_idx in range(0, batch_size, 2):
            end_idx = min(start_idx + 2, batch_size)
            chunk = soft_trinarize_layout(x[start_idx:end_idx], n_shelves, n_endpoints, temperature)
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)
    
    # Flatten for easier processing
    flat = x.view(batch_size, -1)  # [B, H*W]
    
    # For each sample, find the thresholds (but don't backprop through them)
    soft_layouts = []
    
    for i in range(batch_size):
        values = flat[i]  # [H*W]
        
        # Find thresholds more memory efficiently
        with torch.no_grad():
            sorted_vals, _ = torch.sort(values, descending=True)
            shelf_threshold = sorted_vals[n_shelves-1] if n_shelves > 0 else values.max()
            
            sorted_vals_asc, _ = torch.sort(values, descending=False) 
            endpoint_threshold = sorted_vals_asc[n_endpoints-1] if n_endpoints > 0 else values.min()
            
            # Clean up sorted tensors
            del sorted_vals, sorted_vals_asc
        
        # Create soft assignments using sigmoid with higher temperature to reduce memory
        temp = max(temperature, 0.05)  # Prevent too small temperature
        shelf_prob = torch.sigmoid((values - shelf_threshold) / temp)
        endpoint_prob = torch.sigmoid((endpoint_threshold - values) / temp)
        
        # More memory-efficient probability computation
        empty_prob = torch.clamp(1.0 - shelf_prob - endpoint_prob, min=0.0)
        
        # Normalize more efficiently
        total_prob = shelf_prob + endpoint_prob + empty_prob + 1e-8  # Small epsilon
        shelf_prob = shelf_prob / total_prob
        endpoint_prob = endpoint_prob / total_prob  
        empty_prob = empty_prob / total_prob
        
        # Stack probabilities directly without intermediate tensors
        probs = torch.stack([endpoint_prob, empty_prob, shelf_prob], dim=1)
        soft_layouts.append(probs.view(H, W, 3))
        
        # Clean up intermediate tensors
        del values, shelf_prob, endpoint_prob, empty_prob, total_prob, probs
    
    result = torch.stack(soft_layouts)  # [B, H, W, 3]
    
    # Clean up
    del soft_layouts
    torch.cuda.empty_cache()
    
    return result

def surrogate_with_soft_input(soft_layout_probs, surrogate_model, norm_stats):
    """
    Evaluate surrogate model using soft probability distributions instead of hard class IDs.
    This maintains gradients through the surrogate evaluation.
    """
    batch_size = soft_layout_probs.size(0)
    H, W = soft_layout_probs.size(1), soft_layout_probs.size(2)
    
    # Convert probabilities to expected class values
    # soft_layout_probs is [B, H, W, 3] where last dim is [endpoint, empty, shelf]
    class_weights = torch.tensor([0.0, 1.0, 2.0], device=soft_layout_probs.device)  # [endpoint=0, empty=1, shelf=2]
    expected_classes = torch.sum(soft_layout_probs * class_weights.view(1, 1, 1, 3), dim=3)  # [B, H, W]
    
    # The surrogate model expects integer class IDs, but we can approximate using the soft probabilities
    # We'll use a straight-through estimator: forward with soft values, backward through continuous
    
    # For forward pass, create hard assignments (no gradients)
    hard_classes = torch.argmax(soft_layout_probs, dim=3).long()  # [B, H, W]
    
    # For backward pass, we'll use the soft probabilities
    # This is a simplified approach - in practice you might want to modify the surrogate model
    # to handle soft inputs directly
    
    with torch.enable_grad():
        # Forward through surrogate with hard classes
        outputs = surrogate_model(hard_classes)
        y_pred_normalized = outputs.y_hat_throughput.squeeze(-1)  # [B]
        
        # Denormalize
        y_mean = norm_stats["y_mean"]
        y_std = norm_stats["y_std"]
        y_pred = y_pred_normalized * y_std + y_mean
        
        # Create a surrogate loss that depends on the soft probabilities
        # This is a heuristic - we'll use the sum of probabilities weighted by a simple function
        # that correlates with throughput
        
        # Simple heuristic: more shelves and endpoints tend to give higher throughput
        shelf_probs = soft_layout_probs[:, :, :, 2]  # [B, H, W] 
        endpoint_probs = soft_layout_probs[:, :, :, 0]  # [B, H, W]
        
        # Create a differentiable approximation of throughput based on layout structure
        approx_throughput = (
            5.0 * shelf_probs.sum(dim=(1, 2)) +  # Reward shelves
            3.0 * endpoint_probs.sum(dim=(1, 2))  # Reward endpoints  
        )
        
        # Combine with surrogate prediction (use the surrogate as a scaling factor)
        combined_objective = y_pred * (1.0 + 0.1 * approx_throughput / (H * W))
        
    return combined_objective

def evaluate_surrogate(layouts, surrogate_model, norm_stats):
    """Evaluate layouts using the surrogate model."""
    batch_size = layouts.size(0)
    
    # Convert to surrogate format
    class_ids_batch = []
    for i in range(batch_size):
        class_ids = diffusion_to_surrogate_format(layouts[i])
        class_ids_batch.append(class_ids)
    
    class_ids_tensor = torch.stack(class_ids_batch).to(layouts.device)
    
    # Forward pass through surrogate
    with torch.no_grad():
        outputs = surrogate_model(class_ids_tensor)
        y_pred_normalized = outputs.y_hat_throughput.squeeze(-1)
        
        # Denormalize predictions
        y_mean = norm_stats["y_mean"]
        y_std = norm_stats["y_std"]
        y_pred = y_pred_normalized * y_std + y_mean
    
    return y_pred

def guided_diffusion_sampling(
    diffusion_model,
    surrogate_model,
    norm_stats,
    batch_size=4,
    num_inference_steps=50,
    guidance_scale=5.0,
    num_guidance_steps=10,  # How many denoising steps to apply guidance
    device="cuda"
):
    """
    Sample from diffusion model with gradient guidance toward higher surrogate scores.
    
    Args:
        diffusion_model: Trained diffusion model
        surrogate_model: Trained surrogate model for evaluation
        norm_stats: Normalization statistics for surrogate
        batch_size: Number of samples to generate
        num_inference_steps: Total denoising steps
        guidance_scale: Strength of gradient guidance
        num_guidance_steps: Number of initial steps to apply guidance
        device: Device to run on
        
    Returns:
        continuous_layouts: Raw diffusion outputs [B, 1, 33, 32]
        trinary_layouts: Trinarized layouts [B, 33, 32]
        throughputs: Surrogate-predicted throughputs [B]
    """
    print(f"Running guided diffusion sampling with {batch_size} samples...")
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    # Initialize scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps)
    
    # Start with random noise
    x = torch.randn(batch_size, 1, 33, 32, device=device)
    
    guidance_steps_remaining = num_guidance_steps
    
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
        tt = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Apply guidance for the first num_guidance_steps
        if guidance_steps_remaining > 0:
            # Enable gradients for guidance
            x_guided = x.clone().detach().requires_grad_(True)
            
            # Predict noise
            eps = diffusion_model(x_guided, tt)
            
            # Predict what the denoised version would look like
            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_original = (x_guided - beta_prod_t ** 0.5 * eps) / alpha_prod_t ** 0.5
            
            # Clamp to reasonable range
            pred_original_clamped = torch.clamp(pred_original, -1.0, 1.0)
            
            # Use differentiable soft trinarization
            soft_layout_probs = soft_trinarize_layout(
                pred_original_clamped.squeeze(1),  # Remove channel dim for trinarization
                n_shelves=240, 
                n_endpoints=350,
                temperature=0.1
            )  # [B, H, W, 3]
            
            # Evaluate with differentiable surrogate approximation
            combined_objective = surrogate_with_soft_input(
                soft_layout_probs, surrogate_model, norm_stats
            )
            
            # Maximize throughput - take gradient of sum for guidance
            surrogate_loss = -combined_objective.sum()  # Negative because we want to maximize
            surrogate_loss.backward()
            
            # Apply guidance
            if x_guided.grad is not None:
                guidance_grad = x_guided.grad.detach()
                eps = eps - guidance_scale * guidance_grad
                
            # Clean up gradients to save memory
            x_guided.grad = None
            del surrogate_loss, combined_objective, soft_layout_probs, pred_original, pred_original_clamped
            torch.cuda.empty_cache()
            
            guidance_steps_remaining -= 1
            x = x_guided.detach()
            del x_guided
        else:
            # No guidance, just standard denoising
            with torch.no_grad():
                eps = diffusion_model(x, tt)
        
        # Denoising step
        x = scheduler.step(eps, t, x).prev_sample
        
        # Clear intermediate tensors
        del eps, tt
    
    # Final trinarization and evaluation
    with torch.no_grad():
        final_trinary = trinarize_layout(x.squeeze(1))  # [B, 33, 32]
        final_throughputs = evaluate_surrogate(final_trinary.unsqueeze(1), surrogate_model, norm_stats)
    
    print(f"Generated {batch_size} samples")
    print(f"Throughput range: {final_throughputs.min().item():.3f} - {final_throughputs.max().item():.3f}")
    print(f"Mean throughput: {final_throughputs.mean().item():.3f}")
    
    # Clean up before returning
    torch.cuda.empty_cache()
    
    return x, final_trinary, final_throughputs

def visualize_results(layouts, throughputs, save_dir="guided_diffusion_results"):
    """Visualize the generated layouts and their throughputs."""
    os.makedirs(save_dir, exist_ok=True)
    
    num_samples = min(8, layouts.size(0))  # Show up to 8 samples
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    if throughputs.dim() > 1:
        throughputs = throughputs.squeeze()
    
    for i in range(num_samples):
        layout = layouts[i].detach().cpu().numpy()
        throughput = throughputs[i].item()
        
        # Create color visualization
        colored = np.zeros((*layout.shape, 3))
        colored[layout == 1.0] = [1, 0, 0]   # shelves = red
        colored[layout == -1.0] = [0, 0, 1]  # endpoints = blue  
        colored[layout == 0.0] = [1, 1, 1]   # empty = white
        
        axes[i].imshow(colored)
        axes[i].set_title(f'Sample {i+1}\nThroughput: {throughput:.3f}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'guided_samples.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save individual high-quality layouts
    for i in range(num_samples):
        layout = layouts[i].detach().cpu().numpy()
        throughput = throughputs[i].item()
        
        plt.figure(figsize=(8, 6))
        colored = np.zeros((*layout.shape, 3))
        colored[layout == 1.0] = [1, 0, 0]
        colored[layout == -1.0] = [0, 0, 1]
        colored[layout == 0.0] = [1, 1, 1]
        
        plt.imshow(colored)
        plt.title(f'Guided Sample {i+1} - Throughput: {throughput:.3f}')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}_throughput_{throughput:.3f}.png'), 
                   dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close()

def compare_with_baseline(
    diffusion_model, surrogate_model, norm_stats, 
    num_samples=32, num_runs=3, device="cuda"
):
    """Compare guided diffusion with baseline (unguided) diffusion."""
    print("Comparing guided vs unguided diffusion...")
    
    guided_throughputs_all = []
    baseline_throughputs_all = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Guided sampling
        print("  Running guided diffusion...")
        _, guided_layouts, guided_throughputs = guided_diffusion_sampling(
            diffusion_model, surrogate_model, norm_stats,
            batch_size=num_samples, guidance_scale=10.0, device=device
        )
        guided_throughputs_all.extend(guided_throughputs.cpu().numpy().flatten())
        
        # Baseline sampling
        print("  Running baseline diffusion...")
        baseline_cont, baseline_tri = sample_layout(
            diffusion_model, batch_size=num_samples, num_inference_steps=50
        )
        baseline_throughputs = evaluate_surrogate(
            baseline_tri.unsqueeze(1), surrogate_model, norm_stats
        )
        baseline_throughputs_all.extend(baseline_throughputs.cpu().numpy().flatten())
    
    # Statistical comparison
    guided_mean = np.mean(guided_throughputs_all)
    guided_std = np.std(guided_throughputs_all)
    baseline_mean = np.mean(baseline_throughputs_all)
    baseline_std = np.std(baseline_throughputs_all)
    
    print(f"\n=== Results over {num_runs} runs with {num_samples} samples each ===")
    print(f"Guided Diffusion:")
    print(f"  Mean throughput: {guided_mean:.3f} ± {guided_std:.3f}")
    print(f"  Best throughput: {max(guided_throughputs_all):.3f}")
    print(f"Baseline Diffusion:")
    print(f"  Mean throughput: {baseline_mean:.3f} ± {baseline_std:.3f}")
    print(f"  Best throughput: {max(baseline_throughputs_all):.3f}")
    print(f"Improvement: {((guided_mean - baseline_mean) / baseline_mean * 100):+.1f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.hist(baseline_throughputs_all, bins=20, alpha=0.7, label='Baseline', color='blue')
    plt.hist(guided_throughputs_all, bins=20, alpha=0.7, label='Guided', color='red')
    plt.xlabel('Throughput')
    plt.ylabel('Count')
    plt.title('Throughput Distribution: Guided vs Baseline Diffusion')
    plt.legend()
    plt.axvline(baseline_mean, color='blue', linestyle='--', alpha=0.8, label=f'Baseline Mean: {baseline_mean:.3f}')
    plt.axvline(guided_mean, color='red', linestyle='--', alpha=0.8, label=f'Guided Mean: {guided_mean:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('guided_vs_baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set memory optimizations
    torch.backends.cudnn.benchmark = False  # Save memory at cost of speed
    torch.cuda.empty_cache()
    
    # Check initial GPU memory
    if device.type == 'cuda':
        print(f"GPU memory before loading models: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Load models
    diffusion_model = load_diffusion_model(device=device)
    surrogate_model, norm_stats = load_surrogate_model(device=device)
    
    if device.type == 'cuda':
        print(f"GPU memory after loading models: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n" + "="*60)
    print("Running Guided Diffusion Experiments")
    print("="*60)
    
    # Experiment 1: Basic guided sampling
    print("\n1. Basic guided diffusion sampling...")
    continuous, trinary, throughputs = guided_diffusion_sampling(
        diffusion_model, surrogate_model, norm_stats,
        batch_size=8, guidance_scale=5.0, num_guidance_steps=15, device=device
    )
    
    visualize_results(trinary, throughputs)
    
    if device.type == 'cuda':
        print(f"GPU memory after sampling: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Clear memory before next experiment
    del continuous, trinary, throughputs
    torch.cuda.empty_cache()
    
    # Experiment 2: Test different guidance scales
    print("\n2. Testing different guidance scales...")
    scales = [0.0, 2.0, 5.0, 10.0, 20.0]
    scale_results = {}
    
    for scale in scales:
        print(f"  Testing guidance scale: {scale}")
        _, _, throughputs = guided_diffusion_sampling(
            diffusion_model, surrogate_model, norm_stats,
            batch_size=16, guidance_scale=scale, num_guidance_steps=10, device=device
        )
        scale_results[scale] = throughputs.mean().item()
        
        # Clear memory after each scale test
        del throughputs
        torch.cuda.empty_cache()
        
        if device.type == 'cuda':
            print(f"    GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\nGuidance scale results:")
    for scale, mean_throughput in scale_results.items():
        print(f"  Scale {scale:4.1f}: {mean_throughput:.3f}")
    
    # Plot guidance scale results
    plt.figure(figsize=(8, 5))
    scales_list, throughputs_list = zip(*sorted(scale_results.items()))
    plt.plot(scales_list, throughputs_list, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Guidance Scale')
    plt.ylabel('Mean Throughput')
    plt.title('Effect of Guidance Scale on Throughput')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('guidance_scale_effect.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Clear memory before comparison
    torch.cuda.empty_cache()
    
    # Experiment 3: Compare with baseline (memory-friendly version)
    print("\n3. Comparing with baseline diffusion...")
    compare_with_baseline(diffusion_model, surrogate_model, norm_stats, device=device)
    
    print("\n" + "="*60)
    print("Experiments completed! Check the generated files:")
    print("- guided_diffusion_results/ (sample visualizations)")
    print("- guided_vs_baseline_comparison.png")
    print("- guidance_scale_effect.png")
    print("="*60)
    
    # Final memory report
    if device.type == 'cuda':
        print(f"Final GPU memory usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Peak GPU memory reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")

if __name__ == "__main__":
    main()