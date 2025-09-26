import os
import numpy as np
import torch
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
import pandas as pd
from tqdm import trange, tqdm
from diffusion_guided_emitter import DiffusionGuidedEmitter
from collections import deque

# Import the model classes and helper functions from training script
from train_qd_diffusion_model import UnconditionalUNet, GridDataset, sample_layout

# Add parent directory to path for surrogate import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surrogate.dsage import SurrogateModelEmb

# Global variables for surrogate model (will be loaded once)
_surrogate_model = None
_surrogate_norm_stats = None

# ----------------------------
# Model Loading Functions
# ----------------------------

# ----------------------------
# Model Loading Functions
# ----------------------------

def load_trained_diffusion_model(checkpoint_dir, device="cuda", use_reward=False):
    """
    Load a trained diffusion model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., "qd_diffusion_model/checkpoint-10")
        device: Device to load the model on
        use_reward: Whether the model was trained with reward conditioning
        
    Returns:
        tuple: (model, dataset, normalization_params)
    """
    print(f"Loading diffusion model from {checkpoint_dir}")
    
    # Load normalization parameters
    norm_params_path = os.path.join(checkpoint_dir, "normalization_params.pt")
    if os.path.exists(norm_params_path):
        norm_params = torch.load(norm_params_path, map_location=device)
        reward_mean = norm_params["reward_mean"]
        reward_std = norm_params["reward_std"]
        print(f"Loaded normalization params: mean={reward_mean:.3f}, std={reward_std:.3f}")
    else:
        norm_params = {}
        reward_mean = 0.0
        reward_std = 1.0
        print("Warning: No normalization params found, using defaults")
    
    # Create model instance
    model = UnconditionalUNet()
    
    # Load the UNet weights
    if use_reward:
        # For reward-conditioned models, load UNet2DConditionModel
        model.unet = UNet2DConditionModel.from_pretrained(checkpoint_dir)
        
        # Load reward embedding weights if they exist
        reward_emb_path = os.path.join(checkpoint_dir, "reward_embedding.pt")
        if os.path.exists(reward_emb_path):
            reward_emb_state = torch.load(reward_emb_path, map_location=device)
            model.reward_embedding.load_state_dict(reward_emb_state)
            print("Loaded reward embedding weights")
        else:
            print("Warning: No reward embedding weights found")
    else:
        # For non-reward models, load UNet2DModel
        model.unet = UNet2DModel.from_pretrained(checkpoint_dir)
        print("Loaded UNet2D model weights")
    
    model = model.to(device)
    model.eval()
    
    # Create a dummy dataset instance for normalization parameters
    # This is a lightweight version that just holds the normalization params
    class DummyDataset:
        def __init__(self, reward_mean, reward_std):
            self.reward_mean = reward_mean
            self.reward_std = reward_std
            
        def denormalize_reward(self, r):
            return r * self.reward_std + self.reward_mean
    
    dataset = DummyDataset(reward_mean, reward_std)
    
    return model, dataset, norm_params

def load_latest_checkpoint(model_dir="qd_diffusion_model", use_reward=False, device="cuda"):
    """
    Load the latest checkpoint from the model directory.
    
    Args:
        model_dir: Directory containing checkpoints
        use_reward: Whether to load reward-conditioned model
        device: Device to load on
        
    Returns:
        tuple: (model, dataset, checkpoint_path)
    """
    # Find all checkpoint directories
    checkpoint_dirs = []
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    
    # Sort by checkpoint number and get the latest
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = checkpoint_dirs[-1]
    checkpoint_path = os.path.join(model_dir, latest_checkpoint)
    
    print(f"Loading latest checkpoint: {checkpoint_path}")
    model, dataset, norm_params = load_trained_diffusion_model(
        checkpoint_path, device=device, use_reward=use_reward
    )
    
    return model, dataset, checkpoint_path

def load_surrogate_model(checkpoint_path="checkpoints/surrogate_stage3_throughput_best.pth", device="cuda"):
    """Load the trained surrogate model for throughput prediction."""
    global _surrogate_model, _surrogate_norm_stats
    
    print(f"Loading surrogate model from {checkpoint_path}")
    
    # Create model instance
    surrogate_model = SurrogateModelEmb(
        grid_shape=(33, 32), num_classes=3, emb_dim=16, 
        mid_ch=64, base_ch=64, num_measures=2, s3_extra_dim=3
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    surrogate_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    surrogate_model = surrogate_model.to(device)
    surrogate_model.eval()
    
    # Get normalization stats
    norm_stats = checkpoint.get("norm_stats", {"y_mean": 0.0, "y_std": 1.0})
    
    _surrogate_model = surrogate_model
    _surrogate_norm_stats = norm_stats
    
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

# ---------- Utility: layout transforms ----------

def trinarize_continuous_layout(x_img, n_shelves=240, n_endpoints=350):
    """x_img: [1, 33, 32] in [-1,1] -> trinary in {-1,0,1} with fixed counts."""
    normalized = (x_img + 1.0) / 2.0
    flat = normalized.view(-1)
    # Shelves = top n, endpoints = bottom n
    _, top_idx = torch.topk(flat, k=n_shelves)
    _, bot_idx = torch.topk(flat, k=n_endpoints, largest=False)
    tri = torch.zeros_like(flat)
    tri[top_idx] = 1.0
    tri[bot_idx] = -1.0
    return tri.view(1, 33, 32)

def flatten_layout(x_img):
    """[1,33,32] -> (33*32,) float numpy"""
    return x_img.view(-1).detach().cpu().numpy().astype(np.float32)

def unflatten_layout(vec):
    """(33*32,) -> [1,33,32] torch"""
    t = torch.tensor(vec, dtype=torch.float32)
    return t.view(1, 33, 32)

# ---------- Behavior features ----------

def _fixed_workstation_mask(H=33, W=32, device="cpu", dtype=torch.float32):
    """Pink 'w' mask: columns 0 and W-1, on rows where r % 3 == 1."""
    rows = torch.arange(H, device=device)
    cols = torch.arange(W, device=device)
    R, C = torch.meshgrid(rows, cols, indexing="ij")
    mask = ((C == 0) | (C == W - 1)) & (R % 3 == 1)
    return mask.to(dtype=dtype)

def _count_components_discrete(shelf_grid_bool):
    """Exact 4-connected component count on a boolean HxW tensor (no grad)."""
    H, W = shelf_grid_bool.shape
    comp = 0
    # Convert to CPU numpy indices for speed; keep it simple & robust
    shelf_np = shelf_grid_bool.detach().cpu().numpy()
    visited_np = np.zeros_like(shelf_np, dtype=bool)
    nbrs = [(1,0),(-1,0),(0,1),(0,-1)]
    for r in range(H):
        for c in range(W):
            if shelf_np[r, c] and not visited_np[r, c]:
                comp += 1
                # BFS
                dq = deque([(r, c)])
                visited_np[r, c] = True
                while dq:
                    rr, cc = dq.popleft()
                    for dr, dc in nbrs:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and shelf_np[nr, nc] and not visited_np[nr, nc]:
                            visited_np[nr, nc] = True
                            dq.append((nr, nc))
    return torch.tensor(float(comp), dtype=torch.float32, device=shelf_grid_bool.device)

def _components_proxy_soft(shelf_prob):
    """
    Differentiable surrogate for #components:
    comp ≈ sum_i s_i  - sum_(i~j) s_i * s_j  over 4-neighbors.
    This undercounts in presence of cycles but provides useful gradients.
    """
    # Horizontal and vertical neighbor products
    horiz = (shelf_prob[:, :-1] * shelf_prob[:, 1:]).sum()
    vert  = (shelf_prob[:-1, :] * shelf_prob[1:, :]).sum()
    nodes = shelf_prob.sum()
    comp_soft = nodes - (horiz + vert)
    return comp_soft

def _average_task_length_discrete(endpoint_mask, w_mask):
    """
    Average Manhattan distance between all endpoint cells and all w cells.
    endpoint_mask, w_mask: HxW boolean/float tensors with 1 where present.
    """
    device = endpoint_mask.device
    H, W = endpoint_mask.shape
    ep_idx = (endpoint_mask > 0.5).nonzero(as_tuple=False)
    w_idx  = (w_mask > 0.5).nonzero(as_tuple=False)
    if ep_idx.numel() == 0 or w_idx.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    # pairwise L1 via broadcasting
    ep_r = ep_idx[:, 0].view(-1, 1)
    ep_c = ep_idx[:, 1].view(-1, 1)
    w_r  = w_idx[:, 0].view(1, -1)
    w_c  = w_idx[:, 1].view(1, -1)
    dists = (ep_r - w_r).abs() + (ep_c - w_c).abs()  # [Ne, Nw]
    return dists.float().mean()

def _average_task_length_soft(endpoint_prob, w_mask):
    """
    Differentiable expected Manhattan distance between endpoint distribution and fixed w cells:
    E[d] = sum_{i,j} E_i * W_j * d(i,j) / (sum_i E_i * sum_j W_j).
    """
    device = endpoint_prob.device
    H, W = endpoint_prob.shape
    rows = torch.arange(H, device=device).view(-1, 1)
    cols = torch.arange(W, device=device).view(1, -1)

    # Precompute Manhattan distance grid to each w cell, then weight by E
    # Construct distance-to-w array by convolution-like trick:
    # d(i,j; p,q) = |i-p| + |j-q|. We'll compute via broadcasting.
    w_idx = (w_mask > 0.5).nonzero(as_tuple=False)
    if w_idx.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # distances from every grid cell to each w cell
    wr = w_idx[:, 0].view(1, 1, -1)  # [1,1,Nw]
    wc = w_idx[:, 1].view(1, 1, -1)
    R = rows.view(H, 1, 1)  # [H,1,1]
    C = cols.view(1, W, 1)  # [1,W,1]
    dists = (R - wr).abs() + (C - wc).abs()  # [H,W,Nw]

    # weight by endpoint_prob and w presence (w_mask is fixed binary)
    E_sum = endpoint_prob.sum().clamp(min=1e-8)
    W_sum = w_mask.sum().clamp(min=1e-8)
    # expected distance
    exp_d = (endpoint_prob.unsqueeze(-1) * dists).sum() / (E_sum * W_sum)
    return exp_d

def _as_batched(layouts_input):
    if isinstance(layouts_input, np.ndarray):
        if layouts_input.ndim == 1:
            return torch.from_numpy(layouts_input).float().view(1, 1, 33, 32)
        elif layouts_input.ndim == 2:
            B = layouts_input.shape[0]
            return torch.from_numpy(layouts_input).float().view(B, 1, 33, 32)
        else:
            raise ValueError(f"Unexpected numpy array shape: {layouts_input.shape}")
    elif isinstance(layouts_input, torch.Tensor):
        if layouts_input.dim() == 3:   # [1,33,32]
            return layouts_input.unsqueeze(0)
        elif layouts_input.dim() == 4: # [B,1,33,32]
            return layouts_input
        else:
            raise ValueError(f"Unexpected tensor shape: {layouts_input.shape}")
    else:
        raise ValueError(f"Unexpected input type: {type(layouts_input)}")

def behavior_fn(
    layouts_input,
    return_gradients=False,
    n_shelves=240,
    n_endpoints=350,
    tau=0.05,           # temperature for the sigmoid “soft top-k”
):
    """
    Features:
      [0] = number of connected shelf components (4-neighborhood)
      [1] = average task length (mean L1 distance between endpoints and fixed 'w')

    Forward:
      - If return_gradients=False: hard top-k trinarize then compute exact features.
      - If return_gradients=True: forward values from hard trinarization, but gradients
        come from soft masks built around the k-th thresholds (STE-style).
    """
    layouts = _as_batched(layouts_input)
    B, _, H, W = layouts.shape
    device = layouts.device

    # Ensure we can take grads if requested
    if return_gradients:
        layouts = layouts.clone().detach().requires_grad_(True)

    w_mask = _fixed_workstation_mask(H, W, device=device)

    hard_vals = []
    soft_vals = []

    for i in range(B):
        x = layouts[i, 0]  # [H,W], in [-1,1] (continuous diffusion output or already trinary)

        # ---- Hard trinarization for the forward value ----
        hard = trinarize_continuous_layout(x.view(1, H, W), n_shelves=n_shelves, n_endpoints=n_endpoints)[0]
        hard_shelf = (hard == 1.0)
        hard_endp  = (hard == -1.0)

        comps_hard = _count_components_discrete(hard_shelf)
        avglen_hard = _average_task_length_discrete(hard_endp, w_mask)
        hard_vals.append(torch.stack([comps_hard, avglen_hard], dim=0))

        if return_gradients:
            # ---- Soft masks for gradients (around data-dependent thresholds) ----
            # Find k-th thresholds, but don't backprop through them.
            flat = x.view(-1)
            # k-th largest (shelf threshold) = min of top-k values
            topk_vals, _ = torch.topk(flat, k=n_shelves)
            shelf_th = topk_vals.min().detach()
            # k-th smallest (endpoint threshold) = max of bottom-k values
            botk_vals, _ = torch.topk(flat, k=n_endpoints, largest=False)
            endp_th = botk_vals.max().detach()

            # Soft membership probabilities (sigmoids around thresholds)
            # shelves ≈ sigmoid((x - shelf_th)/tau)
            # endpoints ≈ sigmoid((endp_th - x)/tau)
            shelf_prob = torch.sigmoid((x - shelf_th) / tau)
            endp_prob  = torch.sigmoid((endp_th - x) / tau)

            # Differentiable proxies for the two features
            comp_soft  = _components_proxy_soft(shelf_prob)
            avglen_soft = _average_task_length_soft(endp_prob, w_mask)

            soft_vals.append(torch.stack([comp_soft, avglen_soft], dim=0))

    hard_features = torch.stack(hard_vals, dim=0)  # [B,2]

    if not return_gradients:
        return hard_features.detach().cpu().numpy()

    # Use soft features ONLY for gradient computation, but RETURN hard values.
    soft_features = torch.stack(soft_vals, dim=0)  # [B,2]

    # Compute per-feature per-pixel grads via backward on soft_features
    gradients = []
    for i in range(B):
        per_item = []
        for j in range(2):
            if layouts.grad is not None:
                layouts.grad.zero_()
            soft_features[i, j].backward(retain_graph=True)
            if layouts.grad is not None:
                g = layouts.grad[i, 0].detach().cpu().numpy().reshape(-1)  # (1056,)
            else:
                g = np.zeros(H * W, dtype=np.float32)
            per_item.append(g)
        gradients.append(np.stack(per_item))  # [2,1056]
    gradients = np.stack(gradients)  # [B,2,1056]

    return hard_features.detach().cpu().numpy(), gradients

# ---------- Objective function with surrogate model ----------

def objective_fn(layouts_input, return_gradients=False):
    """
    Compute objective using surrogate model.
    
    Args:
        layouts_input: Can be:
            - Single layout: torch tensor [1,33,32] in {-1,0,1}  
            - Batch of layouts: numpy array (B,1056) flattened or torch tensor [B,1,33,32]
        return_gradients: if True, also return gradients
    
    Returns:
        If return_gradients=False: objectives (float or numpy array)
        If return_gradients=True: (objectives, gradients)
    """
    global _surrogate_model, _surrogate_norm_stats
    
    # Load surrogate model if not loaded
    if _surrogate_model is None:
        load_surrogate_model()
    
    device = next(_surrogate_model.parameters()).device
    
    # Handle different input formats
    if isinstance(layouts_input, np.ndarray):
        if layouts_input.ndim == 1:
            # Single flattened layout (1056,) -> (1,1,33,32)
            layouts = torch.from_numpy(layouts_input).float().view(1, 1, 33, 32)
        elif layouts_input.ndim == 2:
            # Batch of flattened layouts (B,1056) -> (B,1,33,32)  
            B = layouts_input.shape[0]
            layouts = torch.from_numpy(layouts_input).float().view(B, 1, 33, 32)
        else:
            raise ValueError(f"Unexpected numpy array shape: {layouts_input.shape}")
    elif isinstance(layouts_input, torch.Tensor):
        if layouts_input.dim() == 3:
            # Single layout [1,33,32] -> [1,1,33,32]
            layouts = layouts_input.unsqueeze(0)
        elif layouts_input.dim() == 4:
            # Batch [B,1,33,32]
            layouts = layouts_input
        else:
            raise ValueError(f"Unexpected tensor shape: {layouts_input.shape}")
    else:
        raise ValueError(f"Unexpected input type: {type(layouts_input)}")
    
    batch_size = layouts.size(0)
    layouts = layouts.to(device)
    
    if return_gradients:
        layouts.requires_grad_(True)
    
    # Convert to surrogate format
    class_ids_batch = []
    for i in range(batch_size):
        class_ids = diffusion_to_surrogate_format(layouts[i])
        class_ids_batch.append(class_ids)
    
    class_ids_tensor = torch.stack(class_ids_batch).to(device)  # [B,33,32]

    # Forward pass through surrogate
    with torch.set_grad_enabled(return_gradients):
        outputs = _surrogate_model(class_ids_tensor)
        y_pred_normalized = outputs.y_hat_throughput.squeeze(-1)  # [B]

        # Denormalize predictions
        y_mean = torch.tensor(_surrogate_norm_stats["y_mean"]).to(device)
        y_std = torch.tensor(_surrogate_norm_stats["y_std"]).to(device)
        y_pred = y_pred_normalized * y_std + y_mean

    if return_gradients:
        # Compute gradients
        gradients = []
        for i in range(batch_size):
            if layouts.grad is not None:
                layouts.grad.zero_()
            
            y_pred[i].backward(retain_graph=(i < batch_size-1))
            
            if layouts.grad is not None:
                grad = layouts.grad[i].detach().cpu().numpy()
                gradients.append(grad.flatten())  # Flatten to match expected format
            else:
                gradients.append(np.zeros(1056))
        
        gradients = np.stack(gradients)  # [B,1056]
        objectives = y_pred.detach().cpu().numpy()
        return objectives.reshape(-1), gradients
    else:
        objectives = y_pred.detach().cpu().numpy()
        return objectives.reshape(-1)

# ---------- Diffusion-driven generator ----------

@torch.no_grad()
def generate_batch_with_diffusion(
    model,
    batch_size=16,
    num_inference_steps=50,
):
    """
    Returns:
      tris: list of torch [1,33,32] (trinary {-1,0,1})
      cont: list of torch [1,33,32] (continuous in [-1,1])  # optional to keep
    """
    cont, tris = sample_layout(
        model=model,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
    )
    # cont, tris: torch [B,1,33,32]
    return [tris[i] for i in range(tris.size(0))], [cont[i] for i in range(cont.size(0))]

# ---------- MAP-Elites runner ----------

def run_map_elites_with_diffusion(
    diffusion_model,
    n_iters=500,                # number of outer iterations
    batch_size=10,              # samples per iteration
    guidance_scale=15.0,
    archive_bins=(50, 50),      # grid resolution for features
    feature_ranges=((150, 170), (25, 28)),  # ranges for the two features
    save_dir="me_archive",
    device="cuda"
):
    os.makedirs(save_dir, exist_ok=True)

    initial_model = sample_layout(
        model=diffusion_model,
        batch_size=1,
        num_inference_steps=50,
    )[1][0].cpu().numpy()  # [1,33,32]

    print("initial_model.shape:", initial_model.shape, initial_model.size)

    # 2D feature archive
    archive = GridArchive(
        solution_dim=initial_model.size,
        dims=archive_bins,
        ranges=list(feature_ranges)
    )

    noise_scheduler = DDPMScheduler()
    noise_scheduler.set_timesteps(30)

    emitters = [
        DiffusionGuidedEmitter(
            archive=archive,
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            x0=initial_model.flatten(),
            sigma0=0.2,
            lr=0.05,
            ranker="imp",
            selection_rule="mu",
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            diffusion_model_shape=initial_model.shape,
            device=device
        )
    ]

    scheduler = Scheduler(archive, emitters)
    min_features = np.array([np.inf for r in feature_ranges])
    max_features = np.array([-np.inf for r in feature_ranges])
    # map elites loop
    for it in trange(1, n_iters + 1):

        # Gradient phase
        sols = scheduler.ask_dqd()
        # sols has shape (num_emitters, solution_dim)
        obj, jacobian_obj = objective_fn(sols, return_gradients=True)
        measure, jacobian_measure = behavior_fn(sols, return_gradients=True)
        jacobian_obj = jacobian_obj.reshape(jacobian_obj.shape[0], 1, jacobian_obj.shape[1])
        jacobian = np.concatenate((jacobian_obj, jacobian_measure), axis=1)
        scheduler.tell_dqd(obj, measure, jacobian)

        sols = scheduler.ask()
        objs = objective_fn(sols, return_gradients=False) # (B,)
        bcs  = behavior_fn(sols, return_gradients=False)  # (B, 2)

        for b in range(bcs.shape[1]):
            min_features[b] = min(min_features[b], bcs[:, b].min())
            max_features[b] = max(max_features[b], bcs[:, b].max())
        scheduler.tell(objs, bcs)

        if it % 10 == 0:
            tqdm.write(f"  - Size: {archive.stats.num_elites}")    # Number of elites in the archive. len(archive) also provides this info.
            tqdm.write(f"  - Coverage: {archive.stats.coverage}")  # Proportion of archive cells which have an elite.
            print(f"Objective: {archive.best_elite['objective']}")
            print(
                f"Measures: (1: {archive.best_elite['measures'][0]}, 2: {archive.best_elite['measures'][1]})"
            )
    print("MAP-Elites complete. Saved archive to", save_dir)
    print("Feature ranges observed:")
    for i, (min_f, max_f) in enumerate(zip(min_features, max_features)):
        print(f"  - Feature {i}: [{min_f}, {max_f}]")
    return archive

# ---------- Main function demonstrating usage ----------

def main():
    """
    Example of how to load the trained diffusion model and run MAP-Elites.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the surrogate model first
    try:
        load_surrogate_model(device=device)
        print("Successfully loaded surrogate model")
    except Exception as e:
        print(f"Warning: Could not load surrogate model ({e}), will use heuristic objective")
    
    # Load the diffusion model
    try:
        model, dataset, checkpoint_path = load_latest_checkpoint(
            model_dir="qd_diffusion_model_uncond",  # or "reward_conditioned_diffusion_model_v2"
            device=device
        )
        print(f"Successfully loaded diffusion model from: {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a diffusion model first using train_qd_diffusion_model.py")
        return
    
    # Option 2: Load a specific checkpoint (alternative)
    # model, dataset, norm_params = load_trained_diffusion_model(
    #     "qd_diffusion_model/checkpoint-10",  # specify exact checkpoint
    #     device=device,
    #     use_reward=False
    # )
    
    # Run MAP-Elites with the loaded diffusion model
    archive = run_map_elites_with_diffusion(
        diffusion_model=model,
        n_iters=500,
        batch_size=10,                 # adjust based on your GPU memory
        guidance_scale=15.0,
        archive_bins=(50, 50),
        feature_ranges=((150, 200), (25, 27)),
        save_dir="me_archive_demo",
        device=device
    )
    
    print("MAP-Elites completed successfully!")
    print(f"Archive contains {len(archive)} entries")
    print(f"Objective: {archive.best_elite['objective']}")
    print(
        f"Measures: (1: {archive.best_elite['measures'][0]}, 2: {archive.best_elite['measures'][1]})"
    )

if __name__ == "__main__":
    main()
