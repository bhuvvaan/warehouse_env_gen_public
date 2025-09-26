# surrogate_model_dsage_emb.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
import os
import pickle
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
from surrogate.diagnostics import run_all_diagnostics
from collections import deque

def _fixed_workstation_mask(H=33, W=32, dtype=np.float32):
    """
    Pink 'w' mask (NumPy): columns 0 and W-1, on rows where r % 3 == 1.
    Returns an (H, W) array of 0/1 with requested dtype.
    """
    rows = np.arange(H)[:, None]          # (H,1)
    cols = np.arange(W)[None, :]          # (1,W)
    mask_bool = ((cols == 0) | (cols == W - 1)) & ((rows % 3) == 1)
    return mask_bool.astype(dtype)

def _count_components_discrete(shelf_grid_bool: np.ndarray):
    """
    Exact 4-connected component count on a boolean HxW NumPy array.
    Returns a float (np.float32) count.
    """
    assert shelf_grid_bool.dtype == np.bool_ or shelf_grid_bool.dtype == bool, \
        "shelf_grid_bool should be boolean."
    H, W = shelf_grid_bool.shape
    visited = np.zeros_like(shelf_grid_bool, dtype=bool)
    nbrs = [(1,0),(-1,0),(0,1),(0,-1)]
    comp = 0

    for r in range(H):
        for c in range(W):
            if shelf_grid_bool[r, c] and not visited[r, c]:
                comp += 1
                dq = deque([(r, c)])
                visited[r, c] = True
                while dq:
                    rr, cc = dq.popleft()
                    for dr, dc in nbrs:
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W and
                            shelf_grid_bool[nr, nc] and not visited[nr, nc]):
                            visited[nr, nc] = True
                            dq.append((nr, nc))
    return np.float32(comp)

def _average_task_length_discrete(endpoint_mask: np.ndarray, w_mask: np.ndarray):
    """
    Average Manhattan distance between all endpoint cells and all 'w' cells.
    Inputs are (H,W) arrays; they can be boolean or {0,1} floats/ints.
    Returns a float (np.float32).
    """
    ep_idx = np.argwhere(endpoint_mask > 0.5 if endpoint_mask.dtype != bool else endpoint_mask)
    w_idx  = np.argwhere(w_mask > 0.5 if w_mask.dtype != bool else w_mask)

    if ep_idx.size == 0 or w_idx.size == 0:
        return np.float32(0.0)

    # Broadcast to pairwise L1 distances: (Ne,1) vs (1,Nw)
    dists = (np.abs(ep_idx[:, None, 0] - w_idx[None, :, 0]) +
             np.abs(ep_idx[:, None, 1] - w_idx[None, :, 1]))
    return np.float32(dists.mean())

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
        self.act   = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(x + h)


class S1Repair(nn.Module):
    """Conv(3x3) -> LeakyReLU -> 2xResBlocks -> Conv(1x1) to K logits."""
    def __init__(self, in_ch: int, mid_ch: int, num_classes: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.res1 = ResidualBlock(mid_ch)
        self.res2 = ResidualBlock(mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, num_classes, 1)
    def forward(self, x):  # x: [B,emb_dim,H,W]
        h = self.act(self.conv_in(x))
        h = self.res1(h)
        h = self.res2(h)
        return self.conv_out(h)  # [B,K,H,W]

class S2Usage(nn.Module):
    """Same form; input is repaired one-hot [B,K,H,W]; output per-tile usage logits."""
    def __init__(self, in_ch: int, mid_ch: int):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.res1 = ResidualBlock(mid_ch)
        self.res2 = ResidualBlock(mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, 1, 1)
    def forward(self, x):  # [B,K,H,W]
        h = self.act(self.conv_in(x))
        h = self.res1(h)
        h = self.res2(h)
        return self.conv_out(h)  # [B,1,H,W]

class S3Head(nn.Module):
    """Stride-2 4x4 convs until ~4x4, then FC->FC to 1+M scalars."""
    def __init__(self, in_ch: int, in_hw: Tuple[int,int],
                 base_ch: int = 64, max_ch: int = 512, num_outputs: int = 1, extra_dim: int = 0):
        super().__init__()
        H, W = in_hw
        layers = []
        ch = base_ch
        layers += [nn.Conv2d(in_ch, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.1, inplace=True)]
        h = (H+2*1-4)//2 + 1
        w = (W+2*1-4)//2 + 1
        while min(h, w) > 4:
            ch2 = min(max_ch, ch*2)
            layers += [nn.Conv2d(ch, ch2, 4, 2, 1),
                       nn.BatchNorm2d(ch2),
                       nn.LeakyReLU(0.1, inplace=True)]
            ch = ch2
            h = (h+2*1-4)//2 + 1
            w = (w+2*1-4)//2 + 1
        self.backbone = nn.Sequential(*layers)
        self.flatten  = nn.Flatten()
        
        # adjust FC input dimension: conv features + optional extra
        self.feat_dim = ch * h * w
        self.extra_dim = extra_dim
        self.fc1 = nn.Linear(self.feat_dim + extra_dim, 256)
        self.fc2 = nn.Linear(256, num_outputs)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, extra_global: torch.Tensor | None = None):
        """
        x: [B,K+1,H,W]
        extra_global: [B,D] or None
        """
        h = self.backbone(x)
        h = self.flatten(h)   # [B, feat_dim]
        if extra_global is not None:
            assert extra_global.shape[0] == h.shape[0], "batch size mismatch"
            h = torch.cat([h, extra_global], dim=1)  # concat along feature dim
        h = self.act(self.fc1(h)); h = self.drop(h)
        return self.fc2(h)  # [B,num_outputs]

@dataclass
class SurrogateOutputs:
    s1_logits: torch.Tensor   # [B,K,H,W]
    s1_onehot: torch.Tensor   # [B,K,H,W] (hard, detached)
    s2_logits: torch.Tensor   # [B,1,H,W]
    s2_prob: torch.Tensor     # [B,1,H,W] normalized over HW
    y_hat: torch.Tensor         # [B, 1+M]
    y_hat_throughput: torch.Tensor  # [B,1]
    y_hat_measures: torch.Tensor    # [B,M]

class SurrogateModelEmb(nn.Module):
    """
    Inputs:
      x_ids: [B,H,W] (long), class ids in [0..K-1]
    Pipeline:
      embed(x_ids) -> S1 -> hard one-hot -> S2 -> normalized usage -> concat -> S3
    """
    def __init__(self, grid_shape=(33,32), num_classes=3, emb_dim=16,
                 mid_ch=64, base_ch=64, num_measures=2, s3_extra_dim=0):
        super().__init__()
        self.num_measures = num_measures
        self.K = num_classes
        self.embedding = nn.Embedding(num_classes, emb_dim)  # paper-style categorical embedding
        self.s1 = S1Repair(in_ch=emb_dim, mid_ch=mid_ch, num_classes=num_classes)
        self.s2 = S2Usage(in_ch=num_classes, mid_ch=mid_ch)
        self.s3 = S3Head(in_ch=num_classes+1, in_hw=grid_shape, base_ch=base_ch, num_outputs=1+num_measures, extra_dim=s3_extra_dim)
        self.class_mapping = {0:"endpoint", 1:"empty", 2:"shelf"}

    @staticmethod
    def _ids_to_emb_feats(embedding: nn.Embedding, x_ids: torch.Tensor) -> torch.Tensor:
        # x_ids: [B,H,W] -> embed -> [B,H,W,D] -> permute to [B,D,H,W]
        feats = embedding(x_ids)                 # [B,H,W,D]
        return feats.permute(0,3,1,2).contiguous()

    @torch.no_grad()
    def _hw_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape  # C=1
        flat = logits.view(B, -1)
        prob = F.softmax(flat, dim=1).view(B, 1, H, W)
        return prob

    @torch.no_grad()
    def _hard_one_hot_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        pred = logits.argmax(dim=1)                  # [B,H,W]
        return F.one_hot(pred, num_classes=self.K).permute(0,3,1,2).float()  # [B,K,H,W]

    def forward(self, x_ids: torch.Tensor, extra_global: torch.Tensor | None = None) -> SurrogateOutputs:
        # x_ids: [B,H,W] (long)
        emb = self._ids_to_emb_feats(self.embedding, x_ids)   # [B,emb_dim,H,W]

        s1_logits = self.s1(emb)                              # [B,K,H,W]
        s1_oh = self._hard_one_hot_from_logits(s1_logits).detach()

        s2_logits = self.s2(s1_oh)                            # [B,1,H,W]
        s2_prob   = self._hw_softmax(s2_logits)               # normalized usage distribution

        if extra_global is None:
            with torch.no_grad():                                 # don’t backprop into s1/s2 via globals
                extra_global = self._global3_from_s1s2(s1_oh, s2_prob)  # [B,3]

        x_head = torch.cat([s1_oh, s2_prob], dim=1)           # [B,K+1,H,W]
        y_all = self.s3(x_head, extra_global=extra_global)    # [B, 1+M]

        thr = y_all[:, :1]                                    # [B,1]
        meas = y_all[:, 1:] if self.num_measures > 0 else y_all.new_zeros(y_all.size(0), 0)
        return SurrogateOutputs(s1_logits, s1_oh, s2_logits, s2_prob, y_all, thr, meas)

    def _global3_from_s1s2(self, s1_oh: torch.Tensor, s2_prob: torch.Tensor) -> torch.Tensor:
        """
        s1_oh  : [B,3,H,W] repaired hard one-hot, class i aligned with self.class_mapping
        s2_prob: [B,1,H,W] normalized usage (sum over H*W = 1)
        returns: [B,3] = [shelf_frac, endpoint_frac, usage_entropy]
        """
        assert s1_oh.dim()==4 and s1_oh.size(1)==3, "s1_oh must be [B,3,H,W]"
        assert s2_prob.dim()==4 and s2_prob.size(1)==1, "s2_prob must be [B,1,H,W]"

        shelf_ch    = [k for k,v in self.class_mapping.items() if v=="shelf"][0]
        endpoint_ch = [k for k,v in self.class_mapping.items() if v=="endpoint"][0]

        B, _, H, W = s1_oh.shape
        shelf_frac    = s1_oh[:, shelf_ch].mean(dim=(1,2), keepdim=False)     # [B,1]
        endpoint_frac = s1_oh[:, endpoint_ch].mean(dim=(1,2), keepdim=False)  # [B,1]
        shelf_frac = shelf_frac.unsqueeze(1)        # [B,1]
        endpoint_frac = endpoint_frac.unsqueeze(1)

        p = s2_prob.clamp_min(1e-12)                                         # [B,1,H,W]
        ent = -(p * p.log()).sum(dim=(1,2,3), keepdim=False) / math.log(H*W)  # [B,1]
        ent = ent.unsqueeze(1)  # [B,1]

        return torch.cat([shelf_frac, endpoint_frac, ent], dim=1)            # [B,3]
    
    def forward_s1_logits(self, x_ids: torch.Tensor) -> torch.Tensor:
        emb = self._ids_to_emb_feats(self.embedding, x_ids)
        s1_logits = self.s1(emb)                              # [B,K,H,W]
        return s1_logits
    
    def forward_s1(self, x_ids: torch.Tensor) -> torch.Tensor:
        s1_logits = self.forward_s1_logits(x_ids)
        s1_oh = self._hard_one_hot_from_logits(s1_logits).detach()
        return s1_oh
    
    def forward_s2_from_s1(self, s1_oh: torch.Tensor) -> torch.Tensor:
        s2_logits = self.s2(s1_oh)                            # [B,1,H,W]
        s2_prob   = self._hw_softmax(s2_logits)               # normalized usage distribution
        return s2_prob

def loss_s1_cross_entropy(s1_logits, y_repaired_ids):
    return F.cross_entropy(s1_logits, y_repaired_ids)  # per-tile CE

def loss_s2_kl(s2_logits, usage_target_prob):
    B,_,H,W = s2_logits.shape
    logp = F.log_softmax(s2_logits.view(B,-1), dim=1)         # [B,HW]
    tgt  = usage_target_prob.view(B,-1).clamp_min(1e-8)       # [B,HW], sum=1
    return F.kl_div(logp, tgt, reduction="batchmean")

def loss_s3_mse(y_hat, y_true):
    return F.mse_loss(y_hat, y_true)

def loss_s3_measures_mse(y_meas_pred, y_meas_true):
    return F.mse_loss(y_meas_pred, y_meas_true)

def compute_measures_from_float_grid(grid_float):
    """
    grid_float: (H,W) with values in {-1,0,1}
      shelves=1, endpoints=-1, empty=0
    returns: np.array([num_components, avg_task_len], dtype=np.float32)
    """
    H, W = grid_float.shape
    shelves = (grid_float == 1.0)
    endpoints = (grid_float == -1.0)
    w_mask = _fixed_workstation_mask(H, W).astype(bool)
    comps = _count_components_discrete(shelves)
    avgL  = _average_task_length_discrete(endpoints, w_mask)
    return np.array([comps, avgL], dtype=np.float32)

def compute_measures_batch(grids_float):
    """
    grids_float: (N,H,W) floats in {-1,0,1}
    returns: (N,2) float32
    """
    return np.stack([compute_measures_from_float_grid(grids_float[i]) for i in range(grids_float.shape[0])], axis=0)

@torch.no_grad()
def predict(model: SurrogateModelEmb, x_ids: torch.Tensor):
    model.eval()
    outs = model(x_ids.to(next(model.parameters()).device))
    repaired_ids = outs.s1_logits.argmax(dim=1)  # [B,H,W]
    return {
        "repaired_ids": repaired_ids,
        "usage_prob": outs.s2_prob,
        "scalars": outs.y_hat,                    # [B,1+M] (z-space)
        "throughput": outs.y_hat_throughput,     # [B,1]
        "measures": outs.y_hat_measures,         # [B,M]
    }

def load_unified_data_splits(splits_dir='unified_data_splits'):
    """Load the unified data splits created by unified_data_preparation.py"""
    print(f"Loading unified data splits from {splits_dir}...")
    
    # Check if pickle file exists
    splits_file = os.path.join(splits_dir, 'data_splits.pkl')
    if os.path.exists(splits_file):
        with open(splits_file, 'rb') as f:
            splits = pickle.load(f)
        print("Loaded data splits from pickle file")
    else:
        # Fallback to numpy files
        print("Pickle file not found, loading from numpy files...")
        splits = {
            'train': {
                'grids': np.load(os.path.join(splits_dir, 'train_grids.npy')),
                'heatmaps': np.load(os.path.join(splits_dir, 'train_heatmaps.npy')),
                'throughputs': np.load(os.path.join(splits_dir, 'train_throughputs.npy')),
                'grid_ids': np.load(os.path.join(splits_dir, 'train_grid_ids.npy'), allow_pickle=True)
            },
            'test': {
                'grids': np.load(os.path.join(splits_dir, 'test_grids.npy')),
                'heatmaps': np.load(os.path.join(splits_dir, 'test_heatmaps.npy')),
                'throughputs': np.load(os.path.join(splits_dir, 'test_throughputs.npy')),
                'grid_ids': np.load(os.path.join(splits_dir, 'test_grid_ids.npy'), allow_pickle=True)
            }
        }
    
    # Load metadata
    metadata_file = os.path.join(splits_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata: {metadata}")
    
    print(f"Training set: {len(splits['train']['grids'])} samples")
    print(f"Test set: {len(splits['test']['grids'])} samples")
    
    return splits


def to_float_grid_minus1_0_1(grids):
    """
    Accepts:
      - (N,H,W) floats in {-1,0,1}, or
      - (N,2,H,W) with channel0=shelves, channel1=endpoints (binary masks)
    Returns: (N,H,W) float32 in {-1,0,1}
    """
    if grids.ndim == 3:
        return grids.astype(np.float32)
    if grids.ndim == 4 and grids.shape[1] == 2:
        shelves = (grids[:, 0] > 0.5).astype(np.float32)
        endpoints = (grids[:, 1] > 0.5).astype(np.float32)
        empty = 1.0 - np.clip(shelves + endpoints, 0, 1)
        return shelves*1.0 + endpoints*(-1.0) + empty*0.0
    raise ValueError(f"Unsupported grids shape {grids.shape}")

def floats_to_class_ids(grid_float, mapping={-1.0:0, 0.0:1, 1.0:2}):
    out = np.zeros_like(grid_float, dtype=np.int64)
    for val, cid in mapping.items():
        out[grid_float == val] = cid
    return out

# ---------------------------
# 3) make “unrepaired” by light corruption
# ---------------------------
def corrupt_layout_ids(class_ids, flip_frac=0.02, rng=None):
    """
    class_ids: (H,W) with K=3 classes (0:endpoint,1:empty,2:shelf).
    flip_frac: fraction of tiles to modify.
    mild class-transition bias is used inside.
    """
    if rng is None: rng = np.random.default_rng()
    H,W = class_ids.shape; K = 3
    out = class_ids.copy().reshape(-1)

    n_flip = int(round(flip_frac * out.size))
    if n_flip <= 0: return out.reshape(H,W)

    # sample positions to flip
    idx = rng.choice(out.size, size=n_flip, replace=False)
    cur = out[idx]

    # transition probabilities per current class
    # endpoints->mostly stay, else to empty; shelves->mostly stay, else to empty; empty->stay, small to others
    T = np.array([
        [0.80, 0.18, 0.02],  # from endpoint
        [0.10, 0.80, 0.10],  # from empty
        [0.02, 0.18, 0.80],  # from shelf
    ], dtype=np.float32)

    new_vals = np.empty_like(cur)
    for i, c in enumerate(cur):
        new_vals[i] = rng.choice(np.arange(K), p=T[c])
    out[idx] = new_vals
    return out.reshape(H,W)

# ---------------------------
# 4) pack supervised pairs for the surrogate
# ---------------------------
def build_pairs(grids_float, heatmaps, throughputs,
                num_unrepaired_per_layout=2, flip_frac=0.02, rng=None):
    """
    Treat 'grids_float' as REPAIRED targets (y). We synthesize unrepaired x by corrupting y.
    Returns x_ids, y_ids, usage, y_vec
    """
    if rng is None: rng = np.random.default_rng()
    N,H,W = grids_float.shape

    y_ids = floats_to_class_ids(grids_float)  # repaired ints (N,H,W)

    # normalize heatmaps to sum=1, add channel
    hm = heatmaps.astype(np.float32).reshape(N, -1)
    s = np.clip(hm.sum(axis=1, keepdims=True), 1e-8, None)
    usage = (hm/s).reshape(N, 1, H, W).astype(np.float32)

    thr = throughputs.reshape(N,1).astype(np.float32)

    xs, ys, us, yv = [], [], [], []
    for i in range(N):
        for _ in range(num_unrepaired_per_layout):
            x_ids = corrupt_layout_ids(y_ids[i], flip_frac=flip_frac, rng=rng)
            xs.append(x_ids); ys.append(y_ids[i]); us.append(usage[i]); yv.append(thr[i])

    return (np.stack(xs).astype(np.int64),
            np.stack(ys).astype(np.int64),
            np.stack(us).astype(np.float32),
            np.stack(yv).astype(np.float32))

def zscore_on_train(train_y, *others):
    mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
    std  = train_y.std(axis=0, keepdims=True).astype(np.float32)
    std  = np.clip(std, 1e-8, None)
    def z(x): return (x - mean) / std
    return mean, std, (z(train_y),) + tuple(z(o) for o in others)

# ---------------------------
# 5) dataset for SurrogateModelEmb
# ---------------------------
class SurrogatePairsDataset(Dataset):
    def __init__(self, x_ids, y_ids, usage, y_vec, y_meas):
        self.x_ids = torch.from_numpy(x_ids).long()   # (N,H,W)
        self.y_ids = torch.from_numpy(y_ids).long()   # (N,H,W)
        self.usage = torch.from_numpy(usage).float()  # (N,1,H,W), sum=1
        self.y_vec = torch.from_numpy(y_vec).float()  # (N,1) z-scored
        self.y_meas = torch.from_numpy(y_meas).float()  # (N,1) z-scored
    def __len__(self): return self.x_ids.shape[0]
    def __getitem__(self, i):
        return {"x_ids": self.x_ids[i],
                "y_ids": self.y_ids[i],
                "usage": self.usage[i],
                "y_vec": self.y_vec[i],
                "y_meas": self.y_meas[i]}

def global3_from_s1s2(
    s1_oh: torch.Tensor, 
    s2_prob: torch.Tensor, 
    mapping: dict = {0: "endpoint", 1: "empty", 2: "shelf"}
) -> torch.Tensor:
    """
    Build 3-dim global vector from *repaired* layout and predicted usage:
      [shelf_frac, endpoint_frac, usage_entropy]

    Args:
      s1_oh   : [B, 3, H, W] hard one-hot of repaired classes
                channel i corresponds to class index i, where mapping gives the semantic.
      s2_prob : [B, 1, H, W] normalized usage distribution (sum over H*W = 1)
      mapping : dict from class index -> {"empty","shelf","endpoint"}

    Returns:
      global3 : [B, 3] = [shelf_frac, endpoint_frac, usage_entropy]
    """
    assert s1_oh.dim() == 4 and s1_oh.size(1) == 3, "s1_oh must be [B,3,H,W]"
    assert s2_prob.dim() == 4 and s2_prob.size(1) == 1, "s2_prob must be [B,1,H,W]"

    # find channel indices for each semantic class
    shelf_ch    = [k for k,v in mapping.items() if v == "shelf"][0]
    endpoint_ch = [k for k,v in mapping.items() if v == "endpoint"][0]
    # empty_ch  = [k for k,v in mapping.items() if v == "empty"][0]  # not needed here

    B, _, H, W = s1_oh.shape

    # fractions from repaired layout
    shelf_frac    = s1_oh[:, shelf_ch].mean(dim=(1,2), keepdim=False)     # [B,1]
    endpoint_frac = s1_oh[:, endpoint_ch].mean(dim=(1,2), keepdim=False)  # [B,1]
    shelf_frac    = shelf_frac.unsqueeze(1)        # [B,1]
    endpoint_frac = endpoint_frac.unsqueeze(1)   # [B,1]

    # entropy of usage (normalized by log(HW) so roughly in [0,1])
    p = s2_prob.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=(1,2,3), keepdim=False) / math.log(H * W)  # [B,1]
    ent = ent.unsqueeze(1)  # [B,1]

    return torch.cat([shelf_frac, endpoint_frac, ent], dim=1)  # [B,3]

@torch.no_grad()
def stage_metrics(stage, model, batch, device, *, norm_stats=None, extra_global_fn=None):
    """
    Compute validation metrics (no backprop).
    - stage=1: repair accuracy
    - stage=2: usage KL + L1 alignment
    - stage=3: throughput MSE (z-space) and RMSE (optionally in original space if norm_stats provided)
    extra_global_fn: callable(batch, device) -> [B,D] if you use global features in s3
    """
    device = torch.device(device)
    model.eval()
    with torch.no_grad():
        x_ids = batch["x_ids"].to(device).long()

        if stage == 1:
            # s1 only
            s1_logits = model.forward_s1_logits(x_ids)            # [B,K,H,W]
            y_ids = batch["y_ids"].to(device).long()
            loss = loss_s1_cross_entropy(s1_logits, y_ids).item()
            pred = s1_logits.argmax(dim=1)                        # [B,H,W]
            acc = (pred == y_ids).float().mean().item()
            return {"loss": loss, "acc": acc}

        elif stage == 2:
            # s1 frozen -> s2 logits
            s1_oh     = model.forward_s1(x_ids)                   # [B,K,H,W], detached internally
            s2_logits = model.s2(s1_oh)                           # [B,1,H,W]
            s2_prob   = model._hw_softmax(s2_logits)              # [B,1,H,W]

            usage = batch["usage"].to(device).float()             # [B,1,H,W], sum=1
            loss = loss_s2_kl(s2_logits, usage).item()
            l1   = (s2_prob - usage).abs().mean().item()
            return {"loss": loss, "l1": l1}

        else:
            # stage == 3: s1,s2 frozen -> s3
            s1_oh   = model.forward_s1(x_ids)                     # [B,K,H,W]
            s2_prob = model.forward_s2_from_s1(s1_oh)             # [B,1,H,W]

            # build extra globals (same as training)
            if extra_global_fn is None:
                extra_global = global3_from_s1s2(s1_oh, s2_prob, mapping={0:"endpoint",1:"empty",2:"shelf"})
            else:
                extra_global = extra_global_fn(batch, device)               # [B,D]

            y_vec = batch["y_vec"].to(device).float()              # [B,1] in z-space
            y_meas = batch["y_meas"].to(device).float()      # [B,M] z-scored measures
            x_head = torch.cat([s1_oh, s2_prob], dim=1)            # [B,K+1,H,W]

            y_all = model.s3(x_head, extra_global=extra_global) # [B,1+M]
            thr_pred  = y_all[:, :1]
            meas_pred = y_all[:, 1:]

            loss_thr  = loss_s3_mse(thr_pred, y_vec).item()
            loss_meas = loss_s3_measures_mse(meas_pred, y_meas).item()
            loss = loss_thr + loss_meas

            rmse = math.sqrt(F.mse_loss(thr_pred, y_vec).item())

            metrics = {"loss": loss, "rmse": rmse}

            return metrics

def _freeze_all(model: nn.Module):
    for p in model.parameters(): p.requires_grad_(False)

def _unfreeze(params):
    for p in params: p.requires_grad_(True)

def save_checkpoint(model, optimizer, epoch, stage, save_path, extra=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra: ckpt.update(extra)
    torch.save(ckpt, save_path)
    print(f"💾 Saved best checkpoint → {save_path}")

def counts_from_ids(x_ids: torch.Tensor) -> torch.Tensor:
    # x_ids: [B,H,W] long in {0,1,2}
    oh = torch.nn.functional.one_hot(x_ids, num_classes=3).float()  # [B,H,W,3]
    counts = oh.view(x_ids.size(0), -1, 3).sum(dim=1)               # [B,3]
    counts = counts / float(x_ids.size(1)*x_ids.size(2))
    return counts  # [B,3]

def sharpen_distribution(p, T=0.5, eps=1e-12):
    """
    p: [B,1,H,W] probability map (sums to 1 per sample)
    T < 1 sharpens (makes distribution peakier)
    T > 1 smooths (makes distribution flatter)
    """
    B, C, H, W = p.shape
    p = p.view(B, -1)                   # [B, HW]
    p = (p.clamp_min(eps) ** (1.0 / T)) # temperature scaling
    p = p / p.sum(dim=1, keepdim=True)  # renormalize
    return p.view(B, C, H, W)

# --- main training function with early stopping ---
def train_stage(
    model: nn.Module,
    train_loader,
    val_loader=None,
    *,
    stage: int,
    epochs: int = 20,
    lr: float = 1e-3,
    betas=(0.9,0.99),
    device: str = "cuda",
    ckpt_dir: str = "checkpoints",
    ckpt_name: str | None = None,
    patience: int = 10,                 # early stop patience (epochs without val improvement)
    sched_factor: float = 0.5,         # ReduceLROnPlateau factor
    sched_patience: int = 2,           # epochs without val improvement before LR drop
    norm_stats: dict | None = None,    # e.g., {"y_mean": y_mean, "y_std": y_std}
):
    """
    Trains one stage; saves the best (lowest val loss) checkpoint.
    Returns dict with training history and path to best checkpoint.
    """
    device = torch.device(device)
    model.to(device)

    # freeze/unfreeze per stage
    _freeze_all(model)
    if stage == 1:
        trainable = list(model.s1.parameters())
        if not ckpt_name: ckpt_name = "surrogate_stage1_repair_best.pth"
    elif stage == 2:
        trainable = list(model.s2.parameters())
        if not ckpt_name: ckpt_name = "surrogate_stage2_usage_best.pth"
    elif stage == 3:
        trainable = list(model.s3.parameters())
        if not ckpt_name: ckpt_name = "surrogate_stage3_throughput_best.pth"
    else:
        raise ValueError("stage must be 1, 2, or 3")
    _unfreeze(trainable)

    optimizer = torch.optim.Adam(trainable, lr=lr, betas=betas)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor, patience=sched_patience
    )

    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, ckpt_name)
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs+1):
        # --- train ---
        model.train()
        tot_loss, n_samples = 0.0, 0
        for batch in train_loader:
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)

            x_ids = batch["x_ids"].to(device).long()

            if stage == 1:
                model.train()
                model.s1.train()
                model.s2.eval()
                model.s3.eval()

                y_ids = batch["y_ids"].to(device).long()
                s1_logits = model.forward_s1_logits(x_ids)
                loss = loss_s1_cross_entropy(s1_logits, y_ids)
            elif stage == 2:
                model.train()
                model.s1.eval()
                model.s2.train()
                model.s3.eval()

                usage = batch["usage"].to(device).float()
                B = usage.size(0)
                assert torch.allclose(usage.view(B,-1).sum(dim=1), torch.ones(B, device=usage.device), atol=1e-4)
                usage_sharp = sharpen_distribution(usage, T=2.0)

                # get s1_oh once, frozen
                with torch.no_grad():
                    s1_oh = model.forward_s1(x_ids)
                s2_logits = model.s2(s1_oh)
                loss = loss_s2_kl(s2_logits, usage_sharp)
            elif stage == 3:
                model.train()
                model.s1.eval()
                model.s2.eval()
                model.s3.train()

                y_vec = batch["y_vec"].to(device).float()
                y_meas = batch["y_meas"].to(device).float()
                with torch.no_grad():                     # no grads, no graph, outputs are detached
                    s1_out = model.forward_s1(x_ids)
                    s2_out = model.forward_s2_from_s1(s1_out)
                    global3 = global3_from_s1s2(s1_out, s2_out, mapping={0:"endpoint",1:"empty",2:"shelf"})

                    x_head = torch.cat([s1_out, s2_out], dim=1)         # [B,4,H,W]
                
                y_all = model.s3(x_head, extra_global=global3)               # [B,1+M]
                thr_pred  = y_all[:, :1]
                meas_pred = y_all[:, 1:]

                loss_thr  = loss_s3_mse(thr_pred, y_vec)
                loss_meas = loss_s3_measures_mse(meas_pred, y_meas)
                loss = loss_thr + 0.1 * loss_meas
            
            loss.backward()
            optimizer.step()

            bs = x_ids.size(0)
            tot_loss += loss.item() * bs
            n_samples += bs

        train_loss = tot_loss / max(1, n_samples)
        history["train_loss"].append(train_loss)

        # --- validate ---
        if val_loader is not None:
            model.eval()
            tot_vloss, vn = 0.0, 0
            # optional: aggregate a couple metrics
            agg = {}
            with torch.no_grad():
                for vbatch in val_loader:
                    mets = stage_metrics(stage, model, vbatch, device)
                    bs = vbatch["x_ids"].size(0)
                    tot_vloss += mets["loss"] * bs
                    vn += bs
                    for k,v in mets.items():
                        if k=="loss": continue
                        agg[k] = agg.get(k, 0.0) + v*bs
            val_loss = tot_vloss / max(1, vn)
            history["val_loss"].append(val_loss)
            for k in list(agg.keys()):
                agg[k] /= max(1, vn)

            # report
            if stage == 1:
                print(f"[Stage {stage}] Epoch {ep:03d}/{epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                      f"val_acc={agg.get('acc',0):.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            elif stage == 2:
                print(f"[Stage {stage}] Epoch {ep:03d}/{epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                      f"val_L1={agg.get('l1',0):.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"[Stage {stage}] Epoch {ep:03d}/{epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                      f"val_RMSE={agg.get('rmse',0):.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")

            # scheduler + early stopping
            scheduler.step(val_loss)
            if val_loss + 1e-9 < best_val:
                best_val = val_loss
                no_improve = 0
                # Save checkpoint w/ optional normalization stats
                extra = {"norm_stats": norm_stats} if norm_stats else None
                save_checkpoint(model, optimizer, ep, stage, best_path, extra)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹️  Early stopping (no val improvement for {patience} epochs).")
                    break
        else:
            # no validation loader → just print train loss and save last
            print(f"[Stage {stage}] Epoch {ep:03d}/{epochs}  train_loss={train_loss:.5f}")
            save_checkpoint(model, optimizer, ep, stage, best_path, extra={"norm_stats": norm_stats})

    return {"history": history, "best_path": best_path, "best_val": best_val}

@torch.no_grad()
def evaluate_surrogate_throughput(model, test_loader, device="cuda", norm_stats=None):
    model.eval()
    device = torch.device(device)
    model.to(device)

    y_true, y_pred = [], []
    for batch in test_loader:
        x_ids = batch["x_ids"].to(device).long()
        y_vec = batch["y_vec"].to(device).float()

        out = model(x_ids)
        preds = out.y_hat_throughput.squeeze(1).cpu().numpy()
        targets = y_vec.squeeze(1).cpu().numpy()

        y_pred.append(preds)
        y_true.append(targets)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # De-normalize if you saved normalization stats
    if norm_stats is not None:
        mean, std = norm_stats["y_mean"], norm_stats["y_std"]
        y_true = y_true * std + mean
        y_pred = y_pred * std + mean

    residuals = y_true - y_pred

    mse = np.mean(residuals**2)
    rmse = math.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    corr = np.corrcoef(y_true, y_pred)[0,1]

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": corr,
        "max_error": max_error,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
    }

    print("TEST SET THROUGHPUT PREDICTION METRICS")
    print("="*50)
    print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.6f}")
    print(f"Mean Absolute Error (MAE):     {metrics['mae']:.6f}")
    print(f"R-squared (R²):                {metrics['r2']:.6f}")
    print(f"Correlation Coefficient:       {metrics['correlation']:.6f}")
    print(f"Maximum Error:                 {metrics['max_error']:.6f}")
    print(f"Residual Mean:                 {metrics['residual_mean']:.6f}")
    print(f"Residual Std Dev:              {metrics['residual_std']:.6f}")
    print("="*50)

    return metrics

if __name__ == "__main__":
    splits = load_unified_data_splits()

    # unify grid format to (N,33,32) floats in {-1,0,1}
    train_grids_f = to_float_grid_minus1_0_1(splits['train']['grids'])
    test_grids_f  = to_float_grid_minus1_0_1(splits['test']['grids'])

    # Compute measures from the repaired grids (targets)
    train_meas = compute_measures_batch(train_grids_f)  # (N,2)
    test_meas  = compute_measures_batch(test_grids_f)   # (N,2)

    train_heat = splits['train']['heatmaps']  # (N,33,32)
    test_heat  = splits['test']['heatmaps']
    train_thr  = splits['train']['throughputs']  # (N,)
    test_thr   = splits['test']['throughputs']

    # create (x:unrepaired, y:repaired) pairs
    x_tr, y_tr, u_tr, v_tr = build_pairs(train_grids_f, train_heat, train_thr,
                                         num_unrepaired_per_layout=2, flip_frac=0.1)
    x_te, y_te, u_te, v_te = build_pairs(test_grids_f,  test_heat,  test_thr,
                                         num_unrepaired_per_layout=1, flip_frac=0.1)
    
    # IMPORTANT: replicate measures per synthesized unrepaired sample count
    # train has *2* unrepaired per layout, test has *1*
    rep_tr = x_tr.shape[0] // train_grids_f.shape[0]
    rep_te = x_te.shape[0] // test_grids_f.shape[0]
    y_meas_tr = np.repeat(train_meas, repeats=rep_tr, axis=0).astype(np.float32)  # (N*rep_tr,2)
    y_meas_te = np.repeat(test_meas,  repeats=rep_te, axis=0).astype(np.float32)
    
    # z-score throughput on TRAIN only
    y_mean, y_std, (v_tr_z, v_te_z) = zscore_on_train(v_tr, v_te)

    # datasets & loaders
    train_ds = SurrogatePairsDataset(x_tr, y_tr, u_tr, v_tr_z, y_meas_tr)
    val_ds   = SurrogatePairsDataset(x_te, y_te, u_te, v_te_z, y_meas_te)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

    print("Prepared for SurrogateModelEmb:")
    print("  x_ids:", train_ds.x_ids.shape, "(unrepaired int grids)")
    print("  y_ids:", train_ds.y_ids.shape, "(repaired int grids)")
    print("  usage:", train_ds.usage.shape, "(normalized to sum=1)")
    print("  y_vec:", train_ds.y_vec.shape, "(z-scored throughput)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SurrogateModelEmb(grid_shape=(33,32), num_classes=3, emb_dim=16, mid_ch=64, base_ch=64, num_measures=2, s3_extra_dim=3).to(device)
    norm_stats = {"y_mean": y_mean, "y_std": y_std}

    # Stage 1: repair
    # res1 = train_stage(model, train_loader, val_loader,
    #                 stage=1, epochs=30, lr=1e-3,
    #                 ckpt_dir="checkpoints", patience=10,
    #                 norm_stats=norm_stats)

    # # Stage 2: usage
    # res2 = train_stage(model, train_loader, val_loader,
    #                 stage=2, epochs=30, lr=1e-3,
    #                 ckpt_dir="checkpoints", patience=10,
    #                 norm_stats=norm_stats)

    # # Stage 3: throughput
    # res3 = train_stage(model, train_loader, val_loader,
    #                 stage=3, epochs=30, lr=1e-3,
    #                 ckpt_dir="checkpoints", patience=10,
    #                 norm_stats=norm_stats)

    # print("Best checkpoints:")
    # print(res1["best_path"])
    # print(res2["best_path"])
    # print(res3["best_path"])

    ckpt = torch.load("checkpoints/surrogate_stage3_throughput_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    norm_stats = ckpt.get("norm_stats", None)  # contains y_mean / y_std if you saved them
    print("Restored stage:", ckpt["stage"], "epoch:", ckpt["epoch"])
    metrics = evaluate_surrogate_throughput(model, val_loader, device="cuda", norm_stats=norm_stats)
    results = run_all_diagnostics(model, train_loader, val_loader, device=device, norm_stats=norm_stats, out_prefix="surrogate")
    print(results)
