import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
import random


def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seeds set to {seed} for reproducibility")

set_seeds()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_grid_from_map_file(map_number):
    path = f"map_generation/maps/kiva_large_w_mode_grid_{map_number:05d}.map"
    if not os.path.exists(path):
        print(f"Warning: {path} not found"); return None
    with open(path, "r") as f: lines = f.readlines()
    grid_lines = lines[4:]
    return ''.join([line.strip() for line in grid_lines])

def convert_grid_string_to_tensor(grid_str, rows=33, cols=36):
    grid_tensor = torch.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(grid_str):
                ch = grid_str[idx]
                if ch == '@':   grid_tensor[i, j] =  1.0  # shelves
                elif ch == 'e': grid_tensor[i, j] = -1.0  # endpoints
                elif ch in ('.','w'): grid_tensor[i, j] = 0.0
    return grid_tensor[:, 2:-2]  # (33, 32)

class GridDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file).dropna(subset=['throughput'])  # throughput unused now
        self.samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading grids"):
            gs = load_grid_from_map_file(row["grid_number"])
            if gs is None:
                continue
            g = convert_grid_string_to_tensor(gs)  # (33, 32) in {-1,0,1}
            self.samples.append(g)
        if not self.samples:
            raise ValueError("No valid grids loaded!")
        self.targets = torch.stack(self.samples)   # (N,33,32)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx):
        target = self.targets[idx].unsqueeze(0)    # (1,33,32)
        return target
    
class UnconditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=(33, 32),
            in_channels=1, out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D"),
            up_block_types=("UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"),
        )
        self.multiple = 2 ** 4  # 16

    def _pad_to_multiple(self, x, m):
        _, _, h, w = x.shape
        pad_h = (m - (h % m)) % m
        pad_w = (m - (w % m)) % m
        top = pad_h // 2; bottom = pad_h - top
        left = pad_w // 2; right = pad_w - left
        x_pad = F.pad(x, (left, right, top, bottom))
        return x_pad, (top, bottom, left, right), (h, w)

    def forward(self, x_noisy, timesteps):
        x_pad, pads, orig = self._pad_to_multiple(x_noisy, self.multiple)
        top, bottom, left, right = pads
        H, W = orig
        out = self.unet(x_pad, timesteps, return_dict=False)[0]
        return out[..., top:top+H, left:left+W]
    
def train_unconditional_diffusion(
    dataset_path,
    output_dir="uncond_layout_diffusion",
    batch_size=16,
    num_epochs=10,
    learning_rate=1e-4,
    save_every=5,
):
    os.makedirs(output_dir, exist_ok=True)

    ds = GridDataset(dataset_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = UnconditionalUNet().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        running = []
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for target in pbar:
            target = target.to(device)  # [B,1,33,32]

            noise = torch.randn_like(target)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (target.shape[0],), device=device).long()
            x_noisy = noise_scheduler.add_noise(target, noise, t)

            eps_pred = model(x_noisy, t)
            loss = F.mse_loss(eps_pred, noise)

            opt.zero_grad(); loss.backward(); opt.step()
            running.append(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        avg = float(np.mean(running)); losses.append(avg)
        print(f"Epoch {epoch+1}: loss={avg:.6f}")

        if ((epoch+1) % save_every == 0) or (epoch == num_epochs-1):
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.unet.save_pretrained(ckpt_dir)
            print("Saved", ckpt_dir)

    plt.figure(figsize=(8,4)); plt.plot(losses); plt.title("Training loss"); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png")); plt.close()
    return model, ds

@torch.no_grad()
def sample_layout(
    model,
    batch_size=1,
    num_inference_steps=50,
):
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps)

    x = torch.randn(batch_size, 1, 33, 32, device=device)
    for t in tqdm(scheduler.timesteps, desc="Sampling"):
        tt = torch.full((batch_size,), t, device=device, dtype=torch.long)
        eps = model(x, tt)
        x = scheduler.step(eps, t, x).prev_sample

    # Trinarize exactly like before
    tris = []
    for i in range(batch_size):
        normalized = (x[i] + 1.0) / 2.0
        flat = normalized.view(-1)
        _, top_idx = torch.topk(flat, k=240)               # shelves
        _, bot_idx = torch.topk(flat, k=350, largest=False)  # endpoints
        tri = torch.zeros_like(flat)
        tri[top_idx] = 1.0; tri[bot_idx] = -1.0
        tris.append(tri.view(1, 33, 32))
    tris = torch.stack(tris)
    return x, tris

def main():
    dataset_path = "throughput_results.csv"
    model, dataset = train_unconditional_diffusion(
        dataset_path=dataset_path,
        output_dir="qd_diffusion_model_uncond",
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-4,
    )

    # Example: generate a batch after training
    x_cont, x_tris = sample_layout(model, batch_size=8, num_inference_steps=50)
    os.makedirs("samples_uncond", exist_ok=True)
    for i in range(x_tris.size(0)):
        arr = x_tris[i,0].detach().cpu().numpy()
        plt.figure(); plt.imshow(arr, cmap="gray"); plt.axis("off")
        plt.savefig(f"samples_uncond/sample_{i:02d}.png", bbox_inches="tight", pad_inches=0); plt.close()

if __name__ == "__main__":
    main()