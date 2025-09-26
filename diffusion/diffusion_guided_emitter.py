import numpy as np
from ribs.emitters import GradientArborescenceEmitter
import torch


class DiffusionGuidedEmitter(GradientArborescenceEmitter):    
    def __init__(self,
        archive,
        diffusion_model, # Diffusion model to use for sampling
        noise_scheduler, # Noise scheduler for the diffusion model
        x0, # Initial central point
        lr: float, # Learning rate for the optimizer
        guidance_scale: float, # How strongly to apply gradient guidance
        sigma0: float = 0.1, # Sigma if using Gaussian sampling
        bounds=None, # Bounds handling is complex with diffusion
        seed=None,
        diffusion_model_shape=None, # Shape of the input to diffusion model
        input_pad_len=0, # Padding for the input to diffusion model
        device='cuda',
        **kwargs): # Other diffusion model params
        
        super().__init__(archive, x0=x0, lr=lr, sigma0=sigma0, seed=seed, **kwargs)

        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.guidance_scale = guidance_scale
        self.proximity_scale = 10.0
        self.bounds = bounds
        self.input_shape = diffusion_model_shape
        self.input_pad_len = input_pad_len
        self.device = device

    def _denoising_step(self, x, grad):
        # these are the solutions sampled from archive to be updated
        current_sols = torch.tensor(x, dtype=torch.float32).to(device=self.device)
        current_sols = current_sols.unsqueeze(0)
        current_sols = current_sols.repeat(grad.shape[0], 1, 1)

        if self.input_pad_len > 0:
            padding = torch.zeros((current_sols.shape[0], 1, self.input_pad_len), dtype=current_sols.dtype, device=current_sols.device)
            current_sols = torch.cat([current_sols, padding], dim=-1)

        if self.input_shape is not None:
            current_sols = current_sols.view(-1, *self.input_shape)
        
        # add noise to the current solutions
        # x = torch.randn_like(current_sols, device=self.diffusion_model.device)
        noise = torch.randn_like(current_sols)
        timesteps = self.noise_scheduler.timesteps
        t_start = timesteps[len(timesteps) // 2]  # midpoint = reasonable noise level
        timesteps_tensor = torch.tensor([t_start] * current_sols.shape[0], dtype=torch.int32).to(current_sols.device)
        x = self.noise_scheduler.add_noise(current_sols, noise=noise, timesteps=timesteps_tensor)

        grad = torch.tensor(grad).to(device=self.device)
        grad = grad.unsqueeze(1)

        if self.input_pad_len > 0:
            padding = torch.zeros((grad.shape[0], 1, self.input_pad_len), dtype=grad.dtype, device=grad.device)
            grad = torch.cat([grad, padding], dim=-1)

        if self.input_shape is not None:
            grad = grad.view(-1, *self.input_shape)

        # Determine range of timesteps to denoise from t_start to 0
        t_index_start = (timesteps == t_start).nonzero(as_tuple=True)[0].item()
        timesteps_to_use = timesteps[t_index_start:]

        for t in timesteps_to_use:
            x = x.detach().requires_grad_()
            x = x.to(torch.float32)

            # Predict noise
            residual = self.diffusion_model(x, t)

            # Apply gradient guidance
            residual = residual + self.guidance_scale * grad + self.proximity_scale * (x - current_sols)
            
            x = self.noise_scheduler.step(residual, t, x).prev_sample

        x = x.squeeze(1)
        if self.input_pad_len > 0:
            x = x[:, :x.shape[-1] - self.input_pad_len]
        else:
            x = x.view(x.shape[0], -1)
        return x

    def ask(self):
        """Samples new solutions from a diffusion model with gradient guidance from
        both the objective and measure gradients.

        The multivariate Gaussian is parameterized by the evolution strategy
        optimizer ``self._opt`` which samples new central points.

        This method returns ``batch_size`` solutions, even though one solution
        is returned via ``ask_dqd``.

        Returns:
            (:attr:`batch_size`, :attr:`solution_dim`) array -- a batch of new
            solutions to evaluate.
        Raises:
            RuntimeError: This method was called without first passing gradients
                with calls to ask_dqd() and tell_dqd().
        """
        if self._jacobian_batch is None:
            raise RuntimeError("Please call ask_dqd() and tell_dqd() "
                               "before calling ask().")

        grad_coeffs = self._opt.ask()[:, :, None]

        new_sols = self._denoising_step(self._grad_opt.theta, np.sum(self._jacobian_batch * grad_coeffs, axis=1))
        return new_sols.detach().cpu().numpy()