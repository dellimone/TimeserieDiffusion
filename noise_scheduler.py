from typing import Optional

import torch
import torch.nn.functional as F

# ======================== NOISE SCHEDULING ========================

class NoiseScheduler:
    """Handles noise scheduling for diffusion process"""

    def __init__(self, num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 schedule_type: str = 'linear',
                 device: Optional[torch.device] = None):
        self.num_timesteps = num_timesteps
        self.device = device # Store the device

        if schedule_type == 'linear':
            self.betas = self._linear_schedule(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}. Choose 'linear' or 'cosine'.")

        # Move betas to device early
        if self.device is not None:
            self.betas = self.betas.to(self.device)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def _linear_schedule(self, beta_start: float, beta_end: float, num_timesteps: int) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

    def _cosine_schedule(self, num_timesteps: int) -> torch.Tensor:
        s = 0.008
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self,
                 x_start: torch.Tensor,
                 t: torch.Tensor,
                 mask: torch.Tensor,
                 noise: Optional[torch.Tensor] = None):
        """Forward diffusion process - add noise only to target regions"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1)

        # Apply noise only where mask is 0 (target regions)
        noisy_sample = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        # Keep original values where mask is 1 (conditioning regions)
        return x_start * mask + noisy_sample * (1 - mask)
