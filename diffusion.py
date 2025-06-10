import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from noise_scheduler import NoiseScheduler

# ======================== DIFFUSION MODEL ========================

class TimeSeriesDDPM:
    """Main diffusion model class"""

    def __init__(self,
                 denoiser: nn.Module,
                 noise_scheduler: NoiseScheduler,
                 device: str = 'cpu'):

        self.denoiser = denoiser.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device

    def training_step(self,
                      batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step"""

        full_series = batch['full_series'].to(self.device)
        condition = batch['condition'].to(self.device)
        mask = batch['mask'].to(self.device)

        batch_size = full_series.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device)

        # Add noise
        noise = torch.randn_like(full_series)
        noisy_series = self.noise_scheduler.q_sample(full_series, t, mask, noise)

        # Predict noise
        predicted_noise = self.denoiser(noisy_series, t, condition)

        # Compute loss only on target regions (where mask == 0)
        loss_mask = (1 - mask)
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        loss = (loss * loss_mask).sum() / loss_mask.sum()

        return loss

    @torch.no_grad()
    def p_sample(self,
                 x: torch.Tensor,
                 t: torch.Tensor,
                 condition: torch.Tensor,
                 mask: torch.Tensor):
        """Single reverse diffusion step"""
        batch_size = x.shape[0]

        # Predict noise
        predicted_noise = self.denoiser(x, t, condition)

        # Get coefficients
        alpha_t = self.noise_scheduler.alphas[t].view(-1, 1)
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t].view(-1, 1)
        beta_t = self.noise_scheduler.betas[t].view(-1, 1)

        # Compute mean
        coeff1 = 1 / alpha_t.sqrt()
        coeff2 = beta_t / (1 - alpha_cumprod_t).sqrt()
        mean = coeff1 * (x - coeff2 * predicted_noise)

        # Add noise (except for t=0)
        if t[0] > 0:
            posterior_variance = self.noise_scheduler.posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(x)
            x_prev = mean + posterior_variance.sqrt() * noise
        else:
            x_prev = mean

        # Re-impose conditioning
        x_prev = condition * mask + x_prev * (1 - mask)

        return x_prev

    @torch.no_grad()
    def sample(self,
               condition: torch.Tensor,
               mask: torch.Tensor,
               num_inference_steps: Optional[int] = None):
        """Generate samples using reverse diffusion"""
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps

        batch_size, seq_len = condition.shape

        # Start with pure noise in target regions
        x = torch.randn(batch_size, seq_len, device=self.device)
        x = condition * mask + x * (1 - mask)

        # Reverse diffusion
        timesteps = torch.linspace(self.noise_scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_batch, condition, mask)

        return x
