import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from model.noise_scheduler import NoiseScheduler


class TimeSeriesDDPM:
    """Main diffusion model class"""

    def __init__(self, denoiser: nn.Module, noise_scheduler: NoiseScheduler, device: str = 'cpu'):

        self.denoiser = denoiser.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        loss_mask = ~mask
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        loss = (loss * loss_mask).sum() / loss_mask.sum()

        return loss

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor):
        """Single reverse diffusion step"""

        # Predict noise
        predicted_noise = self.denoiser(x, t, condition)

        # Get coefficients and reshape for broadcasting (B, 1, 1)
        alpha_t = self.noise_scheduler.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t].view(-1, 1, 1)
        beta_t = self.noise_scheduler.betas[t].view(-1, 1, 1)

        # Compute mean
        coeff1 = 1 / alpha_t.sqrt()
        coeff2 = beta_t / (1 - alpha_cumprod_t).sqrt()
        mean = coeff1 * (x - coeff2 * predicted_noise)

        # Add noise (except for t=0)
        if t[0] > 0:
            posterior_variance = self.noise_scheduler.posterior_variance[t].view(-1, 1, 1)
            noise = torch.randn_like(x)
            x_prev = mean + posterior_variance.sqrt() * noise
        else:
            x_prev = mean

        # Re-impose conditioning
        x_prev = condition * mask + x_prev * (~mask)

        return x_prev

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, mask: torch.Tensor, num_inference_steps: Optional[int] = None):
        """Generate samples plots using reverse diffusion"""
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps

        batch_size, channels, seq_len = condition.shape

        # Start with pure noise in target regions
        x = torch.randn(batch_size, channels, seq_len, device=self.device)
        x = condition * mask + x * (~mask)

        # Reverse diffusion
        timesteps = torch.linspace(self.noise_scheduler.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_batch, condition, mask)

        return x

if __name__ == '__main__':
    from model.unet import UNet1D

    batch_size = 4
    channels = 2
    seq_len = 16
    cond_len = 8

    x = torch.randn(batch_size, channels, seq_len)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, :, cond_len:] = False
    condition = x * mask
    target = x[:, :, :cond_len]

    batch = {'full_series': x, 'condition': condition, 'mask': mask, 'target': target}

    print(f"batch['full_series].shape: {batch['full_series'].shape}")
    print(f"batch['condition'].shape: {batch['condition'].shape}")
    print(f"batch['target'].shape: {batch['target'].shape}")
    print(f"batch['mask'].shape: {batch['mask'].shape}")

    print(batch)

    noiser = NoiseScheduler()
    denoiser = UNet1D(
        in_channels=channels,
        out_channels=channels,
        model_channels=64,
        num_res_blocks=2,
        attention_resolutions=(2, 4),
        channel_mult=(1, 2, 4),
        num_heads=4
    )

    ddpm = TimeSeriesDDPM(denoiser, noiser)
    print('='*50)
    print('ddpm training step')
    print('='*50)

    print(ddpm.training_step(batch))

    print('=' * 50)
    print('ddpm sample')
    print('=' * 50)

    print(ddpm.sample(condition, mask,1).shape)



