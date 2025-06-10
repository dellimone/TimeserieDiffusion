import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from diffusion import TimeSeriesDDPM

# ======================== TRAINING LOOP ========================

def train_model(model: TimeSeriesDDPM,
                dataloader: DataLoader,
                num_epochs: int = 100,
                learning_rate: float = 1e-3,
                device: str = 'cpu'):
    """Training loop for the diffusion model"""

    optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=learning_rate)
    losses = []

    model.denoiser.train()

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            loss = model.training_step(batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0:
            print(f"Epochloss_mask {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

    return losses