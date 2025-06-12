import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from model.diffusion import TimeSeriesDDPM


def train_model(model: TimeSeriesDDPM,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                num_epochs: int = 100,
                learning_rate: float = 1e-3):
    """
    Training loop for the diffusion model with validation, learning rate scheduler,
    TensorBoard logging, and saving the best model based on validation loss.
    Early stopping is removed.
    """

    optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    # Store losses
    train_losses = []
    val_losses = []

    num_params = sum(p.numel() for p in model.denoiser.parameters() if p.requires_grad)
    print(f"Denoiser model parameters: {num_params:,}")
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.denoiser.train() # Set denoiser to training mode
        epoch_train_losses = []
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            loss = model.training_step(batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)

            optimizer.step()
            epoch_train_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.denoiser.eval() # Set denoiser to evaluation mode
        epoch_val_losses = []
        with torch.no_grad(): # Disable gradient calculations for validation
            for batch_idx, batch in enumerate(val_dataloader):

                val_loss = model.training_step(batch)
                epoch_val_losses.append(val_loss.item())

        avg_val_loss = np.mean(epoch_val_losses)
        val_losses.append(avg_val_loss)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.8f}")

        # Learning Rate Scheduler step (step with validation loss)
        scheduler.step(avg_val_loss)

    return train_losses, val_losses
