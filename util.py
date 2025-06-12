import torch
from matplotlib import pyplot as plt

from model.noise_scheduler import NoiseScheduler
from model.unet import SinusoidalPositionalEmbedding

def visualize_forward_diffusion(
    scheduler: NoiseScheduler,
    time_serie: torch.Tensor,
    mask: torch.Tensor,
    timesteps_to_show: list[int],
    figure_size: tuple[int, int] = (12, 10)
):
    """
    Visualizes the forward diffusion process by showing the original signal
    and noisy versions at selected timesteps.

    Args:
        scheduler (NoiseScheduler): An instance of the NoiseScheduler class.
        time_serie (torch.Tensor): The initial clean signal.
                                     Expected shape: (batch_size, sequence_length)
        mask (torch.Tensor): A mask tensor. Noise is applied where mask is 0,
                             and the original signal is preserved where mask is 1.
                             Expected shape: (batch_size, sequence_length)
        timesteps_to_show (list[int]): A list of timesteps at which to visualize
                                       the noisy signal.
        figure_size (tuple[int, int]): Size of the matplotlib figure.
    """
    plt.figure(figsize=figure_size)

    for i, timestep in enumerate(timesteps_to_show):
        # Ensure timestep tensor has correct dimensions, typically a single element batch for `t`
        t_tensor = torch.tensor([timestep])
        noisy_sample = scheduler.q_sample(time_serie, t_tensor, mask)
        # Squeeze and convert to numpy for plotting
        noisy_sample_np = noisy_sample.squeeze().numpy()
        clean_signal_np = time_serie.squeeze().numpy()

        plt.subplot(len(timesteps_to_show), 1, i + 1)
        plt.plot(noisy_sample_np, label=f'Noisy at timestep {timestep}', color='blue')
        plt.plot(clean_signal_np, label='Original Signal', linestyle='dashed', color='red', alpha=0.7)
        plt.legend(loc='upper right')
        plt.ylim([-2, 2]) # Set fixed y-limits for consistent comparison
        plt.grid(True)
        plt.title(f'Timestep {timestep}') # Add subplot titles

    plt.suptitle('Diffusion Forward Noising Process Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
    plt.show()


def visualize_positional_embedding(embedding_dim: int, max_time_steps: int):
    """
    Visualizes the sinusoidal positional embeddings for a given dimension and number of time steps.
    """
    # Create the embedding module
    pos_embedder = SinusoidalPositionalEmbedding(embedding_dim)

    # Generate time steps
    time_steps = torch.arange(max_time_steps, dtype=torch.float32)

    # Get the embeddings
    embeddings = pos_embedder(time_steps)

    # Convert to numpy for plotting
    embeddings_np = embeddings.detach().cpu().numpy()

    print(f"Embedding shape for dim={embedding_dim}, max_time_steps={max_time_steps}: {embeddings_np.shape}")

    # Plotting
    plt.figure(figsize=(15, 8))

    # Plot each embedding dimension over time steps
    for i in range(min(embedding_dim, 20)):  # Plot up to 20 dimensions for clarity
        plt.plot(time_steps.numpy(), embeddings_np[:, i], label=f'Dim {i}')

    plt.title(f'Sinusoidal Positional Embeddings (Dim={embedding_dim})')
    plt.xlabel('Time Step (Position)')
    plt.ylabel('Embedding Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Heatmap Visualization ---
    plt.figure(figsize=(12, 10))
    plt.imshow(embeddings_np.T, cmap='viridis', origin='lower', aspect='auto',
               extent=[0, max_time_steps - 1, 0, embedding_dim - 1])
    plt.colorbar(label='Embedding Value')
    plt.title(f'Sinusoidal Positional Embeddings Heatmap (Dim={embedding_dim})')
    plt.xlabel('Time Step (Position)')
    plt.ylabel('Embedding Dimension')
    plt.show()

