import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model.noise_scheduler import NoiseScheduler
from model.unet import SinusoidalPositionalEmbedding

def visualize_forward_diffusion(scheduler: NoiseScheduler, time_serie: torch.Tensor, mask: torch.Tensor, timesteps_to_show: list[int], figure_size: tuple[int, int] = (12, 10)
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
    plt.savefig('synthetic_timeseries_noisy.svg', format='svg', bbox_inches='tight', dpi=300)


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

class AttentionBlock1DForViz(nn.Module):
    """Self-attention block for 1D sequences."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(8, channels) # Using 8 groups as an example
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        batch, channels, length = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)  # (batch, 3*channels, length)

        # Reshape for multi-head attention
        qkv = qkv.view(batch, 3, self.num_heads, self.head_dim, length)
        q, k, v = qkv.unbind(1)  # Each: (batch, num_heads, head_dim, length)

        # Compute attention
        # Transpose to (batch, num_heads, length, head_dim) for matmul
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Scaled dot-product attention
        scale = (self.head_dim ** -0.5)
        # Matmul q with k.transpose(-2, -1) to get (batch, num_heads, length, length)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (batch, num_heads, length, head_dim)
        out = out.transpose(-2, -1).contiguous()  # (batch, num_heads, head_dim, length)
        out = out.view(batch, channels, length)

        out = self.proj_out(out)
        return x + out, attn # Return attention for visualization

def visualizeAttention(channels: int = 64, num_heads: int =8, input_length: int = 16):
    batch_size = 1 # Visualizing one sample

    # Instantiate the Attention Block
    attn_block = AttentionBlock1DForViz(channels, num_heads)

    # Create a dummy input tensor
    # (batch_size, channels, input_length)
    dummy_input = torch.randn(batch_size, channels, input_length)

    # Perform a forward pass and get attention weights
    with torch.no_grad(): # No need to calculate gradients for visualization
        output, attention_weights = attn_block(dummy_input)

    # Extract attention for a specific sample and head
    # We'll visualize the first sample (index 0) and the first head (index 0)
    # attention_weights shape: (batch, num_heads, length, length)
    attn_to_plot = attention_weights[0, 0, :, :].cpu().numpy()

    # 6. Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_to_plot, cmap='viridis', origin='lower',
               extent=[0, input_length, 0, input_length])
    plt.colorbar(label='Attention Weight')
    plt.title(f'Attention Heatmap (Sample 0, Head 0)\nInput Length: {input_length}')
    plt.xlabel('Key Position (Input Element Index)')
    plt.ylabel('Query Position (Output Element Index)')
    plt.xticks(np.arange(0, input_length, 4))
    plt.yticks(np.arange(0, input_length, 4))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def generate_time_series(series_type='noisy_sine', length=200):
    """Generate different types of time series for demonstration."""
    t = np.linspace(0, 10, length)

    if series_type == 'noisy_sine':
        # Sine wave with high frequency noise
        ts = np.sin(t) + 0.3 * np.sin(5 * t) + 0.2 * np.random.randn(length)
    elif series_type == 'step':
        # Step function with noise
        ts = np.zeros(length)
        ts[length // 3:2 * length // 3] = 1
        ts += 0.1 * np.random.randn(length)
    elif series_type == 'spike':
        # Signal with spikes
        ts = 0.1 * np.random.randn(length)
        spike_positions = [50, 100, 150]
        for pos in spike_positions:
            if pos < length:
                ts[pos] = 2
    else:
        ts = np.random.randn(length)

    return t, ts

def create_kernels():
    """Create different convolution kernels."""
    kernels = {}

    # Moving average (box filter)
    kernels['moving_average_20'] = np.ones(20) / 20

    # Gaussian kernel
    x = np.arange(-3, 4)
    kernels['gaussian'] = np.exp(-x ** 2 / 2)
    kernels['gaussian'] /= kernels['gaussian'].sum()

    # Edge detection (derivative)
    kernels['edge_detection'] = np.array([-1, 0, 1])

    return kernels

def apply_convolution(signal_data, kernel, mode='same'):
    """Apply convolution with proper handling of edges."""
    return np.convolve(signal_data, kernel, mode=mode)

def visualize_convolution_effects():
    """Create comprehensive visualization of convolution effects."""
    # Generate different time series
    series_types = ['noisy_sine', 'step', 'spike']
    kernels = create_kernels()

    fig, axes = plt.subplots(len(series_types), len(kernels),
                             figsize=(20, 16))
    fig.suptitle('1D Convolution Effects on Different Time Series', fontsize=16)

    for i, series_type in enumerate(series_types):
        t, original = generate_time_series(series_type)

        for j, (kernel_name, kernel) in enumerate(kernels.items()):
            ax = axes[i, j]

            # Apply convolution
            convolved = apply_convolution(original, kernel)

            # Handle length differences
            if len(convolved) != len(original):
                # Pad or truncate to match original length
                if len(convolved) > len(original):
                    start = (len(convolved) - len(original)) // 2
                    convolved = convolved[start:start + len(original)]
                else:
                    convolved = np.pad(convolved,
                                       (0, len(original) - len(convolved)),
                                       'edge')

            # Plot original and convolved signals
            ax.plot(t, original, 'b-', alpha=0.7, linewidth=1, label='Original')
            ax.plot(t, convolved, 'r-', linewidth=2, label='Convolved')

            ax.set_title(f'{series_type}\n{kernel_name}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Set y-axis limits for better visualization
            y_min = min(np.min(original), np.min(convolved))
            y_max = max(np.max(original), np.max(convolved))
            margin = 0.1 * (y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    plt.show()

def kernel_visualization():
    """Visualize the kernels themselves."""
    kernels = create_kernels()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Convolution Kernels', fontsize=14)

    axes = axes.flatten()

    for i, (name, kernel) in enumerate(kernels.items()):
        if i < len(axes):
            ax = axes[i]
            x = np.arange(len(kernel))
            ax.stem(x, kernel, basefmt=" ")
            ax.set_title(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Position')
            ax.set_ylabel('Weight')

    plt.tight_layout()
    plt.show()

