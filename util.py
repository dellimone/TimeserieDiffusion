import torch
from matplotlib import pyplot as plt

from noise_scheduler import NoiseScheduler

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
