import numpy as np
import torch
from matplotlib import pyplot as plt

from diffusion import TimeSeriesDDPM
from dataset import SyntheticTimeSeriesDataset


# ======================== EVALUATION AND VISUALIZATION ========================

def evaluate_model(model: TimeSeriesDDPM, dataset: SyntheticTimeSeriesDataset,
                   num_samples: int = 5, num_inference_steps: int = 50):
    """Evaluate model and create visualizations"""

    model.denoiser.eval()

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]

    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]

            condition = sample['condition'].unsqueeze(0).to(model.device)
            mask = sample['mask'].unsqueeze(0).to(model.device)
            full_series = sample['full_series'].numpy()

            # Generate prediction
            generated = model.sample(condition, mask, num_inference_steps)
            generated = generated.cpu().numpy()[0]

            # Plot
            x_axis = np.arange(len(full_series))
            axes[i].plot(x_axis, full_series, 'b-', label='Ground Truth', alpha=0.7)
            axes[i].plot(x_axis, generated, 'r--', label='Generated', alpha=0.7)

            # Highlight conditioning vs target regions
            condition_region = mask.cpu().numpy()[0] == 1
            target_region = mask.cpu().numpy()[0] == 0

            # Get y-limits for filling
            ymin, ymax = axes[i].get_ylim()

            axes[i].fill_between(x_axis, ymin, ymax, where=condition_region,
                                 alpha=0.2, color='green', label='Conditioning')
            axes[i].fill_between(x_axis, ymin, ymax, where=target_region,
                                 alpha=0.2, color='red', label='Target')

            axes[i].legend()
            axes[i].set_title(f'Sample {i + 1} - {dataset.task_type.capitalize()}')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()