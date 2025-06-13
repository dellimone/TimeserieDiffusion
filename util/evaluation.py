import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def evaluate_model(ddpm_model, test_dataset, num_samples: int = 4, num_inference_steps: Optional[int] = None):
    """
    Evaluate and visualize time series diffusion model results.

    Args:
        ddpm_model: TimeSeriesDDPM model instance.
        test_dataset: Dataset containing test samples plots.
        num_samples: Number of samples plots to evaluate and visualize.
        num_inference_steps: Number of inference steps for sampling (uses model default if None).
    """

    # Set model to evaluation mode
    ddpm_model.denoiser.eval()

    results = []

    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            # Get test sample
            sample = test_dataset[i]

            # Prepare batch (add batch dimension)
            condition = sample['condition'].unsqueeze(0).to(ddpm_model.device)
            mask = sample['mask'].unsqueeze(0).to(ddpm_model.device)
            full_series = sample['full_series'].unsqueeze(0).to(ddpm_model.device)
            target = sample['target'].unsqueeze(0).to(ddpm_model.device)

            # Generate prediction using diffusion model
            predicted = ddpm_model.sample(condition, mask, num_inference_steps)

            # Store results
            results.append({
                'condition': condition.cpu().squeeze(0),
                'target': target.cpu().squeeze(0),
                'predicted': predicted.cpu().squeeze(0),
                'mask': mask.cpu().squeeze(0),
                'full_series': full_series.cpu().squeeze(0)
            })

    # Create visualization
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, result in enumerate(results):
        ax = axes[i]

        # Extract data
        condition = result['condition'].numpy().flatten()
        target = result['target'].numpy().flatten()
        predicted = result['predicted'].numpy().flatten()
        mask = result['mask'].numpy().flatten()

        # Find the split point (first False in mask)
        split_idx = np.where(~mask)[0][0] if np.any(~mask) else len(condition)

        # Create x-axis
        x_total = np.arange(len(condition))
        x_known = x_total[:split_idx]
        x_target = x_total[split_idx:]

        # Plot known/condition part
        ax.plot(x_known, condition[:split_idx], 'b-', linewidth=2, label='Known/Condition', marker='o', markersize=3)

        # Plot target (ground truth) and prediction only for the unknown part
        if len(x_target) > 0:
            ax.plot(x_target, target[split_idx:], 'g-', linewidth=2, label='Ground Truth', marker='s', markersize=3)
            ax.plot(x_target, predicted[split_idx:], 'r--', linewidth=2, label='Predicted', marker='^', markersize=3)

        # Add vertical line at split point
        ax.axvline(x=split_idx - 0.5, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')

        # Formatting
        ax.set_title(f'Sample {i + 1}: Time Series Prediction')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Display error metrics
        if len(x_target) > 0:
            mse = np.mean((target[split_idx:] - predicted[split_idx:]) ** 2)
            mae = np.mean(np.abs(target[split_idx:] - predicted[split_idx:]))
            ax.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()


def evaluate_model_multi(ddpm_model, test_dataset, num_samples: int = 4, num_inference_steps: Optional[int] = None):
    """
    Evaluate and visualize time series diffusion model results.

    Args:
        ddpm_model: TimeSeriesDDPM model instance.
        test_dataset: Dataset containing test samples plots.
        num_samples: Number of samples plots to evaluate and visualize.
        num_inference_steps: Number of inference steps for sampling (uses model default if None).
    """

    # Set model to evaluation mode
    ddpm_model.denoiser.eval()

    results = []

    with torch.no_grad():
        for i in range(min(num_samples, len(test_dataset))):
            # Get test sample
            sample = test_dataset[i]

            # Prepare batch (add batch dimension)
            condition = sample['condition'].unsqueeze(0).to(ddpm_model.device)
            mask = sample['mask'].unsqueeze(0).to(ddpm_model.device)
            full_series = sample['full_series'].unsqueeze(0).to(ddpm_model.device)
            target = sample['target'].unsqueeze(0).to(ddpm_model.device)

            # Generate prediction using diffusion model
            predicted = ddpm_model.sample(condition, mask, num_inference_steps)
            # Store results
            results.append({
                'condition': condition.squeeze(0)[0,:].cpu(),
                'target': target.squeeze(0).squeeze(0).cpu(),
                'predicted': predicted.squeeze(0)[0,:].cpu(),
                'mask': mask.squeeze(0)[0,:].cpu(),
                'full_series': full_series.squeeze(0)[0,:].cpu()
            })

    # Create visualization
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, result in enumerate(results):
        ax = axes[i]

        # Extract data
        condition = result['condition'].numpy().flatten()
        target = result['target'].numpy().flatten()
        predicted = result['predicted'].numpy().flatten()
        mask = result['mask'].numpy().flatten()

        # Find the split point (first False in mask)
        split_idx = np.where(~mask)[0][0] if np.any(~mask) else len(condition)

        # Create x-axis
        x_total = np.arange(len(condition))
        x_known = x_total[:split_idx]
        x_target = x_total[split_idx:]

        # Plot known/condition part
        ax.plot(x_known, condition[:split_idx], 'b-', linewidth=2, label='Known/Condition', marker='o', markersize=3)

        # Plot target (ground truth) and prediction only for the unknown part
        if len(x_target) > 0:
            ax.plot(x_target, target, 'g-', linewidth=2, label='Ground Truth', marker='s', markersize=3)
            ax.plot(x_target, predicted[split_idx:], 'r--', linewidth=2, label='Predicted', marker='^', markersize=3)

        # Add vertical line at split point
        ax.axvline(x=split_idx - 0.5, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')

        # Formatting
        ax.set_title(f'Sample {i + 1}: Time Series Prediction')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Display error metrics
        if len(x_target) > 0:
            mse = np.mean((target - predicted[split_idx:]) ** 2)
            mae = np.mean(np.abs(target - predicted[split_idx:]))
            ax.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()