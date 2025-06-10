import numpy as np
import torch
from torch.utils.data import Dataset

# ======================== SYNTHETIC DATA GENERATION ========================

class SyntheticTimeSeriesDataset(Dataset):
    """Generate synthetic 1D time series for diffusion model training"""

    def __init__(self,
                 num_samples: int = 1000,
                 seq_length: int = 64,
                 condition_length: int = 32,
                 task_type: str = 'forecasting'):
        """
        Args:
            num_samples: Number of time series to generate
            seq_length: Total length of each time series
            condition_length: Length of conditioning segment
            task_type: 'forecasting' or 'imputation'
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.condition_length = condition_length
        self.task_type = task_type
        self.target_length = seq_length - condition_length

        # Generate all data upfront for consistency
        self.data = self._generate_data()

    def _generate_data(self):
        """Generate diverse synthetic time series patterns"""
        data = []

        for i in range(self.num_samples):
            # Mix different patterns
            pattern_type = i % 4

            if pattern_type == 0:  # Sine wave with noise
                freq = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                t = np.linspace(0, 4 * np.pi, self.seq_length)
                series = amplitude * np.sin(freq * t + phase) + np.random.normal(0, 0.1, self.seq_length)

            elif pattern_type == 1:  # AR(1) process
                phi = np.random.uniform(0.3, 0.9)
                series = np.zeros(self.seq_length)
                series[0] = np.random.normal(0, 1)
                for t in range(1, self.seq_length):
                    series[t] = phi * series[t - 1] + np.random.normal(0, 0.3)

            elif pattern_type == 2:  # Trend with seasonality
                trend = np.linspace(np.random.uniform(-1, 1), np.random.uniform(-1, 1), self.seq_length)
                seasonal = 0.3 * np.sin(2 * np.pi * np.arange(self.seq_length) / 8)
                noise = np.random.normal(0, 0.1, self.seq_length)
                series = trend + seasonal + noise

            else:  # Random walk
                steps = np.random.normal(0, 0.2, self.seq_length)
                series = np.cumsum(steps)

            # Normalize to [-1, 1]
            series = (series - series.mean()) / (series.std() + 1e-8)
            series = np.clip(series, -3, 3) / 3

            data.append(series.astype(np.float32))

        return np.array(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        series = self.data[idx]

        if self.task_type == 'forecasting':
            # Use first part as condition, predict the rest
            target = series[self.condition_length:]
            mask = np.concatenate([np.ones(self.condition_length), np.zeros(self.target_length)])

            # Make condition tensor full length, zeroing out the target region
            condition = series.copy()
            condition[self.condition_length:] = 0

        else:  # imputation
            # Randomly mask some values in the middle
            condition = series.copy()
            mask = np.ones(self.seq_length)

            # Create random gaps
            gap_start = np.random.randint(self.condition_length // 2, self.seq_length - self.target_length)
            gap_end = gap_start + self.target_length
            condition[gap_start:gap_end] = 0  # Zero out missing values
            mask[gap_start:gap_end] = 0
            target = series[gap_start:gap_end]

        return {
            'full_series': torch.tensor(series, dtype=torch.float32),
            'condition': torch.tensor(condition, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32)
        }

