import torch
import numpy as np
from torch.utils.data import Dataset

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
        # Validate task_type
        if task_type not in ['forecasting', 'imputation']:
            raise ValueError("task_type must be 'forecasting' or 'imputation'")
        self.task_type = task_type

        # target_length is primarily used for forecasting here
        self.target_length = seq_length - condition_length

        # Generate all data upfront for consistency
        self.data = self._generate_data()

    def _generate_data(self):
        """Generate diverse synthetic time series patterns"""
        data = []

        for i in range(self.num_samples):
            # Cycle through different patterns
            pattern_type = i % 4

            if pattern_type == 0:  # Sine wave with noise
                freq = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                t = np.linspace(0, 4 * np.pi, self.seq_length)
                series = amplitude * np.sin(freq * t + phase) + np.random.normal(0, 0.1, self.seq_length)

            elif pattern_type == 1:  # Exponential Decay/Growth with Noise
                initial_value = np.random.uniform(0.5, 2.0)
                # Small negative k for decay
                k = np.random.uniform(-0.1, 0)
                t = np.arange(self.seq_length)
                series = initial_value * np.exp(k * t) + np.random.normal(0, 0.08, self.seq_length)

            elif pattern_type == 2:  # Trend with seasonality
                trend = np.linspace(np.random.uniform(-1, 1), np.random.uniform(-1, 1), self.seq_length)
                # Ensure seasonality period is not too large for short sequences
                seasonal_period = np.random.randint(4, min(self.seq_length // 2, 16) + 1)
                seasonal = 0.3 * np.sin(2 * np.pi * np.arange(self.seq_length) / seasonal_period)
                noise = np.random.normal(0, 0.1, self.seq_length)
                series = trend + seasonal + noise

            elif pattern_type == 3:  # Ornstein-Uhlenbeck (OU) Process
                theta = np.random.uniform(0.05, 0.2) # Rate of reversion
                mu = np.random.uniform(-0.5, 0.5)   # Long-term mean
                sigma = np.random.uniform(0.05, 0.3) # Volatility
                dt = 0.1 # Time step

                series = np.zeros(self.seq_length)
                series[0] = np.random.uniform(-1, 1) # Initial value

                for t in range(1, self.seq_length):
                    series[t] = series[t-1] + theta * (mu - series[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)

            # Normalize to [-1, 1]
            series = (series - series.mean()) / (series.std() + 1e-8)
            series = np.clip(series, -3, 3) / 3  # Clip to -3,3 std deviations and then scale to -1,1

            data.append(series.astype(np.float32))

        return np.array(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        series = self.data[idx]

        if self.task_type == 'forecasting':
            if self.target_length < 0:
                raise ValueError("seq_length must be greater than or equal to condition_length for forecasting.")

            target = series.copy()  # Return full series as target
            mask = np.concatenate(
                [np.ones(self.condition_length, dtype=bool), np.zeros(self.target_length, dtype=bool)])

            # Make condition tensor full length, zeroing out the target region
            condition = series.copy()
            condition[self.condition_length:] = 0.0  # Use float for consistency

        if self.task_type == 'imputation':
            condition = series.copy()
            mask = np.ones(self.seq_length, dtype=bool)
            target = series.copy()  # Return full series as target

            # Determine number of gaps: 2 to 4, but not more than seq_length // 5 to ensure space
            num_gaps = np.random.randint(2, min(5, max(2, self.seq_length // 5)))
            # For very short sequences, ensure at least one gap is possible
            if self.seq_length < 10:
                num_gaps = 1

            # Determine max individual gap length
            max_single_gap_len = max(1, self.seq_length // (num_gaps * 2))  # Ensure at least 1, and leaves space

            masked_ranges = []  # To keep track of already masked (start, end) tuples [inclusive start, exclusive end)

            for _ in range(num_gaps):
                attempts = 0
                gap_found = False
                while attempts < 50 and not gap_found:  # Limit attempts to avoid infinite loops or slow generation
                    current_gap_length = np.random.randint(1, max_single_gap_len + 1)
                    if self.seq_length - current_gap_length < 0:  # Gap is too long for the sequence
                        break

                    start_candidate = np.random.randint(0, self.seq_length - current_gap_length + 1)
                    end_candidate = start_candidate + current_gap_length

                    # Check for overlap with existing masked ranges
                    overlap = False
                    for (m_start, m_end) in masked_ranges:
                        # Overlap occurs if the intervals [start_candidate, end_candidate) and [m_start, m_end) intersect.
                        # This happens if: (start_candidate < m_end) AND (end_candidate > m_start)
                        if (start_candidate < m_end and end_candidate > m_start):
                            overlap = True
                            break

                    if not overlap:
                        # Found a valid non-overlapping spot
                        masked_ranges.append((start_candidate, end_candidate))

                        # Apply mask and condition updates
                        segment_indices = np.arange(start_candidate, end_candidate)
                        condition[segment_indices] = 0.0  # Mask by setting to zero
                        mask[segment_indices] = False
                        gap_found = True
                    attempts += 1

        return {
            'full_series': torch.tensor(series, dtype=torch.float32).unsqueeze(0),
            'condition': torch.tensor(condition, dtype=torch.float32).unsqueeze(0),
            'target': torch.tensor(target, dtype=torch.float32).unsqueeze(0),
            'mask': torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        }