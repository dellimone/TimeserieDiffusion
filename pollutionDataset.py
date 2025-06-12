import torch
from torch.utils.data import Dataset
import csv
from typing import Tuple, Dict


class PollutionDataset(Dataset):
    def __init__(self,
                 path: str = 'data/pollution.csv',
                 missing_channels: Tuple[int, ...] = (0,),
                 missing_num: Tuple[int, ...] = (24*2,),
                 task_type: str = 'forecasting'):
        assert task_type in {'forecasting', 'imputation', 'both'}, "Invalid task_type"

        self.path = path
        self.missing_channels = missing_channels
        self.missing_num = missing_num
        self.task_type = task_type

        self.data = self._extract_data(path)

    def _extract_data(self, path: str) -> torch.Tensor:
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            drop_indices = [header.index('date'), header.index('wnd_dir')]

            for row in reader:
                values = [float(x) if x else float('nan') for i, x in enumerate(row) if i not in drop_indices]
                if not any(torch.isnan(torch.tensor(values))):
                    data.append(values)

        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Normalize (Min-Max)
        min_vals = data_tensor.min(dim=0).values
        max_vals = data_tensor.max(dim=0).values
        data_normalized = (data_tensor - min_vals) / (max_vals - min_vals + 1e-8)

        # Reshape to (weeks, channels, hours)
        hours_per_week = 168
        n = data_normalized.shape[0] // hours_per_week
        data_trimmed = data_normalized[:n * hours_per_week]
        return data_trimmed.view(n, hours_per_week, -1).permute(0, 2, 1)  # (weeks, channels, hours)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series = self.data[idx]  # (channels, seq_len)
        full_series = series.clone()
        condition = series.clone()
        mask = torch.ones_like(series, dtype=torch.bool)

        for ch, num in zip(self.missing_channels, self.missing_num):
            if self.task_type in {'forecasting'}:
                mask[ch, -num:] = False

        condition[~mask] *= 0
        target_values = full_series[~mask].view(1, -1)  # shape (1, total_missing)

        return {
            'full_series': full_series,   # (channels, seq_len)
            'condition': condition,       # (channels, seq_len)
            'target': target_values,      # (1, total_missing)
            'mask': mask                  # (channels, seq_len)
    }

if __name__ == '__main__':

    from torch.utils.data import DataLoader

    csv_path = 'data/pollution.csv'  # Replace with your dataset path
    missing_channels: Tuple[int, ...] = (0,)
    missing_num: Tuple[int, ...] = (2 * 24,)
    task_type = 'forecasting'

    # Initialize dataset
    dataset = PollutionDataset(
        path=csv_path,
        missing_channels=missing_channels,
        missing_num=missing_num,
        task_type=task_type
    )

    # Wrap in DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate over batches
    batch = next(iter(dataloader))
    full_series = batch['full_series']  # (B, channels, seq_len)
    condition = batch['condition']  # (B, channels, seq_len) - input to model
    target = batch['target']  # (B, total_missing) - ground truth values
    mask = batch['mask']  # (B, channels, seq_len) - indicates missing entries

    print("Full Series Shape:", full_series.shape)
    print("Condition Shape:", condition.shape)
    print("Target Shape:", target.shape)
    print("Mask Shape:", mask.shape)