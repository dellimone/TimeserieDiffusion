import torch
import torch.nn as nn

class MLPDenoiser(nn.Module):
    """Simple MLP denoiser - baseline architecture"""

    def __init__(self, seq_length: int, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.seq_length = seq_length

        # Time embedding
        time_embed_dim = 32  # Fixed dimension for time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Main MLP
        input_dim = seq_length * 2 + time_embed_dim  # noisy_x + condition + time_embed
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])

        layers.append(nn.Linear(hidden_dim, seq_length))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor):
        batch_size = x.shape[0]

        # Time embedding
        t_embed = self.time_embed(t.float().unsqueeze(-1))

        # Flatten and concatenate
        x_flat = x.view(batch_size, -1)
        cond_flat = condition.view(batch_size, -1)


        combined = torch.cat([x_flat, cond_flat, t_embed], dim=-1)

        # Predict noise
        noise_pred = self.mlp(combined)
        return noise_pred.view(batch_size, self.seq_length)