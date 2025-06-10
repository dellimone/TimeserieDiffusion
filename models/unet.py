import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding for time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaGN(nn.Module):
    """Adaptive Group Normalization conditioned on time embedding."""

    def __init__(self, num_groups: int, num_channels: int, emb_dim: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * num_channels)
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        # emb: (batch, emb_dim)

        # Apply group normalization
        x = self.group_norm(x)

        # Get scale and shift from embedding
        emb_out = self.emb_layers(emb)  # (batch, 2 * channels)
        scale, shift = emb_out.chunk(2, dim=1)  # Each: (batch, channels)

        # Apply adaptive scaling and shifting
        scale = scale.unsqueeze(-1)  # (batch, channels, 1)
        shift = shift.unsqueeze(-1)  # (batch, channels, 1)

        return x * (1 + scale) + shift


class ResidualBlock1D(nn.Module):
    """A 1D Residual Block with time and condition embeddings."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 kernel_size: int = 3, num_groups: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # First convolution path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = AdaGN(num_groups, out_channels, time_emb_dim)

        # Second convolution path
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = AdaGN(num_groups, out_channels, time_emb_dim)

        # Residual connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, length)
        # time_emb: (batch, time_emb_dim)

        h = x

        # First conv block
        h = self.conv1(h)
        h = self.norm1(h, time_emb)
        h = F.silu(h)

        # Add time embedding
        time_proj = self.time_emb_proj(time_emb).unsqueeze(-1)  # (batch, out_channels, 1)
        h = h + time_proj

        # Second conv block
        h = self.conv2(h)
        h = self.norm2(h, time_emb)
        h = F.silu(h)

        # Residual connection
        return h + self.shortcut(x)


class AttentionBlock1D(nn.Module):
    """Self-attention block for 1D sequences."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(8, channels)
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
        q = q.transpose(-2, -1)  # (batch, num_heads, length, head_dim)
        k = k.transpose(-2, -1)  # (batch, num_heads, length, head_dim)
        v = v.transpose(-2, -1)  # (batch, num_heads, length, head_dim)

        # Scaled dot-product attention
        scale = (self.head_dim ** -0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (batch, num_heads, length, head_dim)
        out = out.transpose(-2, -1).contiguous()  # (batch, num_heads, head_dim, length)
        out = out.view(batch, channels, length)

        out = self.proj_out(out)
        return x + out


class Downsample1D(nn.Module):
    """Downsampling block with 1D convolution."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        else:
            self.conv = nn.AvgPool1d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsampling block with ConvTranspose1D."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
        else:
            self.conv = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1D(nn.Module):
    """1D U-Net denoiser for time series"""

    def __init__(self,
                 in_channels: int = 1,
                 model_channels: int = 128,
                 out_channels: int = 1,
                 num_res_blocks: int = 2,
                 attention_resolutions: tuple = (2, 4),
                 channel_mult: tuple = (1, 2, 4),
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 time_embed_dim: int = None):

        super().__init__()

        if time_embed_dim is None:
            time_embed_dim = model_channels * 4

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.dropout = dropout

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Condition embedding (for conditioning on input)
        self.cond_embed = nn.Conv1d(in_channels, model_channels, 1)

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, model_channels, 3, padding=1)

        # Build encoder layers
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        # Track channels at each resolution for skip connections
        self.skip_channels = []

        ch = model_channels
        ds = 1  # Current downsampling factor

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # Residual blocks for this level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                blocks.append(ResidualBlock1D(ch, out_ch, time_embed_dim))
                ch = out_ch

                # Add attention if needed
                if ds in attention_resolutions:
                    blocks.append(AttentionBlock1D(ch, num_heads))

            self.down_blocks.append(blocks)
            self.skip_channels.append(ch)

            # Add downsampling (except for last level)
            if level < len(channel_mult) - 1:
                self.down_samples.append(Downsample1D(ch))
                ds *= 2
            else:
                self.down_samples.append(None)

        # Middle block
        self.mid_block1 = ResidualBlock1D(ch, ch, time_embed_dim)
        self.mid_attn = AttentionBlock1D(ch, num_heads)
        self.mid_block2 = ResidualBlock1D(ch, ch, time_embed_dim)

        # Build decoder layers
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            # Residual blocks for this level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):  # +1 for extra block after skip
                # First block needs to handle skip connection
                if i == 0:
                    in_ch = ch + self.skip_channels[level]  # current + skip
                else:
                    in_ch = out_ch

                blocks.append(ResidualBlock1D(in_ch, out_ch, time_embed_dim))
                ch = out_ch

                # Add attention if needed
                if ds in attention_resolutions:
                    blocks.append(AttentionBlock1D(ch, num_heads))

            self.up_blocks.append(blocks)

            # Add upsampling (except for first level in reverse)
            if level > 0:
                self.up_samples.append(Upsample1D(ch))
                ds //= 2
            else:
                self.up_samples.append(None)

        # Output projection
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                condition: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length) - noisy input
            timesteps: (batch,) - diffusion timesteps
            condition: (batch, channels, length) - conditioning signal
        """

        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Combine input with condition
        if condition is not None:
            # Add conditioning information
            cond_emb = self.cond_embed(condition)
            h = self.input_proj(x) + cond_emb
        else:
            h = self.input_proj(x)

        # Encoder path - store skip connections
        skips = []

        for level, (blocks, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            for block in blocks:
                if isinstance(block, ResidualBlock1D):
                    h = block(h, time_emb)
                else:  # AttentionBlock1D
                    h = block(h)

            skips.append(h)

            if downsample is not None:
                h = downsample(h)

        # Middle blocks
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Decoder path - use skip connections
        for level, (blocks, upsample) in enumerate(zip(self.up_blocks, self.up_samples)):
            # Add skip connection for first block
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for block in blocks:
                if isinstance(block, ResidualBlock1D):
                    h = block(h, time_emb)
                else:  # AttentionBlock1D
                    h = block(h)

            if upsample is not None:
                h = upsample(h)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = UNet1D(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(2, 4),
        channel_mult=(1, 2, 4),
        num_heads=4
    ).to(device)

    # Test inputs
    batch_size = 4
    seq_length = 64

    x = torch.randn(batch_size, 1, seq_length).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    condition = torch.randn(batch_size, 1, seq_length).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x, timesteps, condition)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output, output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("UNet1D implementation test passed!")