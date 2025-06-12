import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Creates sinusoidal positional embeddings for timestep encoding.
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize the sinusoidal positional embedding.

        Args:
            embedding_dim: Dimension of the output embedding
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert timesteps to sinusoidal embeddings.

        Args:
            timesteps: Tensor of shape (batch_size,) containing timestep values

        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim) containing
                       sinusoidal positional embeddings
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2

        # Create frequency scaling factors using exponential decay
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0)) *
            torch.arange(half_dim, device=device) / half_dim
        )

        # Apply frequencies to timesteps to get phase arguments
        args = timesteps[:, None] * frequencies[None, :]

        # Create embeddings using both cosine and sine
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return embeddings


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization conditioned on time embedding.

    This module applies Group Normalization and then modulates the normalized
    features using scale and shift parameters derived from the time embedding.
    This allows the model to adapt its normalization based on the diffusion timestep.
    """

    def __init__(self, num_groups: int, num_channels: int, emb_dim: int):
        """
        Initialize Adaptive Group Normalization.

        Args:
            num_groups: Number of groups for GroupNorm (typically 8 or 32)
            num_channels: Number of input channels to normalize
            emb_dim: Dimension of the time embedding
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels

        # Standard Group Normalization
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

        # MLP to generate scale and shift parameters from time embedding
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * num_channels)
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive group normalization.

        Args:
            x: Input tensor of shape (batch, channels, length)
            emb: Time embedding of shape (batch, emb_dim)

        Returns:
            Normalized and modulated tensor of shape (batch, channels, length)
        """
        # Apply standard group normalization first
        x = self.group_norm(x)

        # Generate scale and shift parameters from time embedding
        emb_out = self.emb_layers(emb)  # (batch, 2 * channels)
        scale, shift = emb_out.chunk(2, dim=1)  # Each: (batch, channels)

        # Reshape for broadcasting over length dimension
        scale = scale.unsqueeze(-1)  # (batch, channels, 1)
        shift = shift.unsqueeze(-1)  # (batch, channels, 1)

        # Apply adaptive modulation: scale around 1, then shift
        return x * (1 + scale) + shift


class ResidualBlock1D(nn.Module):
    """
    A 1D Residual Block with time embedding conditioning.

    This block follows the typical ResNet structure with two convolutions,
    but adds time embedding conditioning via AdaGN and includes dropout
    for regularization.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 kernel_size: int = 3, num_groups: int = 8, dropout: float = 0.0):
        """
        Initialize the residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_emb_dim: Dimension of time embedding
            kernel_size: Convolution kernel size (default: 3)
            num_groups: Number of groups for GroupNorm (default: 8)
            dropout: Dropout probability (default: 0.0)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # Time embedding projection - adds time information to features
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # First convolution path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.norm1 = AdaGN(num_groups, out_channels, time_emb_dim)

        # Dropout layer for regularization
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Second convolution path
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size // 2)
        self.norm2 = AdaGN(num_groups, out_channels, time_emb_dim)

        # Second dropout layer
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip connection - handles channel dimension mismatch
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch, in_channels, length)
            time_emb: Time embedding of shape (batch, time_emb_dim)

        Returns:
            Output tensor of shape (batch, out_channels, length)
        """
        h = x

        # First convolution block
        h = self.conv1(h)
        h = self.norm1(h, time_emb)
        h = F.silu(h)
        h = self.dropout1(h)

        # Add time embedding information
        time_proj = self.time_emb_proj(time_emb).unsqueeze(-1)  # (batch, out_channels, 1)
        h = h + time_proj

        # Second convolution block
        h = self.conv2(h)
        h = self.norm2(h, time_emb)
        h = F.silu(h)
        h = self.dropout2(h)

        # Residual connection
        return h + self.shortcut(x)


class AttentionBlock1D(nn.Module):
    """
    Self-attention block for 1D sequences.

    This implements multi-head self-attention to allow the model to attend
    to different parts of the sequence.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        """
        Initialize the attention block.

        Args:
            channels: Number of input/output channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        # Layer normalization before attention
        self.norm = nn.GroupNorm(8, channels)

        # Combined query, key, value projection
        self.qkv = nn.Conv1d(channels, channels * 3, 1)

        # Output projection
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to the input.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor of shape (batch, channels, length) with residual connection
        """
        batch, channels, length = x.shape

        # Normalize input
        h = self.norm(x)

        # Generate queries, keys, and values
        qkv = self.qkv(h)  # (batch, 3*channels, length)

        # Reshape for multi-head attention
        qkv = qkv.view(batch, 3, self.num_heads, self.head_dim, length)
        q, k, v = qkv.unbind(1)  # Each: (batch, num_heads, head_dim, length)

        # Transpose for attention computation
        q = q.transpose(-2, -1)  # (batch, num_heads, length, head_dim)
        k = k.transpose(-2, -1)  # (batch, num_heads, length, head_dim)
        v = v.transpose(-2, -1)  # (batch, num_heads, length, head_dim)

        # Scaled dot-product attention
        scale = (self.head_dim ** -0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, num_heads, length, head_dim)

        # Reshape back to original format
        out = out.transpose(-2, -1).contiguous()  # (batch, num_heads, head_dim, length)
        out = out.view(batch, channels, length)

        # Output projection and residual connection
        out = self.proj_out(out)
        return x + out


class Downsample1D(nn.Module):
    """
    Downsampling block for reducing sequence length by factor of 2.

    Can use either strided convolution (learnable) or average pooling.
    """

    def __init__(self, channels: int, use_conv: bool = True):
        """
        Initialize downsampling block.

        Args:
            channels: Number of channels
            use_conv: If True, use learnable conv; if False, use avg pooling
        """
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            # Learnable downsampling with strided convolution
            self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        else:
            # Simple average pooling
            self.conv = nn.AvgPool1d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample input by factor of 2."""
        return self.conv(x)


class Upsample1D(nn.Module):
    """
    Upsampling block for increasing sequence length by factor of 2.

    Can use either transposed convolution (learnable) or nearest neighbor interpolation.
    """

    def __init__(self, channels: int, use_conv: bool = True):
        """
        Initialize upsampling block.

        Args:
            channels: Number of channels
            use_conv: If True, use learnable conv; if False, use interpolation
        """
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            # Learnable upsampling with transposed convolution
            self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
        else:
            # Simple nearest neighbor upsampling
            self.conv = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample input by factor of 2."""
        return self.conv(x)


class UNet1D(nn.Module):
    """
    1D U-Net architecture for time series denoising in diffusion models.

    The architecture includes:
    - Time embedding for diffusion timestep conditioning
    - Optional conditioning on external signals
    - Multi-scale processing with skip connections
    - Self-attention at specified resolutions
    - Dropout for regularization
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 model_channels: int = 64,
                 num_res_blocks: int = 2,
                 attention_resolutions: tuple = (2, 4),
                 channel_mult: tuple = (1, 2, 4),
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 time_embed_dim: int = None):
        """
        Initialize the 1D U-Net model.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            model_channels: Base number of channels
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions where to apply attention
            channel_mult: Channel multipliers for each resolution
            num_heads: Number of attention heads
            dropout: Dropout probability
            time_embed_dim: Time embedding dimension
        """
        super().__init__()

        # Set default time embedding dimension
        if time_embed_dim is None:
            time_embed_dim = model_channels * 4

        # Store configuration
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.dropout = dropout

        # Time embedding network
        # Converts scalar timesteps to high-dimensional embeddings
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Condition embedding (for external conditioning signals)
        self.cond_embed = nn.Conv1d(in_channels, model_channels, 1)

        # Input projection - maps input channels to model channels
        self.input_proj = nn.Conv1d(in_channels, model_channels, 3, padding=1)

        # Build encoder (downsampling) path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        # Track channels at each resolution for skip connections
        self.skip_channels = []

        ch = model_channels
        ds = 1  # Current downsampling factor

        # Create encoder blocks at each resolution
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # Create residual blocks for this resolution level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                blocks.append(ResidualBlock1D(ch, out_ch, time_embed_dim, dropout=dropout))
                ch = out_ch

                # Add self-attention at specified resolutions
                if ds in attention_resolutions:
                    blocks.append(AttentionBlock1D(ch, num_heads))

            self.down_blocks.append(blocks)
            self.skip_channels.append(ch)

            # Add downsampling (except for the deepest level)
            if level < len(channel_mult) - 1:
                self.down_samples.append(Downsample1D(ch))
                ds *= 2
            else:
                self.down_samples.append(None)

        # Middle block (bottleneck)
        # Additional processing at the deepest resolution
        self.mid_block1 = ResidualBlock1D(ch, ch, time_embed_dim, dropout=dropout)
        self.mid_attn = AttentionBlock1D(ch, num_heads)
        self.mid_block2 = ResidualBlock1D(ch, ch, time_embed_dim, dropout=dropout)

        # Build decoder (upsampling) path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # Create decoder blocks (reverse order of encoder)
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            # Create residual blocks for this resolution level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):  # +1 for extra block after skip connection
                # First block needs to handle concatenated skip connection
                if i == 0:
                    in_ch = ch + self.skip_channels[level]  # current + skip channels
                else:
                    in_ch = out_ch

                blocks.append(ResidualBlock1D(in_ch, out_ch, time_embed_dim, dropout=dropout))
                ch = out_ch

                # Add self-attention at specified resolutions
                if ds in attention_resolutions:
                    blocks.append(AttentionBlock1D(ch, num_heads))

            self.up_blocks.append(blocks)

            # Add upsampling (except for the final output level)
            if level > 0:
                self.up_samples.append(Upsample1D(ch))
                ds //= 2
            else:
                self.up_samples.append(None)

        # Output projection - maps back to desired output channels
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                condition: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x: Noisy input tensor of shape (batch, in_channels, length)
            timesteps: Diffusion timesteps of shape (batch,)
            condition: Optional conditioning signal of shape (batch, in_channels, length)

        Returns:
            Denoised output tensor of shape (batch, out_channels, length)
        """

        # Ensure input has correct number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input has {x.shape[1]} channels, expected {self.in_channels}")

        # Generate time embedding for all timesteps
        time_emb = self.time_embed(timesteps)

        # Combine input with optional conditioning
        if condition is not None:
            # Project conditioning signal and add to input projection
            cond_emb = self.cond_embed(condition)
            h = self.input_proj(x) + cond_emb
        else:
            h = self.input_proj(x)

        # Encoder path - progressively downsample while storing skip connections
        skips = []

        for level, (blocks, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            # Apply all blocks at this resolution level
            for block in blocks:
                if isinstance(block, ResidualBlock1D):
                    h = block(h, time_emb)
                else:  # AttentionBlock1D
                    h = block(h)

            # Store for skip connection
            skips.append(h)

            # Downsample if not at deepest level
            if downsample is not None:
                h = downsample(h)

        # Middle processing at bottleneck
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Decoder path - progressively upsample while using skip connections
        for level, (blocks, upsample) in enumerate(zip(self.up_blocks, self.up_samples)):
            # Concatenate skip connection for first block
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            # Apply all blocks at this resolution level
            for block in blocks:
                if isinstance(block, ResidualBlock1D):
                    h = block(h, time_emb)
                else:  # AttentionBlock1D
                    h = block(h)

            # Upsample if not at final level
            if upsample is not None:
                h = upsample(h)

        # Final output processing
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h


# Example usage and testing
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test 1: Single channel
    print("\n=== Test 1: Single Channel  ===")
    model_single = UNet1D(
        in_channels=1,
        out_channels=1,
        model_channels=64,
        num_res_blocks=2,
        attention_resolutions=(1, 2, 4),
        channel_mult=(1, 2, 4),
        num_heads=4,
        dropout=0.1
    ).to(device)

    batch_size = 4
    seq_length = 64

    # Test with single channel input
    x_single = torch.randn(batch_size, 1, seq_length).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    condition_single = torch.randn(batch_size, 1, seq_length).to(device)

    with torch.no_grad():
        output_single = model_single(x_single, timesteps, condition_single)
        print(f"Input shape: {x_single.shape}")
        print(f"Output shape: {output_single.shape}")
        print(f"Single channel model parameters: {sum(p.numel() for p in model_single.parameters()):,}")

    # Test 2: Multi-channel
    print("\n=== Test 2: Multi-Channel ===")
    model_multi = UNet1D(
        in_channels=3,
        out_channels=3,
        model_channels=64,
        num_res_blocks=2,
        attention_resolutions=(2, 4),
        channel_mult=(1, 2, 4),
        num_heads=4,
        dropout=0.1
    ).to(device)

    # Test with multi channel input
    x_multi = torch.randn(batch_size, 3, seq_length).to(device)
    condition_multi = torch.randn(batch_size, 3, seq_length).to(device)

    with torch.no_grad():
        output_multi = model_multi(x_multi, timesteps, condition_multi)
        print(f"Input shape: {x_multi.shape}")
        print(f"Output shape: {output_multi.shape}")
        print(f"Multi-channel model parameters: {sum(p.numel() for p in model_multi.parameters()):,}")

    # Test 3: Different input/output channels
    print("\n=== Test 3: Different Input/Output Channels ===")
    model_diff = UNet1D(
        in_channels=5,
        out_channels=2,
        model_channels=64,
        num_res_blocks=2,
        attention_resolutions=(2, 4),
        channel_mult=(1, 2, 4),
        num_heads=4,
        dropout=0.2
    ).to(device)

    x_diff = torch.randn(batch_size, 5, seq_length).to(device)
    condition_diff = torch.randn(batch_size, 5, seq_length).to(device)

    with torch.no_grad():
        output_diff = model_diff(x_diff, timesteps, condition_diff)
        print(f"Input shape: {x_diff.shape}")
        print(f"Output shape: {output_diff.shape}")
        print(f"Different I/O model parameters: {sum(p.numel() for p in model_diff.parameters()):,}")

    # Test 4: No conditioning
    print("\n=== Test 4: No Conditioning ===")
    with torch.no_grad():
        output_no_cond = model_multi(x_multi, timesteps)
        print(f"Output shape (no condition): {output_no_cond.shape}")

    print("\n=== All tests completed successfully! ===")