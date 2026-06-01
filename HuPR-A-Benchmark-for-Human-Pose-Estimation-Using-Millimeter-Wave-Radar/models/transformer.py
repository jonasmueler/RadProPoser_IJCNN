"""Transformer module for spatiotemporal feature aggregation."""

from typing import Optional

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding that dynamically slices to input sequence length."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Parameter(
            torch.ones(1, max_seq_len, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feedforward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)

        return x


class Transformer(nn.Module):
    """Transformer encoder with input/output projections and learnable positional encoding.

    Args:
        input_dim: Dimension of input features.
        seq_len: Maximum sequence length for positional encoding.
        d_model: Internal model dimension.
        num_heads: Number of attention heads.
        d_ff: Feedforward network hidden dimension.
        num_layers: Number of transformer blocks.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 128,
        seq_len: int = 32,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = LearnablePositionalEncoding(
            seq_len, d_model
        )
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Input tensor of shape (B, seq_len, input_dim).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (B, seq_len, input_dim).
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.transformer_layers:
            x = layer(x, mask)

        return self.output_layer(x)


if __name__ == "__main__":
    seq_len = 8
    input_dim = 128
    batch_size = 64

    model = Transformer(input_dim, seq_len)
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    print(output.shape)
