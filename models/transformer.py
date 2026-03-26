import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Parameter(torch.ones(1, max_seq_len, d_model))  # Max sequence length

    def forward(self, x):
        
        return x + self.pos_embedding  # Dynamically slice position embedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)

        # Feedforward network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim=128, seq_len=16, d_model=256, num_heads=8, d_ff=512, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)  # Input projection to d_model
        self.positional_encoding = LearnablePositionalEncoding(seq_len, d_model)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, input_dim)  # Output projection back to input_dim

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.transformer_layers:
            x = layer(x, mask)

        return self.output_layer(x)


if __name__ == "__main__":
    # Example usage
    seq_len = 8
    input_dim = 128
    batch_size = 64

    model = Transformer(input_dim, seq_len)
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    print(output.shape)  # Expected output: (batch_size, seq_len, input_dim)