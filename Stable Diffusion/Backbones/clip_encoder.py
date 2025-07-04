import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention layer."""
    def __init__(self, n_head, n_embed, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        head_size = C // self.n_head
        q, k, v = [t.view(B, T, self.n_head, head_size).transpose(1, 2) for t in (q, k, v)]
        
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-normalization."""
    def __init__(self, n_head, n_embed, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, n_embed, dropout)
        self.ln_1 = nn.LayerNorm(n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class TextTransformer(nn.Module):
    """The Text Encoder for CLIP."""
    def __init__(self, n_layers, n_head, n_embed, vocab_size, context_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Parameter(torch.zeros(1, context_length, n_embed))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(n_head, n_embed, dropout) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(n_embed)
        self.register_buffer("causal_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())

    def forward(self, x):
        token_embs = self.token_embedding(x)
        pos_embs = self.positional_embedding[:, :x.size(1), :]
        x = token_embs + pos_embs
        
        mask = self.causal_mask[:x.size(1), :x.size(1)]
        
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
            
        return self.ln_final(x)