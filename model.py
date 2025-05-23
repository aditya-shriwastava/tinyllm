import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, context_len, embedding_size, head_size, dropout):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_size = head_size

        self.register_buffer("tril_mask", torch.tril(torch.ones(context_len, context_len)))

        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # x: (batch, seq_len, embedding_size)
        k = self.key(x) # (batch, seq_len, head_size)
        q = self.query(x) # (batch, seq_len, head_size)
        v = self.value(x) # (batch, seq_len, head_size)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embedding_size ** 0.5)  # (batch, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(self.tril_mask[:x.shape[1], :x.shape[1]] == 0, float('-inf'))  # Mask out future tokens

        # Optionally, apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Calculate the output
        output = torch.matmul(attn_weights, v)  # (batch, seq_len, head_size)
        return output


class MultiHead(nn.Module):
    def __init__(self, context_len, embedding_size, num_heads, dropout):
        super().__init__()
        assert embedding_size % num_heads == 0, f"embedding_size ({embedding_size}) must be divisible by num_heads ({num_heads})"
        head_size = embedding_size // num_heads
        self.heads = nn.ModuleList([Head(context_len, embedding_size, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.dropout(self.proj(outputs))
        return outputs


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, context_len, embedding_size, num_heads, dropout):
        super().__init__()
        self.attn_head = MultiHead(context_len, embedding_size, num_heads, dropout)
        self.ff = FeedForward(embedding_size, dropout)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.attn_head(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyLLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        embedding_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1
        ): 
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding_table = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.position_embedding_table = torch.nn.Embedding(self.context_len, self.embedding_size)
        self.blocks = nn.Sequential(
            *[Block(self.context_len, self.embedding_size, self.num_heads, dropout) for _ in range(self.num_layers)],
            nn.LayerNorm(self.embedding_size)
        )
        self.proj = torch.nn.Linear(embedding_size, vocab_size)

    def forward(self, input_seq, target=None):
        token_embeddings = self.token_embedding_table(input_seq)  # (batch, seq_len, embedding_size)
        position_embeddings = self.position_embedding_table(torch.arange(input_seq.shape[1], device=input_seq.device))  # (seq_len, embedding_size)
        x = token_embeddings + position_embeddings  # (batch, seq_len, embedding_size)

        x = self.blocks(x)

        logits = self.proj(x)  # (batch, seq_len, vocab_size)

        if target is None:
            loss = None
        else:
            # Flatten logits and targets for cross-entropy
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            target_flat = target.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)

        return logits, loss
