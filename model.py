import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, block_size, embedding_size, head_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.head_size = head_size

        self.register_buffer("tril_mask", torch.tril(torch.ones(block_size, block_size)))

        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

    def forward(self, x): # x: (batch, seq_len, embedding_size)
        k = self.key(x) # (batch, seq_len, head_size)
        q = self.query(x) # (batch, seq_len, head_size)
        v = self.value(x) # (batch, seq_len, head_size)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embedding_size ** 0.5)  # (batch, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(self.tril_mask[:x.shape[1], :x.shape[1]] == 0, float('-inf'))  # Mask out future tokens

        # Optionally, apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len, seq_len)

        # Calculate the output
        output = torch.matmul(attn_weights, v)  # (batch, seq_len, head_size)
        return output


class TinyLLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embedding_size: int,
        head_size: int
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_size = embedding_size
        self.head_size = head_size

        self.token_embedding_table = torch.nn.Embedding(self.vocab_size, self.embedding_size)
        self.position_embedding_table = torch.nn.Embedding(self.block_size, self.embedding_size)
        self.attn_head = Head(block_size, embedding_size, head_size)
        self.proj = torch.nn.Linear(head_size, vocab_size)

    def forward(self, input_seq, target=None):
        token_embeddings = self.token_embedding_table(input_seq)  # (batch, seq_len, embedding_size)
        position_embeddings = self.position_embedding_table(torch.arange(input_seq.shape[1], device=input_seq.device))  # (seq_len, embedding_size)
        x = token_embeddings + position_embeddings  # (batch, seq_len, embedding_size)

        # Pass through the attention head
        attn_out = self.attn_head(x)  # (batch, seq_len, head_size)

        logits = self.proj(attn_out)  # (batch, seq_len, vocab_size)

        if target is None:
            loss = None
        else:
            # Flatten logits and targets for cross-entropy
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            target_flat = target.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)

        return logits, loss
