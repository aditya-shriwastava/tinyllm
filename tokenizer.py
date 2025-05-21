from typing import List, Optional, Sequence, Union
import torch

class Tokenizer:
    def __init__(self, vocab: Optional[Sequence[str]] = None, data: Optional[str] = None) -> None:
        if vocab is not None:
            self.vocab: List[str] = list(vocab)
        elif data is not None:
            self.vocab: List[str] = sorted(set(data))
        else:
            raise ValueError("Provide either vocab or data for Tokenizer initialization.")
        self.stoi = {ch: int(i) for i, ch in enumerate(self.vocab)}
        self.itos = {int(i): ch for i, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, tokens: Union[torch.Tensor, Sequence[int]]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ''.join([self.itos[token] for token in tokens])

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> List[str]:
        return self.vocab
