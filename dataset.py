from torch.utils.data import Dataset
from tokenizer import Tokenizer

# context_len: Its the length of input used to train the model at each step.
class TextDataset(Dataset):
    def __init__(self, data: str, context_len: int, tokenizer: Tokenizer):
        self.data = data
        self.context_len = context_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.__len__() + idx
        input_seq = self.data[idx:idx + self.context_len]
        target = self.data[idx + 1:idx + self.context_len + 1]
        return self.tokenizer.encode(input_seq), self.tokenizer.encode(target)


def load_datasets(
    data: str,
    context_len: int,
    tokenizer: Tokenizer,
    train_ratio: float = 0.9
):
    train_size = int(len(data) * train_ratio)
    train_dataset = TextDataset(data[:train_size], context_len=context_len, tokenizer=tokenizer)
    test_dataset = TextDataset(data[train_size:], context_len=context_len, tokenizer=tokenizer)
    return train_dataset, test_dataset
