from torch.utils.data import Dataset
from tokenizer import Tokenizer

# block_size: Its the length of input used to train the model at each step.
class TextDataset(Dataset):
    def __init__(self, data: str, block_size: int, tokenizer: Tokenizer):
        self.data = data
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.__len__() + idx
        input_seq = self.data[idx:idx + self.block_size]
        target = self.data[idx + 1:idx + self.block_size + 1]
        return self.tokenizer.encode(input_seq), self.tokenizer.encode(target)


def load_datasets(
    data: str,
    block_size: int,
    tokenizer: Tokenizer,
    train_ratio: float = 0.9
):
    train_size = int(len(data) * train_ratio)
    train_dataset = TextDataset(data[:train_size], block_size=block_size, tokenizer=tokenizer)
    test_dataset = TextDataset(data[train_size:], block_size=block_size, tokenizer=tokenizer)
    return train_dataset, test_dataset
