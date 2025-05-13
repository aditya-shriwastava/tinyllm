#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset


# block_size: Its the length of input used to train the model at each step.
class TinyShakespeareDataset(Dataset):
    def __init__(self , file_path: str, block_size: int):
        self.file_path = file_path
        self.block_size = block_size

        with open(file_path, 'r') as f:
            self.data = f.read()

    def __len__(self):
        return len(self.data) - self.block_size + 1

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.__len__() + idx
        input_seq = self.data[idx:idx + self.block_size]
        return input_seq


def main():
    tiny_shakespeare_dataset = TinyShakespeareDataset(file_path='./tinyshakespeare.txt', block_size=128)
    print(tiny_shakespeare_dataset[0])

if __name__ == '__main__':
    main()
