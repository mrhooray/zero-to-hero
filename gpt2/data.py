from os.path import abspath, dirname, join

import tiktoken
import torch

DATA_PATH = join(dirname(abspath(__file__)), "input.txt")

enc = tiktoken.get_encoding("gpt2")

with open(DATA_PATH) as f:
    text = f.read()

tokens = torch.tensor(enc.encode(text), dtype=torch.long)

n = int(0.9 * len(tokens))
_train = tokens[:n]
_val = tokens[n:]


class DataLoader:
    def __init__(self, split, batch_size, block_size, device):
        self.data = _train if split == "train" else _val
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

    def next_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        offsets = torch.arange(self.block_size)
        X = self.data[ix.unsqueeze(1) + offsets].to(self.device)
        Y = self.data[ix.unsqueeze(1) + offsets + 1].to(self.device)
        return X, Y
