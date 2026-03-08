import tiktoken
import torch
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")

VAL_DOCS = 1024 * 16


class DataLoader:
    def __init__(self, split, batch_size, block_size, device, rank=0, world_size=1):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.buf = []

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        if split == "val":
            ds = ds.take(VAL_DOCS)
        else:
            ds = ds.skip(VAL_DOCS)
            if world_size > 1:
                ds = ds.shard(num_shards=world_size, index=rank)

        self._ds = ds
        self._iter = iter(ds)

    def _fill(self, n):
        while len(self.buf) < n:
            try:
                doc = next(self._iter)
            except StopIteration:
                self._iter = iter(self._ds)
                doc = next(self._iter)
            self.buf.extend(enc.encode_ordinary(doc["text"]))

    def next_batch(self):
        n = self.batch_size * self.block_size + 1
        self._fill(n)
        tokens = torch.tensor(self.buf[:n], dtype=torch.long, device=self.device)
        self.buf = self.buf[n:]
        X = tokens[:-1].view(self.batch_size, self.block_size)
        Y = tokens[1:].view(self.batch_size, self.block_size)
        return X, Y
