import csv
import math
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# TODO: Replace this
# Using a very naive tokenization scheme for now so we can train for now.
# PAD is just to make tensor rectangular, always start with BOS and end with EOS.
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 256 + BYTE_OFFSET


def hex_to_tokens(hex_str: str) -> list[int]:
    """Convert a hex string to a list of token IDs with BOS/EOS."""
    # remove once we have proper tokenization
    byte_vals = bytes.fromhex(hex_str)
    return [BOS_ID] + [b + BYTE_OFFSET for b in byte_vals] + [EOS_ID]


class BHiveDataset(Dataset):
    """Dataset for bhive throughput data with naive tokenization."""

    def __init__(
        self,
        csv_path: str | Path,
        max_seq_len: int = 512,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        log_targets: bool = True,
    ):
        csv_path = Path(csv_path)
        samples: list[tuple[str, float]] = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                hex_str, throughput = row[0], float(row[1])
                if not hex_str:
                    continue
                # +2 for BOS/EOS
                if len(hex_str) // 2 + 2 > max_seq_len:
                    continue
                samples.append((hex_str, throughput))

        # Deterministic shuffle and split
        # TODO: Later we should just use canonical split? @henry
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=gen).tolist()
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.items: list[tuple[list[int], float]] = []
        for i in selected:
            hex_str, throughput = samples[i]
            tokens = hex_to_tokens(hex_str)
            target = math.log(throughput) if log_targets else throughput
            self.items.append((tokens, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        tokens, target = self.items[idx]
        return torch.tensor(tokens, dtype=torch.long), len(tokens), target


def collate_fn(
    batch: list[tuple[torch.Tensor, int, float]],
) -> dict[str, torch.Tensor]:
    """Pad sequences and return input_ids, lengths, and targets."""
    token_seqs, lengths, targets = zip(*batch, strict=True)
    input_ids = pad_sequence(list(token_seqs), batch_first=True, padding_value=PAD_ID)
    return {
        "input_ids": input_ids,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.float32),
    }

class CollateLM:
    """
    Pad sequences and return input-ids, attention_mask, and labels (with pad positions masked as -100).
    Drops sequences that failed disassembly.
    """

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = [x for x in batch if x.numel() > 0]
        if len(batch) == 0:
            dummy = torch.tensor([[self.pad_id]], dtype=torch.long)
            return {
                "input_ids": dummy,
                "attention_mask": (dummy != self.pad_id).long(),
                "labels": torch.full_like(dummy, -100),
            }

        input_ids = pad_sequence(batch, batch_first=True, padding_value=self.pad_id)
        attention_mask = (input_ids != self.pad_id).long()

        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}