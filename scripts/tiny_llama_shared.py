"""Shared tokenizer, data loading and config for tiny LLaMA comparison."""
from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader


LM_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:\"'()-[]{}<>=+*/@#$%&\n\t"
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_TOKENS = 4  # pad, bos, eos, unk

LMConfig = {
    "dim": 256,
    "heads": 4,
    "num_layers": 6,
    "vocab_size": len(LM_VOCAB) + SPECIAL_TOKENS + 1,
    "seq_len": 512,
    "batch_size": 8,
    "lr": 3e-4,
    "max_steps": 50000,
    "eval_interval": 1000,
    "warmup_steps": 1000,
    "device": "auto",
    "data_dir": str(Path(__file__).resolve().parents[1] / "data" / "wikitext-2"),
    "dataset": "wikitext2",  # "wikitext2" or "wikitext103"
}


class CharTokenizer:
    """Character-level tokenizer for LM comparison."""

    def __init__(self, vocab: str = LM_VOCAB):
        self.vocab = vocab
        self._char_to_id = {
            ch: i + SPECIAL_TOKENS for i, ch in enumerate(vocab)
        }
        self._id_to_char = {
            i + SPECIAL_TOKENS: ch for i, ch in enumerate(vocab)
        }
        self.vocab_size = len(vocab) + SPECIAL_TOKENS + 1

    def encode(self, text: str) -> list[int]:
        ids = [BOS_ID]
        for ch in text:
            ids.append(self._char_to_id.get(ch, UNK_ID))
        ids.append(EOS_ID)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        chars = []
        for i in ids:
            if i < SPECIAL_TOKENS:
                continue
            chars.append(self._id_to_char.get(i, "?"))
        return "".join(chars)


def download_wikitext2(data_dir: str) -> Path:
    """Download WikiText-2 if not present. Returns path to raw text file."""
    data_path = Path(data_dir)
    raw_path = data_path / "wiki.train.tokens"
    if raw_path.exists():
        return raw_path

    data_path.mkdir(parents=True, exist_ok=True)
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    zip_path = data_path / "wikitext-2-v1.zip"

    print(f"Downloading WikiText-2 from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    return data_path / "wikitext-2" / "wiki.train.tokens"


def download_wikitext103(data_dir: str) -> Path:
    """Download WikiText-103 if not present. Returns path to raw text file."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    # Check for any .tokens file
    for f in data_path.iterdir():
        if f.suffix == ".tokens":
            return f
    wt2 = download_wikitext2(data_dir)
    return wt2


def load_text(file_path: Path, max_chars: int | None = None) -> str:
    """Load text file, optionally truncated."""
    text = file_path.read_text(encoding="utf-8")
    if max_chars:
        text = text[:max_chars]
    return text


class LMDataset(Dataset):
    """Language modeling dataset with fixed-length sequences."""

    def __init__(self, tokenizer: CharTokenizer, text: str, seq_len: int):
        self.seq_len = seq_len
        ids = tokenizer.encode(text)
        self.sequences: list[torch.Tensor] = []
        for start in range(0, len(ids) - seq_len - 1, seq_len):
            chunk = ids[start: start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.sequences.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]  # x, targets


def create_dataloader(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LMDataset(tokenizer, text, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
