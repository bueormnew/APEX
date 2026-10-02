"""Dependency-light byte tokenizer and random next-token text windows."""

import json
from pathlib import Path
from typing import Iterable, List, Union

import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """UTF-8 byte tokenizer with reserved padding, BOS and EOS ids."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    byte_offset = 3
    vocab_size = 259

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        tokens = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_bos:
            tokens.insert(0, self.bos_token_id)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: Iterable[int], skip_special_tokens: bool = True) -> str:
        byte_values = []
        for token in tokens:
            token = int(token)
            if self.byte_offset <= token < self.vocab_size:
                byte_values.append(token - self.byte_offset)
            elif not skip_special_tokens and token in (self.bos_token_id, self.eos_token_id):
                continue
        return bytes(byte_values).decode("utf-8", errors="replace")

    def save(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps({"type": "utf8-byte", "vocab_size": self.vocab_size}, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(file_path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ByteTokenizer":
        metadata = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if metadata != {"type": "utf8-byte", "vocab_size": cls.vocab_size}:
            raise ValueError("Unsupported APEX tokenizer format")
        return cls()


def read_corpus(path: Union[str, Path]) -> str:
    """Read one text file or concatenate supported text files below a directory."""
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        extensions = {".txt", ".md", ".csv", ".json", ".html", ".log"}
        files = sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in extensions)
    else:
        raise FileNotFoundError(f"Corpus path does not exist: {path}")
    if not files:
        raise ValueError(f"No supported text files found in {path}")
    return "\n".join(file.read_text(encoding="utf-8") for file in files)


class RandomTextWindowDataset(Dataset):
    """Sample next-token windows from a one-dimensional token stream."""

    def __init__(
        self,
        tokens: Union[List[int], torch.Tensor],
        sequence_length: int,
        samples: int,
        random_sampling: bool = True,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.sequence_length = sequence_length
        self.samples = samples
        self.random_sampling = random_sampling
        self.max_start = self.tokens.numel() - sequence_length - 1
        if self.max_start < 0:
            raise ValueError(
                f"Need at least sequence_length + 1 tokens ({sequence_length + 1}), "
                f"got {self.tokens.numel()}"
            )

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int):
        if self.random_sampling:
            start = int(torch.randint(self.max_start + 1, ()).item())
        elif self.samples <= 1:
            start = self.max_start // 2
        else:
            start = round(index * self.max_start / (self.samples - 1))
        window = self.tokens[start : start + self.sequence_length + 1]
        return window[:-1], window[1:]


def make_validation_dataset(tokens: List[int], sequence_length: int, batches: int, batch_size: int):
    return RandomTextWindowDataset(
        tokens,
        sequence_length=sequence_length,
        samples=max(1, batches * batch_size),
        random_sampling=False,
    )
