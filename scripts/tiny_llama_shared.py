"""Shared tokenizer, data loading and config for tiny LLaMA comparison."""
from __future__ import annotations

import hashlib
import shutil
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
WIKITEXT2_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
WIKITEXT2_MAX_EXTRACTED_BYTES = 128 * 1024 * 1024
WIKITEXT2_MAX_MEMBER_BYTES = 64 * 1024 * 1024
WIKITEXT2_MAX_ZIP_MEMBERS = 32
WIKITEXT_HF_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT103_REVISION = WIKITEXT_HF_REVISION

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


def file_sha256(file_path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest for a local file.

    中文说明:
    - 调用方 / Called by: `validate_downloaded_archive` and tests.
    - 调用对象 / Calls: `hashlib.sha256`, `Path.open`.
    - 作用 / Purpose: 下载后用固定摘要校验数据包，防止静默篡改或错误 XML 被当作数据。
    - 参数 / Parameters: `file_path` 是本地文件；`chunk_size` 控制流式读取块大小。
    - 返回 / Returns: 小写十六进制 SHA256 摘要。
    - 错误处理 / Error handling: 文件读取错误直接向上抛出。
    - 副作用 / Side effects: 只读文件。

    English documentation:
    Function name:
        file_sha256
    Purpose:
        Compute a streaming SHA256 digest for a local file.
    Called by:
        Download validation helpers and tests.
    Calls:
        `hashlib.sha256` and file reads.
    Parameters:
        - file_path: file to hash.
        - chunk_size: bytes read per iteration.
    Returns:
        Lowercase hexadecimal SHA256 digest.
    Error handling:
        Propagates filesystem read errors.
    Side effects:
        Reads the file only.
    """
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_downloaded_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
    max_archive_bytes: int,
    label: str,
) -> None:
    """Validate archive size, SHA256 and ZIP structure before extraction.

    中文说明:
    - 调用方 / Called by: future ZIP dataset loaders and tests.
    - 调用对象 / Calls: `Path.stat`, `file_sha256`, `zipfile.is_zipfile`.
    - 作用 / Purpose: 把下载文件当作“封条包裹”检查，确认大小合理、摘要匹配且确实是 ZIP。
    - 参数 / Parameters: `archive_path` 是下载文件；`expected_sha256` 是固定摘要；
      `max_archive_bytes` 是下载包大小上限；`label` 用于错误信息。
    - 返回 / Returns: None.
    - 错误处理 / Error handling: 缺失、超限、摘要不符或非 ZIP 时抛 `RuntimeError`。
    - 副作用 / Side effects: 只读文件，不解压。

    English documentation:
    Function name:
        validate_downloaded_archive
    Purpose:
        Check archive size, hash and ZIP magic before extraction.
    Called by:
        Future ZIP dataset loaders and tests.
    Calls:
        `Path.stat`, `file_sha256`, and `zipfile.is_zipfile`.
    Parameters:
        Download path, expected digest, size limit and diagnostic label.
    Returns:
        None.
    Error handling:
        Raises `RuntimeError` when the downloaded archive is unsafe or unexpected.
    Side effects:
        Reads the downloaded file only.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise RuntimeError(f"{label} archive was not created: {archive_path}")
    archive_size = archive_path.stat().st_size
    if archive_size <= 0:
        raise RuntimeError(f"{label} archive is empty: {archive_path}")
    if archive_size > max_archive_bytes:
        raise RuntimeError(
            f"{label} archive exceeds size limit: {archive_size} > {max_archive_bytes} bytes"
        )
    actual_sha256 = file_sha256(archive_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"{label} archive SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"{label} archive is not a valid ZIP file: {archive_path}")


def _resolve_zip_member_path(destination: Path, member_name: str) -> Path:
    """Resolve one ZIP member path and reject traversal attempts.

    中文说明:
    - 调用方 / Called by: `safe_extract_zip`.
    - 调用对象 / Calls: `Path.resolve`, `Path.is_relative_to`.
    - 作用 / Purpose: 防止 ZIP 条目名通过 `../`、绝对路径或 Windows 盘符写到目标目录外。
    - 参数 / Parameters: `destination` 是解压根目录；`member_name` 是 ZIP 条目名。
    - 返回 / Returns: 安全的目标文件路径。
    - 错误处理 / Error handling: 路径逃逸时抛 `ValueError`。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        _resolve_zip_member_path
    Purpose:
        Resolve a ZIP member under the destination directory.
    Called by:
        `safe_extract_zip`.
    Calls:
        `Path.resolve` and `Path.is_relative_to`.
    Parameters:
        Extraction root and member name.
    Returns:
        Safe target path.
    Error handling:
        Raises `ValueError` for path traversal or absolute paths.
    Side effects:
        None.
    """
    raw_name = Path(member_name)
    if raw_name.is_absolute() or raw_name.drive or raw_name.root:
        raise ValueError(f"ZIP member uses an absolute path: {member_name!r}")
    target_path = (destination / raw_name).resolve()
    if not target_path.is_relative_to(destination.resolve()):
        raise ValueError(f"ZIP member escapes extraction directory: {member_name!r}")
    return target_path


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    max_total_bytes: int,
    max_member_bytes: int,
    max_members: int = WIKITEXT2_MAX_ZIP_MEMBERS,
    max_compression_ratio: int = 100,
) -> None:
    """Extract a ZIP archive after path and resource-limit validation.

    中文说明:
    - 调用方 / Called by: future ZIP dataset loaders; tests use it to cover hostile ZIP files.
    - 调用对象 / Calls: `zipfile.ZipFile`, `_resolve_zip_member_path`, `shutil.copyfileobj`.
    - 作用 / Purpose: 替代裸 `extractall`，避免 Zip Slip、超大文件和压缩炸弹风险。
    - 参数 / Parameters: `archive_path` 是 ZIP 文件；`destination` 是目标目录；
      `max_total_bytes` 是所有文件解压后总大小上限；`max_member_bytes` 是单文件上限；
      `max_members` 限制条目数量；`max_compression_ratio` 限制单条目膨胀倍率。
    - 返回 / Returns: None.
    - 错误处理 / Error handling: 路径逃逸、文件超限、疑似压缩炸弹或坏 ZIP 抛异常。
    - 副作用 / Side effects: 校验通过后写入目标目录。

    English documentation:
    Function name:
        safe_extract_zip
    Purpose:
        Safely extract a ZIP archive with path and resource checks.
    Called by:
        Future ZIP dataset loaders and tests.
    Calls:
        ZIP metadata APIs, path resolver, and streaming copy.
    Parameters:
        Archive path, destination, total/member byte/member count limits and compression ratio limit.
    Returns:
        None.
    Error handling:
        Raises on traversal, size limit breaches, suspicious compression ratios or bad ZIPs.
    Side effects:
        Writes extracted files only after metadata validation succeeds.
    """
    archive_path = Path(archive_path)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        members = zip_file.infolist()
        if len(members) > max_members:
            raise ValueError(f"ZIP member count exceeds limit: {len(members)} > {max_members}")
        total_size = 0
        safe_members = []
        for member in members:
            target_path = _resolve_zip_member_path(destination, member.filename)
            if member.is_dir():
                safe_members.append((member, target_path))
                continue
            if member.file_size < 0:
                raise ValueError(f"ZIP member has invalid size: {member.filename!r}")
            if member.file_size > max_member_bytes:
                raise ValueError(
                    f"ZIP member exceeds size limit: {member.filename!r} "
                    f"{member.file_size} > {max_member_bytes} bytes"
                )
            total_size += member.file_size
            if total_size > max_total_bytes:
                raise ValueError(
                    f"ZIP extracted size exceeds limit: {total_size} > {max_total_bytes} bytes"
                )
            if member.compress_size > 0:
                ratio = member.file_size / member.compress_size
                if ratio > max_compression_ratio:
                    raise ValueError(
                        f"ZIP member compression ratio is too high: {member.filename!r}"
                    )
            safe_members.append((member, target_path))

        for member, target_path in safe_members:
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def download_wikitext2(data_dir: str) -> Path:
    """Download WikiText-2 from a pinned HuggingFace dataset revision.

    中文说明:
    - 调用方 / Called by: `load_wikitext2_splits`.
    - 调用对象 / Calls: `datasets.load_dataset`, `Path.write_text`.
    - 作用 / Purpose: 首次运行 tiny LLaMA 对比时从固定 commit 下载 WikiText-2；
      旧 S3 ZIP 源已返回重定向/证书错误，因此不再默认依赖裸 ZIP 解压路径。
    - 参数 / Parameters: `data_dir` 是缓存目录。
    - 返回 / Returns: `wiki.train.tokens` 路径。
    - 错误处理 / Error handling: `datasets` 缺失、下载失败或文件写出失败时抛 `RuntimeError`。
    - 副作用 / Side effects: 首次运行会写出 train/valid/test `.tokens` 文件。

    English documentation:
    Function name:
        download_wikitext2
    Purpose:
        Download WikiText-2 from a pinned HuggingFace dataset revision.
    Called by:
        `load_wikitext2_splits`.
    Calls:
        `datasets.load_dataset` and text file writes.
    Parameters:
        - data_dir: cache directory.
    Returns:
        Path to `wiki.train.tokens`.
    Error handling:
        Raises when the datasets package, download or file writing fails.
    Side effects:
        Writes train/valid/test split files.
    """
    data_path = Path(data_dir)
    legacy_raw_path = data_path / "wiki.train.tokens"
    extracted_raw_path = data_path / "wikitext-2" / "wiki.train.tokens"
    if legacy_raw_path.exists():
        return legacy_raw_path
    if extracted_raw_path.exists():
        return extracted_raw_path

    data_path.mkdir(parents=True, exist_ok=True)
    extracted_raw_path.parent.mkdir(parents=True, exist_ok=True)
    print("Loading WikiText-2 via HuggingFace datasets...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            revision=WIKITEXT_HF_REVISION,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download WikiText-2 via datasets: {e}") from e

    split_paths = {
        "train": extracted_raw_path.parent / "wiki.train.tokens",
        "validation": extracted_raw_path.parent / "wiki.valid.tokens",
        "test": extracted_raw_path.parent / "wiki.test.tokens",
    }
    for split_name, output_path in split_paths.items():
        try:
            text = "\n".join(dataset[split_name]["text"])
            output_path.write_text(text, encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to write WikiText-2 {split_name} split: {e}") from e

    if not extracted_raw_path.exists():
        raise RuntimeError(f"WikiText-2 train file missing after download: {extracted_raw_path}")
    return extracted_raw_path


def load_wikitext2_splits(data_dir: str) -> dict[str, Path | None]:
    """Return WikiText-2 split paths, downloading the archive if needed.

    中文说明:
    - 调用方 / Called by: `scripts.tiny_llama_baseline`, `scripts.tiny_llama_mhdsra2`.
    - 调用对象 / Calls: `download_wikitext2`, `Path.exists`.
    - 作用 / Purpose: 为 tiny LLaMA 对比提供明确 train/validation split；若旧缓存缺少官方
      valid/test 文件，则由调用方对训练文本做固定尾部分割，避免把训练 batch loss 当成验证 PPL。
    - 参数 / Parameters: `data_dir` 是 WikiText-2 数据缓存目录。
    - 返回 / Returns: `{"train": Path, "valid": Path|None, "test": Path|None}`。
    - 错误处理 / Error handling: 若 train 文件不存在，抛出 `RuntimeError`。
    - 副作用 / Side effects: 首次运行可能下载并解压 WikiText-2。

    English documentation:
    Function name:
        load_wikitext2_splits
    Purpose:
        Resolve explicit train/validation/test files for WikiText-2.
    Called by:
        tiny LLaMA training entry points.
    Calls:
        `download_wikitext2`, `Path.exists`.
    Parameters:
        - data_dir: WikiText-2 cache directory.
    Returns:
        Dict with train path and optional valid/test paths.
    Error handling:
        Raises `RuntimeError` when the train split is missing.
    Side effects:
        May download and extract WikiText-2.
    English keywords:
        wikitext2, split, train, validation, perplexity
    """
    train_path = download_wikitext2(data_dir)
    split_dir = train_path.parent
    valid_path = split_dir / "wiki.valid.tokens"
    test_path = split_dir / "wiki.test.tokens"
    if not train_path.exists():
        raise RuntimeError(f"WikiText-2 train file missing after download: {train_path}")
    return {
        "train": split_dir / "wiki.train.tokens",
        "valid": valid_path if valid_path.exists() else None,
        "test": test_path if test_path.exists() else None,
    }


def split_train_validation_text(
    text: str,
    *,
    validation_chars: int = 200_000,
) -> tuple[str, str]:
    """Split one train-only text file into train and validation tails.

    中文说明:
    - 调用方 / Called by: tiny LLaMA scripts when WikiText-2 valid split is absent.
    - 调用对象 / Calls: string slicing.
    - 作用 / Purpose: 兼容旧 `data/wikitext-2/wiki.train.tokens` 缓存，同时保证 PPL 来自未训练尾部文本。
    - 参数 / Parameters: `text` 是完整训练文本；`validation_chars` 是尾部验证字符预算。
    - 返回 / Returns: `(train_text, validation_text)`。
    - 错误处理 / Error handling: 文本过短时仍至少保留一半用于训练、一半用于验证。
    - 副作用 / Side effects: 无。

    English documentation:
    Function name:
        split_train_validation_text
    Purpose:
        Build a deterministic validation tail when only a train file exists.
    Called by:
        tiny LLaMA scripts.
    Calls:
        String slicing.
    Parameters:
        - text: full training text.
        - validation_chars: validation tail budget.
    Returns:
        `(train_text, validation_text)`.
    Error handling:
        Short text is split in half to keep both sides non-empty.
    Side effects:
        None.
    English keywords:
        validation split, fallback, train tail, perplexity
    """
    if len(text) < 2:
        raise ValueError("Cannot split train/validation text with fewer than two characters.")
    split_size = min(max(1, int(validation_chars)), max(1, len(text) // 5))
    split_at = max(1, len(text) - split_size)
    return text[:split_at], text[split_at:]


def download_wikitext103(data_dir: str) -> dict[str, Path]:
    """Download WikiText-103 via HuggingFace datasets, cache as .tokens files.

    通过 HuggingFace datasets 库下载 WikiText-103，缓存为 .tokens 文件。
    S3 源已失效，改用 datasets 库加载后写出到本地文件。

    Returns {"train": Path, "valid": Path, "test": Path}.
    Raises RuntimeError if download fails.

    Returns:
        {"train": Path, "valid": Path, "test": Path}

    Raises:
        RuntimeError: 下载或写出文件失败时抛出
    """
    data_path = Path(data_dir)
    extract_path = data_path / "wikitext-103"
    splits_map = {
        "train": extract_path / "wiki.train.tokens",
        "valid": extract_path / "wiki.valid.tokens",
        "test": extract_path / "wiki.test.tokens",
    }

    # 如果三个切分文件都已存在，直接返回
    if all(p.exists() for p in splits_map.values()):
        return splits_map

    data_path.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    print("Loading WikiText-103 via HuggingFace datasets...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            revision=WIKITEXT_HF_REVISION,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download WikiText-103 via datasets: {e}") from e

    hf_split_map = {"train": "train", "valid": "validation", "test": "test"}
    for key, path in splits_map.items():
        hf_split = hf_split_map[key]
        print(f"  Writing {key} split ({len(dataset[hf_split])} lines) to {path}...")
        try:
            text = "\n".join(dataset[hf_split]["text"])
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            raise RuntimeError(f"Failed to write {key} split to {path}: {e}") from e

    # 验证文件完整性
    missing = [name for name, p in splits_map.items() if not p.exists()]
    if missing:
        raise RuntimeError(
            f"WikiText-103 file writing incomplete: missing splits {missing}. "
            f"Expected files in {extract_path}"
        )

    return splits_map


def load_text(file_path: Path, max_chars: int | None = None) -> str:
    """Load text file, optionally truncated."""
    text = file_path.read_text(encoding="utf-8")
    if max_chars:
        text = text[:max_chars]
    return text


class LMDataset(Dataset):
    """Language modeling dataset with fixed-length sequences.

    内存高效的 LM 数据集实现。将所有 token ID 存储在单个连续 Tensor 中，
    __getitem__ 通过切片直接索引，避免预创建大量独立 Tensor。

    Memory-efficient LM dataset. Stores all token IDs in a single contiguous
    tensor and slices on-the-fly in __getitem__.
    """

    def __init__(self, tokenizer: CharTokenizer, text: str, seq_len: int):
        self.seq_len = seq_len
        # 避免创建 Python list（518M ints ≈ 14.5GB 峰值内存）
        # 使用 numpy 向量化转换字符到 token ID
        import numpy as np
        n_tokens = len(text) + 2
        # 构建 ASCII 查找表 (0-127)
        lookup = np.full(128, UNK_ID, dtype=np.int64)
        for ch, tid in tokenizer._char_to_id.items():
            c = ord(ch)
            if c < 128:
                lookup[c] = tid
        # 向量化转换：text → bytes → ids
        raw = text.encode("ascii", errors="replace")
        ids_array = lookup[np.frombuffer(raw, dtype=np.uint8)]
        # 组装为 torch Tensor（含 BOS/EOS）
        self.ids = torch.empty(n_tokens, dtype=torch.long)
        self.ids[0] = BOS_ID
        self.ids[1:-1] = torch.from_numpy(ids_array)
        self.ids[-1] = EOS_ID
        self.num_sequences = (n_tokens - 1) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        return (
            self.ids[start: start + self.seq_len],
            self.ids[start + 1: start + self.seq_len + 1],
        )


def create_dataloader(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LMDataset(tokenizer, text, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def create_eval_loader(
    text: str,
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    """Create a non-shuffled DataLoader for evaluation (no drop_last).

    创建用于评测的 DataLoader，不 shuffle、不 drop last，保证所有序列都被评估。

    Args:
        text: 原始文本
        tokenizer: 字符级 tokenizer
        seq_len: 序列长度
        batch_size: 批次大小

    Returns:
        不 shuffle、不 drop_last 的 DataLoader
    """
    dataset = LMDataset(tokenizer, text, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda":
        return torch.device("cuda:0")
    device = torch.device(device_str)
    if device.type == "cuda" and device.index not in (0, None):
        raise ValueError("Only cuda:0 is supported by this project.")
    return torch.device("cuda:0") if device.type == "cuda" else device
