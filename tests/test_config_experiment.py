"""Tests for LMExperimentConfig."""

from __future__ import annotations

import unittest
from pathlib import Path

from config.experiment_config import LMExperimentConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestLMExperimentConfig(unittest.TestCase):
    """LMExperimentConfig 的单元测试。"""

    def test_default_wikitext103(self):
        """默认 dataset 为 wikitext103，data_dir 自动设为对应路径。"""
        cfg = LMExperimentConfig()
        self.assertEqual(cfg.dataset, "wikitext103")
        self.assertEqual(cfg.data_dir, str(_PROJECT_ROOT / "data" / "wikitext-103"))

    def test_wikitext2_data_dir(self):
        """指定 dataset="wikitext2" 时，data_dir 自动切换。"""
        cfg = LMExperimentConfig(dataset="wikitext2")
        self.assertEqual(cfg.dataset, "wikitext2")
        self.assertEqual(cfg.data_dir, str(_PROJECT_ROOT / "data" / "wikitext-2"))

    def test_custom_data_dir(self):
        """显式 data_dir 应覆盖自动推断。"""
        custom = "/custom/path"
        cfg = LMExperimentConfig(dataset="wikitext2", data_dir=custom)
        self.assertEqual(cfg.data_dir, custom)

    def test_eval_batch_size_default(self):
        """eval_batch_size 未指定时默认为 batch_size * 2。"""
        cfg = LMExperimentConfig(batch_size=4)
        self.assertEqual(cfg.eval_batch_size, 8)

    def test_eval_batch_size_custom(self):
        """eval_batch_size 可显式指定。"""
        cfg = LMExperimentConfig(batch_size=4, eval_batch_size=16)
        self.assertEqual(cfg.eval_batch_size, 16)

    def test_to_dict_keys(self):
        """to_dict 应包含所有关键字段。"""
        cfg = LMExperimentConfig()
        d = cfg.to_dict()
        expected_keys = {
            "dim", "heads", "num_layers", "seq_len", "batch_size",
            "lr", "max_steps", "eval_interval", "warmup_steps",
            "device", "data_dir", "dataset",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_variant_label_standard(self):
        """standard 模型的 variant_label 不含 slot/chunk 信息。"""
        cfg = LMExperimentConfig(model_type="standard", dim=256, num_layers=3)
        self.assertEqual(cfg.variant_label(), "standard_d256_l3")

    def test_variant_label_mhdsra2(self):
        """mhdsra2 模型的 variant_label 应包含 slot 和 chunk 信息。"""
        cfg = LMExperimentConfig(
            model_type="mhdsra2", dim=512, num_layers=6, slots=128, chunk_size=64,
        )
        self.assertEqual(cfg.variant_label(), "mhdsra2_d512_l6_s128_c64")

    def test_resolve_torch_device_auto(self):
        """resolve_torch_device 将 "auto" 转为实际设备，不抛异常。"""
        cfg = LMExperimentConfig(device="auto")
        dev = cfg.resolve_torch_device()
        self.assertTrue(hasattr(dev, "type"))

    def test_resolve_torch_device_cpu(self):
        """显式指定 "cpu" 应返回 cpu 设备。"""
        cfg = LMExperimentConfig(device="cpu")
        self.assertEqual(str(cfg.resolve_torch_device()), "cpu")

    def test_max_chars_none(self):
        """max_chars 默认为 None。"""
        cfg = LMExperimentConfig()
        self.assertIsNone(cfg.max_chars)

    def test_max_chars_set(self):
        """max_chars 可指定整数。"""
        cfg = LMExperimentConfig(max_chars=10000)
        self.assertEqual(cfg.max_chars, 10000)

    def test_seed_default(self):
        """seed 默认值为 42。"""
        cfg = LMExperimentConfig()
        self.assertEqual(cfg.seed, 42)

    def test_all_fields_kwargs(self):
        """所有字段可通过 kwargs 正确设置。"""
        cfg = LMExperimentConfig(
            dataset="wikitext2",
            data_dir="/tmp/data",
            max_chars=5000,
            seq_len=256,
            batch_size=16,
            eval_batch_size=32,
            max_steps=100000,
            lr=1e-4,
            warmup_steps=500,
            eval_interval=2000,
            clip_grad_norm=0.5,
            dim=256,
            heads=2,
            num_layers=3,
            model_type="mhdsra2",
            slots=64,
            chunk_size=32,
            local_window_mult=2,
            device="cuda",
            seed=123,
        )
        self.assertEqual(cfg.dataset, "wikitext2")
        self.assertEqual(cfg.data_dir, "/tmp/data")
        self.assertEqual(cfg.max_chars, 5000)
        self.assertEqual(cfg.seq_len, 256)
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.eval_batch_size, 32)
        self.assertEqual(cfg.max_steps, 100000)
        self.assertEqual(cfg.lr, 1e-4)
        self.assertEqual(cfg.warmup_steps, 500)
        self.assertEqual(cfg.eval_interval, 2000)
        self.assertEqual(cfg.clip_grad_norm, 0.5)
        self.assertEqual(cfg.dim, 256)
        self.assertEqual(cfg.heads, 2)
        self.assertEqual(cfg.num_layers, 3)
        self.assertEqual(cfg.model_type, "mhdsra2")
        self.assertEqual(cfg.slots, 64)
        self.assertEqual(cfg.chunk_size, 32)
        self.assertEqual(cfg.local_window_mult, 2)
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(cfg.seed, 123)
