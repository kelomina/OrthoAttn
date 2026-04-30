import unittest

import torch

from scripts.ablation_study import build_model


class TestAblationMHDSRA2(unittest.TestCase):
    """Regression tests for MHDSRA2 ablation model construction.

    中文说明:
    - 调用方 / Called by: `python -m unittest` and `pytest`
    - 调用对象 / Calls: `scripts.ablation_study.build_model`
    - 作用 / Purpose: 确认旧 DSRA 消融入口已经迁移到 MHDSRA2 local/retrieval/window 配置
    - 变量 / Variables: `model` 为消融构建出的 MHDSRA2 模型, `cfg` 为底层 attention 配置
    - 接入 / Integration: 新增消融开关时在本测试补充对应断言
    - 错误处理 / Error handling: 使用断言暴露架构回退或配置未传递问题
    - 关键词 / Keywords:
      ablation|mhdsra2|build_model|local|retrieval|window|config|regression|test|消融
    """

    def test_build_model_passes_mhdsra2_ablation_flags(self) -> None:
        """Validate ablation flags reach the underlying MHDSRA2 layer.

        中文说明:
        - 调用方 / Called by: `unittest`
        - 调用对象 / Calls: `build_model`
        - 作用 / Purpose: 确认消融配置不再构造旧 DSRA 路径, 而是传入 MHDSRA2 配置
        - 变量 / Variables: `config` 为消融开关, `cfg` 为第一层 MHDSRA2 配置
        - 接入 / Integration: 保护 `ablation_study.py` 的模型工厂迁移结果
        - 错误处理 / Error handling: 配置缺失或错误映射会触发断言失败
        - 关键词 / Keywords:
          flags|mhdsra2|ablation|factory|use_local|use_retrieval|local_window|layer|assert|工厂
        """
        model = build_model(
            config={"use_local": False, "use_retrieval": False, "local_window": 3},
            vocab_size=32,
            dim=16,
            K=8,
            kr=2,
            chunk_size=4,
            device=torch.device("cpu"),
        )
        cfg = model.dsra.layer.cfg

        self.assertEqual(model.architecture, "mhdsra2")
        self.assertFalse(cfg.use_local)
        self.assertFalse(cfg.use_retrieval)
        self.assertEqual(cfg.local_window, 3)


if __name__ == "__main__":
    unittest.main()
