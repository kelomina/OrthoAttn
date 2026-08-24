"""RULER NIAH 数据集规范与数据生成器（对齐 NVIDIA/RULER 官方规格）.

RULER Needle-In-A-Haystack Domain Specification and Data Generator (aligned with NVIDIA/RULER).

中文说明:
- 调用方 / Called by:
  * `scripts.benchmark_ruler_niah` (RULER-NIAH 基准评测脚本)
  * `tests.test_ruler_niah_data_generation` (生成器单元与边界测试套件)
- 参考来源 / Reference:
  * NVIDIA/RULER (COLM 2024, arXiv:2404.06654), `scripts/data/synthetic/niah.py` 与 `constants.py`
  * Kamradt LLMTest_NeedleInAHaystack (Paul Graham essays 变体的语料来源)
- 对齐声明 / Alignment Notes:
  1. 噪声海草句与针句模板逐字对齐官方实现:
     - 噪声句: "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
     - 针句:   "One of the special magic {type_v} for {key} is: {value}."
     - 上下文以 "\n" 连接句子; 针句在随机位置插入(排序后逆序 insert);
     - Key 为 "{adjective}-{noun}" 复合词; S-NIAH-1 的 Value 为 7 位十进制数字;
     - 查询从 K 个 key 中无放回采样 Q 个, 答案为对应 value 序列。
  2. 训练适配偏差 (Train-from-scratch Adaptation Deviations, 均已文档化):
     - 分词采用词级封闭词表(含数字位 token), 替代官方 cl100k/nemo tokenizer;
       因此序列长度按"词 token"计而非 BPE token;
     - wonderwords 词库截断为内嵌静态形容词/名词表(保持 "{adj}-{noun}" 组合空间);
     - 评测口径为答案数字序列的精确匹配(Teacher-forcing 下逐步 Top-1 / 序列全对),
       替代官方 LLM 文本生成 + 字符串匹配。
  3. 损失掩码: 仅答案数字位置参与损失(Y=对应数字 token id), 其余位置 Y=PAD(=0,
     配合 CrossEntropyLoss(ignore_index=0)), 与 Zoology MQAR 掩码约定一致。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# 官方规格常量 / Official spec constants (verbatim from NVIDIA/RULER niah.py)
# ---------------------------------------------------------------------------

#: RULER repeat-noise 海草句 / RULER repeat-noise haystack sentence
NOISE_SENTENCE: str = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

#: RULER 针句模板 / RULER needle sentence template
NEEDLE_TEMPLATE: str = "One of the special magic {type_v} for {key} is: {value}."

#: S-NIAH-1 数值位数 / number of digits for S-NIAH-1 values
VALUE_NUM_DIGITS: int = 7

# ---------------------------------------------------------------------------
# 封闭词表素材 / Closed vocabulary material
# ---------------------------------------------------------------------------

#: 形容词表(wonderwords adjectivelist 的静态子集, 与功能词无重叠)
ADJECTIVES: List[str] = [
    "ancient", "brave", "calm", "clever", "bold", "curious", "eager", "fierce",
    "gentle", "happy", "quiet", "rapid", "silent", "tiny", "vivid", "wise",
    "noble", "swift", "lucky", "mighty", "polite", "rustic", "sturdy", "tricky",
    "useful", "witty", "young", "keen", "lush", "crisp", "dense", "fair",
    "grand", "jolly", "mild", "prime", "rich", "smooth", "warm", "cold",
    "dark", "soft", "hard", "long", "short", "shy", "slim", "zesty",
]

#: 名词表(wonderwords nounlist 的静态子集, 与功能词/形容词无重叠)
NOUNS: List[str] = [
    "river", "mountain", "forest", "tiger", "falcon", "dolphin", "lantern",
    "harbor", "meadow", "glacier", "ember", "thicket", "sparrow", "otter",
    "canyon", "willow", "pebble", "storm", "breeze", "summit", "valley",
    "reef", "wolf", "heron", "maple", "cedar", "quartz", "fjord", "prairie",
    "raven", "lynx", "badger", "comet", "dune", "fern", "grove", "husk",
    "iris", "juniper", "kite", "lotus", "marsh", "nettle", "oasis", "pine",
    "quail", "reed", "sage", "tide", "umber", "vine", "wren", "yarrow",
    "anchor", "banner", "cinder", "drum", "eagle", "flint", "gorge", "haven",
    "inlet", "jungle", "knoll", "lagoon", "minnow", "nectar", "opal",
    "plateau", "quarry", "ridge", "shoal", "timber", "urn", "wharf", "zenith",
    "amber", "basalt", "cliff", "delta", "elm", "flame", "geyser", "hollow",
    "isle", "jade", "ledge", "moss", "nylon", "orchid", "puma", "quiver",
    "ribbon", "saddle", "tunnel", "velvet", "walnut",
]


def _build_vocab() -> Tuple[Dict[str, int], List[str]]:
    """构建确定性词-编号映射表 / Build deterministic token-to-id vocabulary.

    中文说明:
    - 调用方 / Called by: `build_ruler_niah_vocab`, 各 token 化函数
    - 作用 / Purpose: 将噪声句、模板句、问句、答案前缀、形容词、名词、数字与
      标点全部纳入封闭词表; PAD=0 保留给损失掩码, 不分配给任何词。
    - 返回 / Returns: (token->id 映射, id->token 列表)
    """
    tokens: List[str] = []
    seen = set()

    def _add(word: str) -> None:
        if word not in seen:
            seen.add(word)
            tokens.append(word)

    # 功能词与标点(来自噪声句/针句/问句/答案前缀的拆分)
    for w in (
        NOISE_SENTENCE.lower().replace(".", " . ").split()
        + NEEDLE_TEMPLATE.format(type_v="numbers", key="a-b", value="0").lower().split()
        + "one a of the special magic numbers word uuids for is : what are all"
        " mentioned in provided text some hidden within following make sure to"
        " memorize it i will quiz you about afterwards".split()
        + ["\n", "?", ",", "-"]
    ):
        _add(w)
    # 数字位 token
    for d in "0123456789":
        _add(d)
    # 键素材词
    for w in ADJECTIVES:
        _add(w)
    for w in NOUNS:
        _add(w)

    tok2id = {"<pad>": 0}
    for idx, w in enumerate(tokens, start=1):
        tok2id[w] = idx
    return tok2id, ["<pad>"] + tokens


VOCAB, ID2TOKEN = _build_vocab()


def _tokenize(text: str) -> List[str]:
    """按词/标点/换行/数字位拆分文本并转小写 / Lowercase word+punct+digit tokenization."""
    out: List[str] = []
    for raw in text.split("\n"):
        if raw == "":
            out.append("\n")
            continue
        for piece in raw.split():
            # 纯数字串(含尾标点)拆成单个数字位 + 尾部标点
            core = piece.strip(",.?!")
            tail = piece[len(core):] if core else ""
            if core.isdigit():
                out.extend(list(core))
            else:
                # 连字符保留为独立 token(RULER 键形如 adj-noun)
                parts = core.split("-")
                for k2, p in enumerate(parts):
                    if k2 > 0:
                        out.append("-")
                    if p:
                        out.append(p.lower())
            for ch in tail:
                out.append(ch)
    return out


@dataclass(frozen=True)
class RulerNiahConfig:
    """RULER-NIAH 任务配置规范 / RULER-NIAH task configuration specification.

    Attributes:
        variant (str): 任务变体, 当前支持 "sniah1"(单针数值检索). Default: "sniah1".
        num_haystack (int): 噪声海草句重复条数, 控制上下文长度. Default: 256.
        num_needle_k (int): 插入上下文的针(key-value)数量 K, >= 1. Default: 1.
        num_needle_q (int): 查询数量 Q, 1 <= Q <= K. Default: 1.
        batch_size (int): 每批样本数 B, >= 1. Default: 8.
        device: 输出张量所在设备. Default: "cpu".
        seed (Optional[int]): 随机数种子(确定性复现). Default: None.
    """

    variant: str = "sniah1"
    num_haystack: int = 256
    num_needle_k: int = 1
    num_needle_q: int = 1
    batch_size: int = 8
    device: str = "cpu"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """参数校验 / Validate parameters."""
        if self.variant != "sniah1":
            raise ValueError(f"unsupported variant: {self.variant}")
        if self.num_haystack < 1:
            raise ValueError("num_haystack must be >= 1")
        if self.num_needle_k < 1:
            raise ValueError("num_needle_k must be >= 1")
        if not (1 <= self.num_needle_q <= self.num_needle_k):
            raise ValueError("num_needle_q must satisfy 1 <= Q <= K")
        if self.num_haystack < self.num_needle_k:
            raise ValueError("num_haystack must be >= num_needle_k")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


def _digits_to_ids(value_str: str) -> List[int]:
    """把 7 位数值字符串转为数字位 token id 列表."""
    return [VOCAB[ch] for ch in value_str]


def generate_ruler_niah_batch(
    cfg: RulerNiahConfig,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
    """按 RULER 规范生成一批 NIAH 样本 / Generate one RULER-NIAH batch.

    中文说明:
    - 调用方 / Called by: `scripts.benchmark_ruler_niah`,
      `tests.test_ruler_niah_data_generation`
    - 被调用方 / Callee: 封闭词表 VOCAB, torch.Generator
    - 作用 / Purpose: 严格按 NVIDIA/RULER niah.py 的噪声模式生成流程构造样本:
      重复噪声句填充上下文 → 针句随机插入(排序后逆序 insert) → 问句与答案前缀
      追加于尾部 → 答案为被查询 key 对应的 7 位数字串(逐数字位监督)。
    - 参数 / Parameters: `cfg` 为任务配置。
    - 返回 / Returns:
      X: LongTensor[B, L] 输入 token 序列;
      Y: LongTensor[B, L] 监督目标(PAD=0 除答案数字位外);
      meta: 每样本元数据列表(query key 文本、答案数字串、针句位置等, 用于测试与诊断)。
    - 错误处理 / Error handling: 配置非法由 `RulerNiahConfig.__post_init__` 抛出。
    - 关键词 / Keywords: ruler|niah|needle|havstack|passkey|domain|generator
    """
    gen = torch.Generator(device="cpu")
    if cfg.seed is not None:
        gen.manual_seed(int(cfg.seed))

    xs: List[List[int]] = []
    ys: List[List[int]] = []
    metas: List[Dict[str, object]] = []

    for _ in range(cfg.batch_size):
        # 1. 采样 keys(无放回) 与 7 位 values
        adj_ids = torch.randperm(len(ADJECTIVES), generator=gen)[: cfg.num_needle_k]
        noun_ids = torch.randperm(len(NOUNS), generator=gen)[: cfg.num_needle_k]
        keys: List[Tuple[str, str]] = [
            (ADJECTIVES[int(a)], NOUNS[int(n)]) for a, n in zip(adj_ids, noun_ids)
        ]
        values: List[str] = []
        for _ in range(cfg.num_needle_k):
            lo = 10 ** (VALUE_NUM_DIGITS - 1)
            hi = 10 ** VALUE_NUM_DIGITS - 1
            v = int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
            values.append(str(v))

        needles = [
            _tokenize(
                NEEDLE_TEMPLATE.format(
                    type_v="numbers", key=f"{k[0]}-{k[1]}", value=v
                )
            )
            for k, v in zip(keys, values)
        ]
        # 官方行为: 打乱针句顺序后按降序位置 insert
        perm = torch.randperm(cfg.num_needle_k, generator=gen).tolist()
        shuffled = [needles[p] for p in perm]

        noise_unit = _tokenize(NOISE_SENTENCE)
        sentences: List[List[str]] = [list(noise_unit) for _ in range(cfg.num_haystack)]
        indexes = sorted(
            torch.randperm(cfg.num_haystack, generator=gen)[: cfg.num_needle_k].tolist(),
            reverse=True,
        )
        needle_positions: List[int] = []
        for index, sent in zip(indexes, shuffled):
            sentences.insert(index, sent)
            needle_positions.append(index)

        context: List[str] = []
        for i, sent in enumerate(sentences):
            if i > 0:
                context.append("\n")
            context.extend(sent)

        # 2. 无放回采样 Q 个查询 key
        q_sel = torch.randperm(cfg.num_needle_k, generator=gen)[: cfg.num_needle_q].tolist()
        query_keys = [keys[q] for q in q_sel]
        answer_digits: List[str] = [values[q] for q in q_sel]

        key_str = ", ".join(f"{a}-{n}" for a, n in query_keys)
        question = _tokenize(
            f"What are all the special magic numbers for {key_str} mentioned"
            " in the provided text?"
        )
        answer_prefix = _tokenize(
            f"The special magic numbers for {key_str} mentioned in the"
            " provided text are"
        )

        seq = context + ["\n"] + question + ["\n"] + answer_prefix
        prefix_len = len(seq)
        # 目标 token(答案数字序列 + 句号)同时进入输入 X(teacher-forcing 布局),
        # 标签按 next-token 约定挂在被预测位置的前一位置: Y[prefix_len-1+j] = target_ids[j]
        target_ids: List[int] = []
        for ans in answer_digits:
            target_ids.extend(_digits_to_ids(ans))
        target_ids.append(VOCAB["."])
        seq = seq + [ID2TOKEN[tid] for tid in target_ids]
        x_ids = [VOCAB[t] for t in seq]
        y_ids = [0] * len(x_ids)
        for off, tid in enumerate(target_ids):
            y_ids[prefix_len - 1 + off] = tid

        xs.append(x_ids)
        ys.append(y_ids)
        metas.append(
            {
                "query_keys": key_str,
                "answers": list(answer_digits),
                "needle_positions": needle_positions,
                "context_len": len(context),
                "seq_len": len(x_ids),
            }
        )

    max_len = max(len(x) for x in xs)
    pad_id = 0
    X = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    Y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for b, (x, y) in enumerate(zip(xs, ys)):
        X[b, : len(x)] = torch.tensor(x, dtype=torch.long)
        Y[b, : len(y)] = torch.tensor(y, dtype=torch.long)
    return X.to(cfg.device), Y.to(cfg.device), metas


def evaluate_ruler_niah_exact_match(
    preds_digit_ids: List[List[int]],
    answers: List[List[str]],
) -> float:
    """序列级精确匹配率 / Sequence-level exact-match accuracy.

    中文说明:
    - 调用方 / Called by: `scripts.benchmark_ruler_niah` 评估循环
    - 作用 / Purpose: 将预测数字 id 序列还原为字符串, 与真值答案做全序列比对;
      多查询时要求每个查询的答案均正确才计 1。
    """
    correct = 0
    for pred, ans in zip(preds_digit_ids, answers):
        pred_str = "".join(str(int(i)) for i in pred)
        ok = len(pred_str) == VALUE_NUM_DIGITS * len(ans)
        for a in ans:
            if a not in pred_str:
                ok = False
        correct += int(ok)
    return correct / max(1, len(answers))
