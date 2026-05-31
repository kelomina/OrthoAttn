"""与预训练混合架构语言模型对话的交互式工具。

Interactive chat tool for the pre-trained hybrid language model (ST + MHDSRA2).

功能 / Features:
1. 加载预训练模型和分词器 / Load pre-trained model and tokenizer
2. 自回归文本生成（支持 greedy / temperature / top-k / top-p 采样）
3. 交互式对话循环 / Interactive chat loop
4. 流式逐 token 输出 / Streaming token-by-token output

调用方 / Called by:
    python scripts/chat.py
    python main.py chat

被调用方 / Calls:
    HybridLanguageModel, FastBPETokenizer

关键词 / Keywords:
    chat, generate, inference, interactive, 对话, 生成, 推理
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pretrain_hybrid_lm import FastBPETokenizer, HybridLanguageModel

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


def load_model_and_tokenizer(
    model_path: str = "models/hybrid_lm/best_model.pt",
    tokenizer_path: str = "models/bpe_tokenizer.json",
):
    """加载预训练模型和分词器。

    Load pre-trained model and tokenizer.

    调用方 / Called by: chat_loop, main
    被调用方 / Calls: FastBPETokenizer.load, HybridLanguageModel, torch.load
    参数 / Parameters:
        model_path: 模型检查点路径 / model checkpoint path
        tokenizer_path: 分词器路径 / tokenizer path
    返回值 / Returns:
        model: 加载权重后的 HybridLanguageModel 实例
        tokenizer: FastBPETokenizer 实例
        config: 检查点中保存的模型配置字典
    """
    tokenizer_file = PROJECT_ROOT / tokenizer_path
    if not tokenizer_file.exists():
        print(f"错误：分词器文件不存在: {tokenizer_file}")
        print("请先运行预训练脚本训练分词器。")
        sys.exit(1)
    tokenizer = FastBPETokenizer.load(str(tokenizer_file))
    print(f"分词器已加载: 词汇表大小={len(tokenizer.encoder):,}")

    model_file = PROJECT_ROOT / model_path
    if not model_file.exists():
        print(f"错误：模型文件不存在: {model_file}")
        print("请先运行预训练脚本训练模型。")
        sys.exit(1)

    checkpoint = torch.load(str(model_file), map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})
    vocab_size = checkpoint.get("vocab_size", len(tokenizer.encoder))

    print(f"检查点信息:")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Step: {checkpoint.get('step', '?'):,}")
    print(f"  Best PPL: {checkpoint.get('best_ppl', float('inf')):.1f}")
    print(f"  配置: {config}")

    model = HybridLanguageModel(
        vocab_size=vocab_size,
        dim=config.get("dim", 256),
        n_layers=config.get("n_layers", 4),
        n_heads=config.get("n_heads", 4),
        slots=config.get("slots", 128),
        chunk_size=config.get("seq_len", 512),
    )

    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        print(f"  ⚠ 缺少键: {len(missing)} 个")
    if unexpected:
        print(f"  ⚠ 多余键: {len(unexpected)} 个")

    model.eval()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型已加载: 参数量={num_params:.2f}M, 设备={DEVICE}")
    return model, tokenizer, config


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    stream: bool = True,
):
    """自回归生成文本。

    Autoregressive text generation with sampling strategies.

    调用方 / Called by: chat_loop
    被调用方 / Calls: model.forward, tokenizer.encode, tokenizer.decode
    参数 / Parameters:
        model: HybridLanguageModel 实例
        tokenizer: FastBPETokenizer 实例
        prompt: 输入提示文本 / input prompt text
        max_new_tokens: 最大生成 token 数 / max number of tokens to generate
        temperature: 采样温度，越高越随机 / sampling temperature
        top_k: top-k 采样的 k 值，0 表示不使用 / top-k value, 0 to disable
        top_p: nucleus 采样的 p 值，1.0 表示不使用 / nucleus sampling p, 1.0 to disable
        repetition_penalty: 重复惩罚系数，>1 抑制重复 / repetition penalty, >1 to suppress
        stream: 是否流式逐 token 输出 / whether to stream output token by token
    返回值 / Returns:
        generated_text: 生成的完整文本（不含 prompt）/ full generated text (excluding prompt)
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    states = None
    generated_ids = []

    bos_id = tokenizer.special_tokens.get(BOS_TOKEN, 1)
    eos_id = tokenizer.special_tokens.get(EOS_TOKEN, 2)

    logits, states = model(input_tensor, states=states)
    next_logits = logits[:, -1, :]

    for step in range(max_new_tokens):
        logits_now = next_logits.clone()

        if repetition_penalty != 1.0:
            for token_id in set(generated_ids):
                if token_id < logits_now.shape[-1]:
                    logits_now[:, token_id] /= repetition_penalty

        logits_now = logits_now / max(temperature, 1e-8)

        if top_k > 0:
            top_k_val = min(top_k, logits_now.shape[-1])
            topk_vals, _ = torch.topk(logits_now, top_k_val, dim=-1)
            threshold = topk_vals[:, -1:]
            logits_now = logits_now.masked_fill(logits_now < threshold, float('-inf'))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_now, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            logits_now = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits_now, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == eos_id:
            break

        generated_ids.append(next_token)

        if stream:
            token_text = tokenizer.decode([next_token])
            print(token_text, end="", flush=True)

        next_input = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        logits, states = model(next_input, states=states)
        next_logits = logits[:, -1, :]

    if stream:
        print()

    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def chat_loop(
    model,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
):
    """交互式对话循环。

    Interactive chat loop with the language model.

    调用方 / Called by: main
    被调用方 / Calls: generate
    参数 / Parameters:
        model: HybridLanguageModel 实例
        tokenizer: FastBPETokenizer 实例
        max_new_tokens: 每次回复最大生成 token 数
        temperature: 采样温度
        top_k: top-k 采样参数
        top_p: nucleus 采样参数
        repetition_penalty: 重复惩罚系数
    """
    print("\n" + "=" * 60)
    print("  DSRA 混合架构语言模型 - 交互式对话")
    print("  Hybrid LM (ST + MHDSRA2) Interactive Chat")
    print("=" * 60)
    print(f"  设备: {DEVICE}")
    print(f"  最大生成长度: {max_new_tokens} tokens")
    print(f"  温度: {temperature} | Top-K: {top_k} | Top-P: {top_p}")
    print(f"  重复惩罚: {repetition_penalty}")
    print("-" * 60)
    print("  输入 'quit' 或 'exit' 退出")
    print("  输入 'clear' 清空对话历史")
    print("  输入 'help' 查看帮助")
    print("  输入 'set <参数> <值>' 修改生成参数")
    print("=" * 60 + "\n")

    conversation_history = ""

    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("再见！")
            break

        if user_input.lower() == "clear":
            conversation_history = ""
            print("对话历史已清空。")
            continue

        if user_input.lower() == "help":
            print("\n可用命令:")
            print("  quit / exit  - 退出对话")
            print("  clear        - 清空对话历史")
            print("  help         - 显示此帮助")
            print("  set temp <值>     - 设置温度 (当前: {:.2f})".format(temperature))
            print("  set topk <值>     - 设置 top-k (当前: {})".format(top_k))
            print("  set topp <值>     - 设置 top-p (当前: {:.2f})".format(top_p))
            print("  set maxlen <值>   - 设置最大生成长度 (当前: {})".format(max_new_tokens))
            print("  set reppen <值>   - 设置重复惩罚 (当前: {:.2f})".format(repetition_penalty))
            print()
            continue

        if user_input.lower().startswith("set "):
            parts = user_input.split()
            if len(parts) >= 3:
                param, value = parts[1], parts[2]
                try:
                    if param == "temp":
                        temperature = max(0.01, float(value))
                        print(f"温度已设置为: {temperature:.2f}")
                    elif param == "topk":
                        top_k = max(0, int(value))
                        print(f"Top-K 已设置为: {top_k}")
                    elif param == "topp":
                        top_p = max(0.0, min(1.0, float(value)))
                        print(f"Top-P 已设置为: {top_p:.2f}")
                    elif param == "maxlen":
                        max_new_tokens = max(1, int(value))
                        print(f"最大生成长度已设置为: {max_new_tokens}")
                    elif param == "reppen":
                        repetition_penalty = max(1.0, float(value))
                        print(f"重复惩罚已设置为: {repetition_penalty:.2f}")
                    else:
                        print(f"未知参数: {param} (可用: temp, topk, topp, maxlen, reppen)")
                except ValueError:
                    print(f"无效的值: {value}")
            else:
                print("用法: set <参数> <值> (如: set temp 0.7)")
            continue

        if conversation_history:
            prompt = conversation_history + " " + user_input
        else:
            prompt = user_input

        print("模型> ", end="", flush=True)
        start_time = time.time()

        generated = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=True,
        )

        elapsed = time.time() - start_time
        num_tokens = len(tokenizer.encode(generated, add_special_tokens=False))
        speed = num_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{num_tokens} tokens, {elapsed:.1f}s, {speed:.1f} tok/s]")

        conversation_history = prompt + " " + generated


def main():
    """主入口函数。

    Entry point for the chat tool.

    调用方 / Called by: python scripts/chat.py
    被调用方 / Calls: load_model_and_tokenizer, chat_loop
    """
    parser = argparse.ArgumentParser(description="DSRA 混合架构语言模型 - 交互式对话工具")
    parser.add_argument("--model", type=str, default="models/hybrid_lm/best_model.pt",
                        help="模型检查点路径 (默认: models/hybrid_lm/best_model.pt)")
    parser.add_argument("--tokenizer", type=str, default="models/bpe_tokenizer.json",
                        help="分词器路径 (默认: models/bpe_tokenizer.json)")
    parser.add_argument("--max-len", type=int, default=128,
                        help="每次回复最大生成 token 数 (默认: 128)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度 (默认: 0.8)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K 采样参数 (默认: 50)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus 采样 Top-P 参数 (默认: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="重复惩罚系数 (默认: 1.2)")
    args = parser.parse_args()

    model, tokenizer, config = load_model_and_tokenizer(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
    )

    chat_loop(
        model,
        tokenizer,
        max_new_tokens=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
