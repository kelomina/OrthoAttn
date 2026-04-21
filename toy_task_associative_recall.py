import random

import torch
import torch.nn as nn
import torch.optim as optim

from dsra_layer import DSRA_Chunk_Layer

class DSRAModel(nn.Module):
    def __init__(self, vocab_size, dim, K=128, kr=8, chunk_size=256, pe_mode='none'):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, dim)
        self.dsra = DSRA_Chunk_Layer(dim, K=K, kr=kr, pe_mode=pe_mode)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def build_causal_context(self, emb):
        shifts = [emb]
        for offset in range(1, 4):
            shifted = torch.zeros_like(emb)
            shifted[:, offset:, :] = emb[:, :-offset, :]
            shifts.append(shifted)
        return sum(shifts)

    def forward(self, x):
        B, SeqLen = x.shape
        emb = self.build_causal_context(self.embedding(x))

        out_list = []
        S_prev = None
        bypass_kv = None
        S_time_prev = None

        chunk_idx = 0
        for i in range(0, SeqLen, self.chunk_size):
            chunk = emb[:, i:i+self.chunk_size, :]
            out_chunk, S_next, next_bypass_kv, S_time_next = self.dsra(
                chunk, S_prev=S_prev, bypass_kv=bypass_kv, S_time_prev=S_time_prev, chunk_idx=chunk_idx
            )

            out_list.append(out_chunk)
            S_prev = S_next
            bypass_kv = next_bypass_kv
            S_time_prev = S_time_next
            chunk_idx += 1

        out = torch.cat(out_list, dim=1)
        out = self.norm(out)
        logits = self.out_proj(out)

        return logits

def generate_associative_recall_data(batch_size, seq_len, vocab_size, num_pairs=10):
    X = torch.zeros(batch_size, seq_len, dtype=torch.long)
    Y = torch.zeros(batch_size, seq_len, dtype=torch.long)
    query_token = 1
    key_token = 2
    value_token = 3
    required_tokens = (2 * num_pairs) + 1

    if vocab_size - 4 < required_tokens:
        raise ValueError("Vocabulary too small for disjoint keys, values, and noise.")

    for b in range(batch_size):
        available_tokens = random.sample(range(4, vocab_size), required_tokens)
        keys = available_tokens[:num_pairs]
        values = available_tokens[num_pairs:2 * num_pairs]
        noise_tokens = available_tokens[2 * num_pairs:]
        pairs = list(zip(keys, values))

        for i in range(seq_len):
            X[b, i] = random.choice(noise_tokens)

        tail_start = seq_len - 2 - (num_pairs * 4)
        available_positions = list(range(tail_start, seq_len - 2, 4))
        if len(available_positions) < num_pairs:
            raise ValueError("Sequence too short to fit the required number of pairs.")

        positions = sorted(random.sample(available_positions, num_pairs))
        for (k, v), pos in zip(pairs, positions):
            X[b, pos] = key_token
            X[b, pos + 1] = k
            X[b, pos + 2] = value_token
            X[b, pos + 3] = v

        query_key, target_value = random.choice(pairs)
        query_pos = seq_len - 2
        X[b, query_pos] = query_token
        X[b, query_pos + 1] = query_key
        Y[b, query_pos + 1] = target_value

    return X, Y

def train_step(model, X, Y, optimizer, criterion):
    optimizer.zero_grad()
    logits = model(X)
    B, SeqLen = X.shape
    target_indices = (Y != 0).nonzero(as_tuple=True)

    if len(target_indices[0]) != B:
        logits_target = logits[:, -1, :]
        targets = Y[:, -1]
    else:
        logits_target = logits[target_indices[0], target_indices[1], :]
        targets = Y[target_indices[0], target_indices[1]]

    loss = criterion(logits_target, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    preds = logits_target.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    acc = correct / B

    return loss.item(), acc

def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_size = 100
    dim = 64
    seq_len = 1024
    chunk_size = 256
    batch_size = 16
    epochs = 1500

    model = DSRAModel(vocab_size=vocab_size, dim=dim, K=64, kr=8, chunk_size=chunk_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(epochs):
        X, Y = generate_associative_recall_data(batch_size, seq_len, vocab_size, num_pairs=20)
        X, Y = X.to(device), Y.to(device)

        loss_val, acc = train_step(model, X, Y, optimizer, criterion)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss_val:.4f} | Accuracy: {acc*100:.1f}%")

            if acc == 1.0 and loss_val < 0.05:
                print("Task solved successfully!")
                break

if __name__ == "__main__":
    train()
