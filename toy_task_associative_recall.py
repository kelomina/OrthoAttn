import torch
import torch.nn as nn
import torch.optim as optim
import random
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

    def forward(self, x):
        """
        x: [B, SeqLen]
        """
        B, SeqLen = x.shape
        emb = self.embedding(x) # [B, SeqLen, dim]
        
        # We will process this in chunks to simulate the block-streaming
        out_list = []
        S_prev = None
        bypass_kv = None
        S_time_prev = None
        
        chunk_idx = 0
        for i in range(0, SeqLen, self.chunk_size):
            chunk = emb[:, i:i+self.chunk_size, :]
            
            # Forward through DSRA
            out_chunk, S_next, next_bypass_kv, S_time_next = self.dsra(
                chunk, S_prev=S_prev, bypass_kv=bypass_kv, S_time_prev=S_time_prev, chunk_idx=chunk_idx
            )
            
            out_list.append(out_chunk)
            
            # Update states for next chunk
            S_prev = S_next
            bypass_kv = next_bypass_kv
            S_time_prev = S_time_next
            chunk_idx += 1
            
        out = torch.cat(out_list, dim=1) # [B, SeqLen, dim]
        out = self.norm(out)
        logits = self.out_proj(out) # [B, SeqLen, vocab_size]
        
        return logits

def generate_associative_recall_data(batch_size, seq_len, vocab_size, num_pairs=10):
    """
    Generate sequences for associative recall.
    Vocabulary:
    0: PAD
    1: QUERY_TOKEN (tells model "what is the value for the preceding key?")
    2 to vocab_size-1: Keys and Values
    """
    X = torch.zeros(batch_size, seq_len, dtype=torch.long)
    Y = torch.zeros(batch_size, seq_len, dtype=torch.long) 
    
    for b in range(batch_size):
        # Generate random key-value pairs
        keys = random.sample(range(2, vocab_size), num_pairs)
        values = random.sample(range(2, vocab_size), num_pairs)
        pairs = list(zip(keys, values))
        
        # Fill sequence with noise
        for i in range(seq_len):
            X[b, i] = random.randint(2, vocab_size - 1)
            
        # Insert pairs randomly without overlap
        # Each pair needs 2 tokens, plus the query needs 2 tokens at the end.
        # We need `num_pairs` slots of size 2.
        available_positions = list(range(0, seq_len - 5, 2))
        if len(available_positions) < num_pairs:
            raise ValueError("Sequence too short to fit the required number of pairs.")
            
        positions = sorted(random.sample(available_positions, num_pairs))
        for (k, v), pos in zip(pairs, positions):
            X[b, pos] = k
            X[b, pos+1] = v
            
        # Pick one key to query at a random valid position near the end, or just at the end
        query_key, target_value = random.choice(pairs)
        
        # Format the end of the sequence: [..., query_key, QUERY_TOKEN]
        query_pos = seq_len - 2
        X[b, query_pos] = query_key
        X[b, query_pos + 1] = 1 # QUERY_TOKEN
        
        # Target for the QUERY_TOKEN position is the value
        Y[b, query_pos + 1] = target_value
        
    return X, Y

def train_step(model, X, Y, optimizer, criterion):
    """
    Unit of Work for a single training step.
    """
    optimizer.zero_grad()
    logits = model(X) # [B, SeqLen, vocab_size]
    
    # Dynamically find the QUERY_TOKEN (1) position for each batch
    B, SeqLen = X.shape
    query_indices = (X == 1).nonzero(as_tuple=True)
    # query_indices[0] are batch indices, query_indices[1] are sequence indices
    
    if len(query_indices[0]) != B:
        # Fallback to last token if query token count doesn't match batch size
        logits_target = logits[:, -1, :]
        targets = Y[:, -1]
    else:
        # Extract the logits and targets exactly at the QUERY_TOKEN positions
        logits_target = logits[query_indices[0], query_indices[1], :]
        targets = Y[query_indices[0], query_indices[1]]
        
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
            
            # Stop if we solved it
            if acc == 1.0 and loss_val < 0.05:
                print("Task solved successfully!")
                break
                
if __name__ == "__main__":
    train()
