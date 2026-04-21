import torch
import torch.nn as nn
import torch.optim as optim
import random
from dsra_model import MultiLayerDSRAModel

def generate_haystack_with_needle(batch_size, seq_len, vocab_size, needle_depth_ratio=0.5):
    """
    Generate a long sequence of noise (haystack), with a specific unique key-value pair (needle)
    hidden at a specific relative depth.
    
    Vocabulary map:
    0: PAD
    1: QUERY_TOKEN (tells model to answer)
    2: NEEDLE_KEY
    3: NEEDLE_VALUE
    4 to vocab_size-1: Haystack Noise
    """
    X = torch.zeros(batch_size, seq_len, dtype=torch.long)
    Y = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    needle_k, needle_v = 2, 3
    
    for b in range(batch_size):
        # Fill with random haystack tokens
        for i in range(seq_len):
            X[b, i] = random.randint(4, vocab_size - 1)
            
        # Determine needle position based on depth ratio
        # Ratio 0.0 means at the very beginning, 1.0 means at the very end
        # We ensure it's not placed in the last few tokens where the query goes
        max_pos = seq_len - 5
        pos = int(max_pos * needle_depth_ratio)
        
        # Plant the needle
        X[b, pos] = needle_k
        X[b, pos+1] = needle_v
        
        # The query at the very end
        X[b, -2] = needle_k
        X[b, -1] = 1 # QUERY_TOKEN
        
        # Target for the last position is the needle value
        Y[b, -1] = needle_v
        
    return X, Y

def run_niah_test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running Needle-In-A-Haystack Test on {device} ---")
    
    # Configuration
    vocab_size = 100
    dim = 64
    seq_len = 8192 # Very long context (8K tokens)
    chunk_size = 256
    num_layers = 2
    K = 64
    kr = 8
    batch_size = 4
    
    model = MultiLayerDSRAModel(
        vocab_size, dim, num_layers, K, kr, chunk_size,
        use_orthogonal_update=True, use_bypass=True
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    epochs = 400
    depths_to_test = [0.1, 0.5, 0.9] # Test needles at 10%, 50%, and 90% of context depth
    
    model.train()
    best_overall_acc = 0.0
    
    for epoch in range(epochs):
        # Dynamically change the depth ratio to make the model robust
        current_depth = random.choice(depths_to_test)
        X, Y = generate_haystack_with_needle(batch_size, seq_len, vocab_size, current_depth)
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        
        B, SeqLen = X.shape
        query_indices = (X == 1).nonzero(as_tuple=True)
        if len(query_indices[0]) != B:
            logits_target = logits[:, -1, :]
            targets = Y[:, -1]
        else:
            logits_target = logits[query_indices[0], query_indices[1], :]
            targets = Y[query_indices[0], query_indices[1]]
            
        loss = criterion(logits_target, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 20 == 0:
            preds = logits_target.argmax(dim=-1)
            correct = (preds == targets).sum().item()
            acc = correct / batch_size
            best_overall_acc = max(best_overall_acc, acc)
            print(f"Epoch {epoch:3d} | Depth: {current_depth:.1f} | Loss: {loss.item():.4f} | Accuracy: {acc*100:5.1f}%")
            
            if best_overall_acc == 1.0 and loss.item() < 0.1:
                print("\nSUCCESS! The model successfully found the needle in 8K context!")
                break
                
    print(f"\nFinal Best Accuracy achieved: {best_overall_acc*100:.1f}%")

if __name__ == '__main__':
    run_niah_test()