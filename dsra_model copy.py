import torch
import torch.nn as nn
from dsra_layer import DSRA_Chunk_Layer

class MultiLayerDSRAModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers=2, K=128, kr=16, chunk_size=256, use_orthogonal_update=True, use_bypass=True, pe_mode='none'):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.pe_mode = pe_mode
        
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Multi-layer stack
        self.layers = nn.ModuleList([
            DSRA_Chunk_Layer(dim, K=K, kr=kr, use_orthogonal_update=use_orthogonal_update, use_bypass=use_bypass, pe_mode=pe_mode)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        """
        x: [B, SeqLen]
        Returns: logits [B, SeqLen, vocab_size]
        """
        B, SeqLen = x.shape
        hidden = self.embedding(x) # [B, SeqLen, dim]
        
        # State lists for each layer
        S_prev_list = [None] * self.num_layers
        S_time_prev_list = [None] * self.num_layers
        bypass_kv_list = [None] * self.num_layers
        
        out_list = []
        
        # Process block by block
        chunk_idx = 0
        for i in range(0, SeqLen, self.chunk_size):
            chunk = hidden[:, i:i+self.chunk_size, :]
            
            # Pass chunk through all layers
            for l_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
                residual = chunk
                # Layer norm before DSRA
                chunk_normed = norm(chunk)
                
                # DSRA forward
                out_chunk, S_next, next_bypass_kv, S_time_next = layer(
                    chunk_normed, 
                    S_prev=S_prev_list[l_idx], 
                    bypass_kv=bypass_kv_list[l_idx],
                    S_time_prev=S_time_prev_list[l_idx],
                    chunk_idx=chunk_idx
                )
                
                # Update layer states
                S_prev_list[l_idx] = S_next
                bypass_kv_list[l_idx] = next_bypass_kv
                S_time_prev_list[l_idx] = S_time_next
                
                # Residual connection
                chunk = residual + out_chunk
                
            out_list.append(chunk)
            chunk_idx += 1
            
        # Reconstruct full sequence
        out = torch.cat(out_list, dim=1) # [B, SeqLen, dim]
        out = self.final_norm(out)
        logits = self.out_proj(out)
        
        return logits
