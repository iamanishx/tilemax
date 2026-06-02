import torch
import torch.nn as nn
import math

class GroupedQuerySlidingWindowAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        
        # Ensure that number of heads is divisible by number of kv heads
        assert num_heads % num_kv_heads == 0
        self.group_size = num_heads // num_kv_heads
        
        self.head_dim = embed_dim // num_heads
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * num_kv_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states):
        B, S, E = hidden_states.shape # Batch, Sequence, Embedding
        
        # 1. Project and reshape heads
        # Q shape: (B, num_heads, S, head_dim)
        # K, V shape: (B, num_kv_heads, S, head_dim)
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Grouped Query mechanism: Repeat K and V to match the number of Q heads
        # Resulting K/V shape: (B, num_heads, S, head_dim)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # 3. Create causal and sliding window mask
        # We want tokens to only attend to themselves and the last 'window_size' tokens
        mask = torch.ones(S, S, device=hidden_states.device, dtype=torch.bool)
        mask = torch.tril(mask) # Causal mask (prevents looking into future)
        
        # Apply sliding window: mask out tokens outside the window_size
        window_mask = torch.ones(S, S, device=hidden_states.device, dtype=torch.bool)
        window_mask = window_mask.triu(diagonal=-self.window_size)
        mask = mask & window_mask
        
        # Convert to additive mask (-inf for padded/masked positions)
        attn_mask = torch.zeros((S, S), device=hidden_states.device)
        attn_mask.masked_fill_(~mask, float('-inf'))
        
        # 4. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask
        scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # 5. Concatenate heads and project output
        output = output.transpose(1, 2).contiguous().view(B, S, E)
        return self.out_proj(output)

# --- Example Usage ---
# Embed dim 512, 8 query heads, 2 KV heads, sliding window of 4 tokens
gqa_swa = GroupedQuerySlidingWindowAttention(embed_dim=512, num_heads=8, num_kv_heads=2, window_size=4)
sample_input = torch.randn(1, 10, 512) # batch=1, seq_len=10, dim=512
output = gqa_swa(sample_input)
print("Output shape:", output.shape)
