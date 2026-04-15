import torch.nn as nn
import torch
from .attention_processors import SimpleAttnProcessor


# base attention
class SimpleAttention(nn.Module):
    def __init__(self, dim, heads, cross_attention_dim=None, processor=None):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.processor = processor or SimpleAttnProcessor()

    def forward(self, x, encoder_hidden_states=None, **kwargs):
        return self.processor(
            self, x, encoder_hidden_states=encoder_hidden_states, **kwargs
        )
