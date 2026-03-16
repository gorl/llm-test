import torch
import torch.nn as nn
import torch.nn.functional as F



# def build_lm_head(d_model: int, vocab_size: int, tok_w: nn.Parameter | None = None, bias: bool = False):

#     if tok_w is None:
#         weight = nn.Parameter(torch.empty(vocab_size, d_model))
#         nn.init.normal_(weight, mean=0.0, std=0.02)
#     else:
#         weight = tok_w

#     bias = nn.Parameter(torch.zeros(vocab_size)) if bias else None
#     return lambda x: F.linear(x,weight, bias)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return F.linear(x, self.weight, self.bias)
    
# from __future__ import annotations

# import torch.nn as nn
# import torch

def build_lm_head(d_model: int, vocab_size: int, tok_w: nn.Parameter | None = None) -> nn.Module:
    lm_head = nn.Linear(d_model, vocab_size, bias=False if tok_w is not None else True)
    if tok_w is not None:
        lm_head.weight = tok_w
    return lm_head