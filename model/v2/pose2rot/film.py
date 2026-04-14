import torch.nn as nn


# =========================================================
# helper: conditioning
# =========================================================
class FiLMCondition(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, x, cond):
        """
        x:    [..., D]
        cond: same shape as x
        """
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + 0.1 * scale) + shift
