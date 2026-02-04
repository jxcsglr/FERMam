import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # 保证已安装 mamba_ssm
from .ss2d import SS2D

# 🔹 通道 Shuffle 函数
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B, H, W, C = x.size()
    channels_per_group = C // groups
    x = x.view(B, H, W, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(B, H, W, -1)
    return x

# 🔹 MedMamba 原始 SS2D 模块

# 🔹 SS_Conv_SSM：CNN + Mamba 并行结构
class SS_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim,
        d_state=16,
        drop_path=0.0,
        mode="conv_ssm",
    ):
        """
        mode:
            - "conv_ssm": CNN + SSM（完整）
            - "conv_only": 仅 CNN
            - "ssm_only": 仅 SSM
        """
        super().__init__()
        assert mode in ["conv_ssm", "conv_only", "ssm_only"]
        self.mode = mode

        self.hidden_dim = hidden_dim
        self.d_half = hidden_dim // 2

        # CNN 分支
        self.conv_branch = nn.Sequential(
            nn.BatchNorm2d(self.d_half),
            nn.Conv2d(self.d_half, self.d_half, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.d_half),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_half, self.d_half, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.d_half),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_half, self.d_half, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # SS2D 分支
        self.ln = nn.LayerNorm(self.d_half)
        self.ss2d = SS2D(d_model=self.d_half, d_state=d_state)

        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)

    def forward(self, x):
        B, H, W, C = x.shape
        x_left, x_right = x.split([self.d_half, self.d_half], dim=-1)

        # ===============================
        # CNN branch
        # ===============================
        if self.mode in ["conv_ssm", "conv_only"]:
            x_left_ = x_left.permute(0, 3, 1, 2)
            x_left_ = self.conv_branch(x_left_)
            x_left_ = x_left_.permute(0, 2, 3, 1)
        else:
            # ssm_only：CNN 分支直接 identity
            x_left_ = x_left

        # ===============================
        # SSM branch
        # ===============================
        if self.mode in ["conv_ssm", "ssm_only"]:
            x_right_ = self.ln(x_right)
            x_right_ = self.ss2d(x_right_)
        else:
            # conv_only：SSM 分支直接 identity
            x_right_ = x_right

        # ===============================
        # Fusion
        # ===============================
        x_fused = torch.cat([x_left_, x_right_], dim=-1)
        x_fused = channel_shuffle(x_fused, groups=2)

        return x_fused + x
