import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .ASFR import SS2D  # 依赖于你的 SS2D 模块


class PyramidMambaFusionBlock(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 添加特征融合层
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # 保持SS2D模块
        self.ssm1 = SS2D(d_model=d_model, dropout=dropout)
        self.ssm2 = SS2D(d_model=d_model, dropout=dropout)
        self.ssm3 = SS2D(d_model=d_model, dropout=dropout)

        self.down1 = nn.AvgPool2d(kernel_size=2)
        self.down2 = nn.AvgPool2d(kernel_size=4)
        self.up1 = lambda x, size: nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.up2 = lambda x, size: nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_ir, x_lm):
        # x_ir, x_lm: [B, N, C]
        # 1. 在特征维度拼接
        x = torch.cat([x_ir, x_lm], dim=-1)  # [B, N, 2*C]

        # 2. 特征融合投影
        x = self.fusion_proj(x)  # [B, N, C]

        # 3. 自动补齐到最近平方数
        B, N, C = x.shape
        new_N = int(np.ceil(N ** 0.5)) ** 2
        pad_len = new_N - N
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)  # [B, new_N, C]
        H = W = int(np.sqrt(new_N))  # 正方形尺寸

        # 4. 转换为图像格式 [B, C, H, W]
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        # 5. 多尺度处理
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x1)

        # 6. 使用SS2D处理各尺度特征
        x1_out = self.ssm1(rearrange(x1, 'b c h w -> b h w c'))
        x1_out = rearrange(x1_out, 'b h w c -> b c h w')

        x2_out = self.ssm2(rearrange(x2, 'b c h w -> b h w c'))
        x2_out = rearrange(x2_out, 'b h w c -> b c h w')
        x2_out = self.up1(x2_out, size=x1.shape[2:])

        x3_out = self.ssm3(rearrange(x3, 'b c h w -> b h w c'))
        x3_out = rearrange(x3_out, 'b h w c -> b c h w')
        x3_out = self.up2(x3_out, size=x1.shape[2:])

        # 7. 融合多尺度特征
        x_fused = x1_out + x2_out + x3_out  # [B, C, H, W]

        # 8. 转回序列
        x_fused = rearrange(x_fused, 'b c h w -> b (h w) c')  # [B, N_pad, C]
        x_fused = x_fused[:, :N, :]  # 去除 padding
        x_fused = self.norm(x_fused)
        return x_fused