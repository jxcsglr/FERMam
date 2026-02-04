import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SS2D(nn.Module):
    """
    Mamba-inspired 2D Selective Scan Module
    专为表情识别任务设计，处理融合后的特征

    输入:
        x: 融合特征 [B, H, W, C]
    输出: 处理后的特征 [B, H, W, C_out]
    """

    def __init__(self, d_model=512, d_state=16, expand=2, dt_rank=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank

        # 输入投影 (带门控机制)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 深度卷积 (提取局部特征)
        self.depthwise_conv = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=3, stride=1,
            padding=1, groups=self.d_inner,
            bias=False
        )

        # 动态参数生成
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)

        # 时间步参数
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # 状态空间参数
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt -> N 投影（新增部分）
        self.dt_to_N_proj = nn.Linear(self.d_inner, d_state)

        # 输出层
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        nn.init.constant_(self.dt_proj.bias, 0.1)
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        nn.init.normal_(self.A_log, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def selective_scan(self, x, dt, A, B_in, C_in, D):
        """
        简化版选择性扫描 (水平方向)
        输入:
            x: [B, L, D] 输入序列
            dt: [B, L, D] 时间步参数
            A: [D, N] 状态矩阵
            B_in: [B, L, N] 输入投影
            C_in: [B, L, N] 输出投影
            D: [D] 跳跃连接
        输出: [B, L, D]
        """
        B, L, D = x.shape
        N = A.shape[1]

        # 离散化参数
        dt = F.softplus(dt)
        A = -torch.exp(A)  # [D, N]
        A = A.unsqueeze(0)  # -> [1, D, N]

        # 状态初始化
        h = torch.zeros(B, D, N, device=x.device)

        # 输出序列
        y = torch.zeros_like(x)

        # 将 dt 映射到 [B, L, N]，以便与 B_in 相乘
        dt_proj = self.dt_to_N_proj(dt)  # [B, L, N]

        for i in range(L):
            dA = torch.exp(dt[:, i].unsqueeze(-1) * A)  # [B, D, N]
            dB = dt_proj[:, i] * B_in[:, i]              # [B, N]
            h = dA * h + dB.unsqueeze(1) * x[:, i].unsqueeze(-1)  # [B, D, N]
            y[:, i] = torch.einsum('bdn,bn->bd', h, C_in[:, i]) + D * x[:, i]

        return y

    def forward(self, x):
        """
        输入:
            x: 融合特征 [B, H, W, C]
        输出: 处理后的特征 [B, H, W, C_out]
        """
        # 1. 输入投影 (带门控)
        xz = self.in_proj(x)  # [B, H, W, d_inner*2]
        x, z = xz.chunk(2, dim=-1)  # [B, H, W, d_inner] each

        # 2. 深度卷积处理
        x_conv = rearrange(x, 'b h w c -> b c h w')
        x_conv = self.depthwise_conv(x_conv)  # [B, d_inner, H, W]
        x_conv = rearrange(x_conv, 'b c h w -> b h w c')

        # 3. 动态参数生成
        x_proj = self.x_proj(x_conv)  # [B, H, W, dt_rank + d_state*2]
        dt_proj_part, B_proj, C_proj = torch.split(
            x_proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        # 4. 时间步参数处理
        dt = self.dt_proj(dt_proj_part)  # [B, H, W, d_inner]

        # 5. 准备扫描
        bs, H, W, D = x_conv.shape
        x_seq = rearrange(x_conv, 'b h w d -> b (h w) d')        # [B, L, D]
        dt_seq = rearrange(dt, 'b h w d -> b (h w) d')           # [B, L, D]
        B_seq = rearrange(B_proj, 'b h w n -> b (h w) n')        # [B, L, N]
        C_seq = rearrange(C_proj, 'b h w n -> b (h w) n')        # [B, L, N]

        # 6. 状态空间扫描
        A = self.A_log.unsqueeze(0).expand(D, -1)                # [D, N]
        y = self.selective_scan(x_seq, dt_seq, A, B_seq, C_seq, self.D)  # [B, L, D]

        # 7. 恢复空间结构
        y = rearrange(y, 'b (h w) d -> b h w d', h=H, w=W)

        # 8. 门控与输出
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y
