import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
from .pyramid_mamba_fusion_block import PyramidMambaFusionBlock

from .IR_MambaBackbone import SS_Conv_SSM


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model

class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.linear1(x)
        w = self.relu(w)
        w = self.linear2(w)
        w = self.sigmoid(w)
        return x * w

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, target_dim)

    def forward(self, x):
        return self.linear(x)

class pyramid_trans_expr(nn.Module):
    def __init__(self, img_size=224, num_classes=7, type="large"):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        depth = {"small": 4, "base": 6, "large": 8}.get(type, 8)

        # 人脸关键点分支
        self.face_landback = MobileFaceNet([112, 112], embedding_size=512, output_name="GDC")
        try:
            ckpt = torch.load('./models/pretrain/mobilefacenet_model_best.pth.tar',
                              map_location=lambda s, l: s)
            state_dict = ckpt['state_dict']
            keys_to_delete = [k for k in state_dict if 'output_layer' in k or 'heatmap_weight' in k]
            for k in keys_to_delete:
                print(f"Delete key: {k}")
                del state_dict[k]
            self.face_landback.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading MobileFaceNet weights: {e}")

        for p in self.face_landback.parameters():
            p.requires_grad = False

        # IR50 特征提取 + Linear 降维
        self.ir_back = Backbone(50, drop_ratio=0.0, mode='ir')
        try:
            ir_ckpt = torch.load('./models/pretrain/ir50.pth', map_location=lambda s, l: s)
            self.ir_back = load_pretrained_weights(self.ir_back, ir_ckpt)
        except Exception as e:
            print(f"Error loading IR50 weights: {e}")

        self.ir_layer = nn.Linear(1024, 512)
        self.ir_ssm = SS_Conv_SSM(hidden_dim=512)


        # 融合模块
        self.fusion = PyramidMambaFusionBlock(d_model=512)
        self.se_block = SE_block(input_dim=512)
        self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)

    def forward(self, x):
        B = x.size(0)
        x_face = F.interpolate(x, size=112, mode='bilinear', align_corners=False)

        # 人脸关键点特征 [B, 512, 7, 7] → [B, 49, 512]
        _, feat_face = self.face_landback(x_face)
        feat_face = feat_face.view(B, 512, -1).permute(0, 2, 1)

        # IR 特征提取 + 降维 + reshape 为 [B, 7, 7, 512]
        feat_ir = self.ir_back(x)
        feat_ir = self.ir_layer(feat_ir)
        feat_ir = feat_ir.view(B, 7, 7, 512)
        feat_ir = self.ir_ssm(feat_ir)

        if feat_ir.dim()==4:
            feat_ir = feat_ir.view(B,49, 512)

        # 融合
        fused_seq = self.fusion(feat_ir, feat_face)  # [B, 49, 512]

        # 分类（取 cls_token = 第一个位置）
        cls_token = fused_seq[:, 0, :]
        cls_token = self.se_block(cls_token)
        out = self.head(cls_token)
        return out, cls_token