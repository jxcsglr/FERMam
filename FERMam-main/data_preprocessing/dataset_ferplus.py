# data_preprocessing/dataset_ferplus.py
import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class FERPlusDataset(Dataset):
    def __init__(self, csv_file, root_dir, split='Training', transform=None, use_soft_labels=True):
        """
        Args:
            csv_file: CSV文件路径
            root_dir: 图片目录
            split: 'Training', 'PublicTest', 'PrivateTest'
            transform: torchvision transforms
            use_soft_labels: True 表示返回 soft label（投票分布），False 返回 hard label（最大投票类别）
        """
        # 自动检测分隔符
        with open(csv_file, 'r') as f:
            first_line = f.readline()
            sep = '\t' if '\t' in first_line else ','

        self.data = pd.read_csv(csv_file, sep=sep)

        if 'Usage' not in self.data.columns or 'Image name' not in self.data.columns:
            raise ValueError(f"CSV文件缺少 'Usage' 或 'Image name' 列，请检查列名: {self.data.columns.tolist()}")

        self.root_dir = root_dir
        self.transform = transform
        self.use_soft_labels = use_soft_labels

        # 只保留指定 split
        self.data = self.data[self.data['Usage'] == split]

        # 去掉空的文件名，并转为字符串
        self.data = self.data.dropna(subset=['Image name'])
        self.data['Image name'] = self.data['Image name'].astype(str)

        self.samples = []
        for _, row in self.data.iterrows():
            img_name = row['Image name']
            img_path = os.path.join(root_dir, img_name)

            # 检查图片是否存在
            if not os.path.exists(img_path):
                print(f"[警告] 图片不存在，已跳过: {img_path}")
                continue

            # neutral 到 contempt 共 8 类
            label_counts = row[2:10].astype(float).values  # 取投票列
            total_votes = np.sum(label_counts)

            if total_votes == 0:
                print(f"[警告] {img_name} 投票数为 0，已跳过")
                continue

            if self.use_soft_labels:
                # 转换为概率分布
                label = torch.tensor(label_counts / total_votes, dtype=torch.float32)
            else:
                # 取最大投票类别作为 hard label
                label = torch.tensor(np.argmax(label_counts), dtype=torch.long)

            self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"{split} 数据集中没有可用的图片，请检查路径和 CSV 是否匹配。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label
