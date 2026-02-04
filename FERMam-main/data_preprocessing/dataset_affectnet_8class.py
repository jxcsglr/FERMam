import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import os
import random
from torchvision.datasets import DatasetFolder, ImageFolder


class Affectdataset_8class(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, basic_aug=False, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

        # 情感类别到索引的映射
        self.class_to_idx = {
            'anger': 0,
            'contempt': 1,
            'disgust': 2,
            'fear': 3,
            'happy': 4,
            'neutral': 5,
            'sad': 6,
            'surprise': 7
        }

        # 确定数据集类型（训练或验证）
        if train:
            self.data_dir = os.path.join(root, 'train')
        else:
            self.data_dir = os.path.join(root, 'valid')

        # 收集所有图像路径和标签
        self.file_paths = []
        self.targets = []

        # 遍历每个类别文件夹
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            # 获取该类别下的所有图像文件
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.file_paths.append(img_path)
                    self.targets.append(class_idx)

        # 如果指定了数据索引子集，则筛选数据
        if self.dataidxs is not None:
            self.file_paths = [self.file_paths[i] for i in self.dataidxs]
            self.targets = [self.targets[i] for i in self.dataidxs]

        #print(f"加载了 {len(self.file_paths)} 张图像，共 {len(self.class_to_idx)} 个类别")

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.targets

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        target = self.targets[idx]

        # 数据增强
        if self.train and self.basic_aug and random.uniform(0, 1) > 0.5:
            index = random.randint(0, 1)
            image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var ** 0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped


def flip_image(image_array):
    return cv2.flip(image_array, 1)

# import numpy as np
# import glob
# from os.path import *
#
# ######################Train set  7class: 283901   8class :287651
# path = ("/home/cezheng/HPE/emotion/dataset/AffectNet/train_set/annotations")
# files =  sorted(glob.glob(path + '/*_exp.npy'))
# id_file = []
# label = []
# for i in range(len(files)):
#     if np.load(files[i]).astype(int)<7:
#         id_file.append(files[i][66:-8])
#         label.append(np.array(np.load(files[i])).tolist())
# print(len(files))
# print("af", len(label))
#
# with open('train_annotations.txt', 'w+') as f:
#     for i in range (len(id_file)):
#         # f.write("%s\n" % item)
#         f.write("%s.jpg %s\n" % (id_file[i], label[i]))
#     f.close()
#
#
# ############################ Test set    3500(7class) 3999(8class)
# path = ("/home/cezheng/HPE/emotion/dataset/AffectNet/valid_set/annotations")
# files =  sorted(glob.glob(path + '/*_exp.npy'))
# id_file = []
# label = []
# for i in range(len(files)):
#     if np.load(files[i]).astype(int)<8:
#         id_file.append(files[i][66:-8])
#         label.append(np.array(np.load(files[i])).tolist())
# print(len(files))
# print("af", len(label))
#
#
# with open('valid_annotations.txt', 'w+') as f:
#     for i in range (len(id_file)):
#         # f.write("%s\n" % item)
#         f.write("%s.jpg %s\n" % (id_file[i], label[i]))
#     f.close()