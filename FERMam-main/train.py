import warnings
warnings.filterwarnings("ignore")

import os
import time
import argparse
import thop
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms

from sklearn.metrics import f1_score

from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from data_preprocessing.sam import SAM
from utils import LabelSmoothingCrossEntropy


from models.emotion_hyp import pyramid_trans_expr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset: rafdb | affectnet | affectnet8class')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=300, help='Train batch size')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Validation batch size')
    parser.add_argument('--modeltype', type=str, default='large', help='Model depth: small | base | large')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer: adam | adamw | sgd')
    parser.add_argument('--lr', type=float, default=4e-5, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--epochs', type=int, default=260, help='Total epochs')
    parser.add_argument('--gpu', type=str, default='0,1', help='GPU ids, comma separated')
    return parser.parse_args()


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state = collections.OrderedDict()
    matched = []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k_ = k[7:]
        else:
            k_ = k
        if k_ in model_dict and model_dict[k_].size() == v.size():
            new_state[k_] = v
            matched.append(k_)
    model_dict.update(new_state)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(matched)} layers from checkpoint")
    return model


def run_training():
    args = parse_args()
    torch.manual_seed(123)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Using GPUs:", os.environ['CUDA_VISIBLE_DEVICES'])

    # 数据增强
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(scale=(0.02,0.1)),
    ])
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 选择数据集和模型
    if args.dataset == 'rafdb':
        num_classes = 7
        train_ds = RafDataSet('/home/cdu-cs/jx/J/data/RAF-DB/basic/', train=True, transform=train_transforms, basic_aug=True)
        val_ds   = RafDataSet('/home/cdu-cs/jx/J/data/RAF-DB/basic/', train=False, transform=val_transforms)
    elif args.dataset == 'affectnet':
        num_classes = 7
        train_ds = Affectdataset('./data/AffectNet/', train=True, transform=train_transforms, basic_aug=True)
        val_ds   = Affectdataset('./data/AffectNet/', train=False, transform=val_transforms)
    elif args.dataset == 'affectnet8class':
        num_classes = 8
        train_ds = Affectdataset_8class('./data/AffectNet/', train=True, transform=train_transforms, basic_aug=True)
        val_ds   = Affectdataset_8class('./data/AffectNet/', train=False, transform=val_transforms)
    else:
        raise ValueError('Unknown dataset')

    print('Train samples:', len(train_ds), 'Val samples:', len(val_ds))

    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True)
    val_loader   = data.DataLoader(val_ds,   batch_size=args.val_batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)

    # 模型初始化
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    # 只在首次训练时统计 FLOPs 和参数量
    #if not args.checkpoint:
        #from thop import profile
        #model_stat = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype).cuda()
        #model_stat.eval()
        #dummy_input = torch.randn(1, 3, 224, 224).cuda()
        #with torch.no_grad():
            #flops, params = profile(model_stat, inputs=(dummy_input,), verbose=False)
        #print(f"\n[Model Statistics]")
        #print(f"Params: {params / 1e6:.2f} M")
        #print(f"FLOPs: {flops / 1e9:.2f} G")
        #model_stat.train()
    model = nn.DataParallel(model).cuda()

    # 加载预训练
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model = load_pretrained_weights(model, ckpt['model_state_dict'])

    # 优化器
    if args.optimizer == 'adamw': base_opt = torch.optim.AdamW
    elif args.optimizer == 'adam': base_opt = torch.optim.Adam
    elif args.optimizer == 'sgd':  base_opt = torch.optim.SGD
    else: raise ValueError('Unsupported optimizer')
    optimizer = SAM(model.parameters(), base_opt, lr=args.lr, rho=0.05, adaptive=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 损失函数
    ce_crit  = nn.CrossEntropyLoss()
    lsce_crit= LabelSmoothingCrossEntropy(smoothing=0.2)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs, feats = model(imgs)
            loss1 = ce_crit(outputs, targets)
            loss2 = lsce_crit(outputs, targets)
            loss  = loss1 + 2*loss2
            loss.backward(); optimizer.first_step(zero_grad=True)

            # second step
            outputs, _ = model(imgs)
            loss1 = ce_crit(outputs, targets)
            loss2 = lsce_crit(outputs, targets)
            (loss1 + 2*loss2).backward(); optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds==targets).sum().item()

        train_acc = correct / len(train_ds)
        train_loss= running_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct  = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.cuda(), targets.cuda()
                outputs, _ = model(imgs)
                loss = ce_crit(outputs, targets)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds==targets).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_acc = correct / len(val_ds)
        val_f1  = f1_score(all_targets, all_preds, average='macro')
        score   = 0.67*val_f1 + 0.33*val_acc
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Score: {score:.4f}")

        # 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join('checkpoint', f'best_epoch{epoch}_acc{val_acc:.4f}.pth'))
            print('Model saved.')

if __name__ == '__main__':
    run_training()