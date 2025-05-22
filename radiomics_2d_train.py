# radiomics_2d_train.py
"""
2D影像组学训练脚本
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, DataLoader as MonaiDataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, ToTensord
from monai.networks.nets import resnet18, densenet121, EfficientNetBN, ShuffleNetV2, ViT
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torch.nn as nn
import torch.optim as optim

# ========== 配置部分 ==========
# 选择模型类型: 'resnet', 'densenet', 'transformer', 'shufflenet'
MODEL_TYPE = 'resnet'  # 直接修改此处
DATA_DIR = './data'
BATCH_SIZE = 2
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 数据准备 ==========
label_df = pd.read_csv(os.path.join(DATA_DIR, 'label.csv'))

# 构建数据字典
train_files = []
for _, row in label_df.iterrows():
    id_ = row['ID']
    label = row['label']
    train_files.append({
        'image': os.path.join(DATA_DIR, 'images', f'{id_}.nii.gz'),
        'mask': os.path.join(DATA_DIR, 'masks', f'{id_}.nii.gz'),
        'label': label
    })

transforms = Compose([
    LoadImaged(keys=['image', 'mask']),
    AddChanneld(keys=['image', 'mask']),
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image', 'mask'])
])

dataset = Dataset(data=train_files, transform=transforms)
dataloader = MonaiDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== 模型构建 ==========
def get_model(model_type):
    if model_type == 'resnet':
        return resnet18(spatial_dims=2, n_input_channels=1, num_classes=2)
    elif model_type == 'densenet':
        return densenet121(spatial_dims=2, in_channels=1, out_channels=2)
    elif model_type == 'shufflenet':
        return ShuffleNetV2(spatial_dims=2, in_channels=1, num_classes=2)
    elif model_type == 'transformer':
        return ViT(in_channels=1, img_size=(128,128), patch_size=16, num_classes=2)  # img_size需根据实际数据调整
    else:
        raise ValueError('不支持的模型类型')

model = get_model(MODEL_TYPE).to(DEVICE)

# ========== 损失函数与优化器 ==========
loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
metric = DiceMetric(include_background=True, reduction="mean")

# ========== 训练主流程 ==========
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_data in dataloader:
        images = batch_data['image'].to(DEVICE)
        masks = batch_data['mask'].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

print('训练完成')

def main():
    # TODO: 实现2D影像组学训练流程
    pass

if __name__ == "__main__":
    main() 