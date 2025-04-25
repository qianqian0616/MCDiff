import numpy as np
import torch.nn as nn
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from models.clip import get_clip_outputs
from models.diffusion import Diffusion
from tqdm import tqdm
import logging
from seed import worker_seeds

worker_seeds()


class CLIPDataset(Dataset):
    def __init__(self, all_image_features, all_text_features):
        #从get_clip_outputs函数获取的图像和文本特征
        self.all_image_features = all_image_features
        self.all_text_features = all_text_features
        #设置batch_size
        self.batch_size = len(all_text_features)

    def __len__(self):
        return self.batch_size
        
    def __getitem__(self, idx):
        #获取当前索引对应的图像和文本特征
        image_feature = self.all_image_features[idx]
        text_feature = self.all_text_features[idx]
        return {"image":image_feature, "text":text_feature}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
all_image_features, all_text_features = get_clip_outputs()
# 定义数据集和数据加载器
clip_dataset = CLIPDataset(all_image_features, all_text_features)
dataloader = DataLoader(clip_dataset, batch_size=12, shuffle=True)

class Config:
    def __init__(self):
        self.in_embed = 512
        self.in_channels = 2
        self.ch = 128
        self.out_ch = 1
        self.ch_mult = (1, 2, 2, 2)
        self.num_res_blocks = 2
        self.attn_resolutions = [512, 256, 128]
        self.resamp_with_conv = 'False'

        self.num_diffusion_timesteps = 1000
        self.timesteps = 20
        self.dropout = 0.0
        self.beta_schedule = 'linear'
        self.beta_start = 0.0015
        self.beta_end = 0.0195
        self.model_ema = False
        self.type = 'simple'
        self.var_type = 'fixedlarge'
        self.ema_rate = 0.9999
        self.ema = False
        self.eta = 0.0

config = Config()
diffusion_model = Diffusion(config)
pretrained_weights_path = ""  
diffusion_model.model.load_state_dict(torch.load(pretrained_weights_path))
logging.basicConfig(filename='train_con_log.txt', level=logging.INFO, format='%(asctime)s, -%(levelname)s - %(message)s')
diffusion_model.model = diffusion_model.model.to(device)
epochs = 50
optimizer = optim.Adam(diffusion_model.model.parameters(), lr=1e-6)

#模型训练
for epoch in range(epochs):
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        image = batch["image"].to(device)
        text = batch["text"].to(device)
        loss_image = diffusion_model.train(image, text)
        loss = loss_image

            #text = batch["text"].to(device)
            #loss_text = diffusion_model.train(text)

            #loss = loss_image + loss_text

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # 打印该轮次的平均损失
    average_loss = total_loss / len(dataloader)
    logging.info(f"Epoch {epoch + 1}/{epochs}, loss: {average_loss}")
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss}")

    # 保存权重
    if (epoch + 1) % 10 == 0:
        torch.save(diffusion_model.model.state_dict(), f"")


