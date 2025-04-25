import argparse
import torch
import numpy as np
from models.diffusion import Diffusion
from models.clip import get_clip_outputs
from models.testtext import generated_text
from models.testunet import Testunet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seed import worker_seeds

worker_seeds()

class CLIPDataset(Dataset):
    def __init__(self, all_image_features, all_text_features):
        self.all_image_features = all_image_features
        self.all_text_features = all_text_features
        self.batch_size = len(all_text_features)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        image_feature = self.all_image_features[idx]
        text_feature = self.all_text_features[idx]
        return {"image": image_feature, "text":text_feature}

def diffusion_sample(data_loader, diffusion_model, num_epochs=1):
    sampled_images = []  # 用于存储每轮迭代的图像采样结果
    sampled_texts = []   #用于存储每轮迭代的文本采样结果

    for epoch in range(num_epochs):   
        for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = batch["image"].to(device)
            images_output = diffusion_model.sample(images)
            sampled_images.append(images_output)

    batch_index = 0 
    for images_output in sampled_images: 
        batch_index += 1
        Testunet(images_output, batch_index)
        generated_text(images_output, "output_file.txt")

    return sampled_images, sampled_texts  

class Config:
    def __init__(self):
        self.in_embed = 512
        self.in_channels = 1
        self.ch = 128
        self.out_ch = 1
        self.ch_mult = (1, 2, 2, 2)
        self.num_res_blocks = 2
        self.attn_resolutions = [512, 256, 128]
        self.resamp_with_conv = 'False'
        self.num_diffusion_timesteps = 1000
        self.timesteps = 1000
        self.dropout = 0.1
        self.beta_schedule = 'linear'
        self.beta_start = 0.00001
        self.beta_end = 0.02
        self.model_ema = False
        self.type = 'simple'
        self.var_type = 'fixedlarge'
        self.ema_rate = 0.9999
        self.ema = False
        self.eta = 0.0
        self.skip_type = 'uniform'

config = Config()
diffusion_model = Diffusion(config)  
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# 加载训练好的权重
pretrained_model_path = ''
state_dict = torch.load(pretrained_model_path)

# 将加载的权重应用到模型
diffusion_model.model.load_state_dict(state_dict)
diffusion_model.model = diffusion_model.model.to(device)

           
if __name__ == "__main__":

    # 加载CLIP输出的文本和图像向量
    all_image_features, all_text_features = get_clip_outputs()
    # 定义数据集和数据加载器
    clip_dataset = CLIPDataset(all_image_features, all_text_features)
    data_loader = DataLoader(clip_dataset, batch_size=1, shuffle=False)
    
    # 进行扩散模型的采样
    sampled_images, sampled_texts = diffusion_sample(data_loader, diffusion_model, num_epochs=1)
   
    
