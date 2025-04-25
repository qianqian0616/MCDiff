import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as nnf
from textdecoder import ClipCaptionPrefix
from tqdm import tqdm
import logging
import os
import numpy as np
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import sys
from itertools import chain
from clip_text import get_clip_outputs
from seed import worker_seeds

worker_seeds()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CLIPDataset(Dataset):
    def __init__(self, all_image_features, all_text_features, all_raw_texts, prefix_length, normalize_prefix=False):
        #从get_clip_outputs函数获取的图像和文本特征
        self.all_image_features = all_image_features
        self.all_text_features = all_text_features
        self.all_raw_texts = all_raw_texts
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("/home/ubuntu/data0/WANG/custom-reidd/gpt2")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        #设置batch_size
        self.batch_size = len(all_text_features)

        # 计算所有文本的标记长度
        all_lengths = np.array([len(self.tokenizer.tokenize(text)) for captions in all_raw_texts for text in captions])
        # 计算最大序列长度，使用均值和标准差
        self.max_seq_len = min(int(all_lengths.mean() + all_lengths.std() * 10), int(all_lengths.max()))

    def __len__(self):
        return self.batch_size
    
    def pad_tokens(self, tokenized_target):
        max_seq_len = self.max_seq_len
        #print("max_seq_len",max_seq_len)
        #print("tokenized_target",tokenized_target)
        # 计算需要填充或截断的长度
        padding = max_seq_len - len(tokenized_target)
        #print("padding",padding)
        if padding > 0:
            # 填充特殊值 -1
            tokenized_target = tokenized_target + [-1] * padding
        else:
            # 截断到最大长度
            tokenized_target = tokenized_target[:max_seq_len]
        # 创建掩码，标识实际序列的部分
        mask = torch.tensor([1] * (max_seq_len - padding) + [0] * max(0, padding), dtype=torch.float32)
        if self.prefix_length > 0:
            # 在掩码的前面加上 prefix_length 个 1
            mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        
        tokenized_target = [0 if token == -1 else token for token in tokenized_target]
        return torch.tensor(tokenized_target), mask
 
    def __getitem__(self, idx):
        #获取当前索引对应的图像和文本特征
        image_feature = self.all_image_features[idx]
        text_feature = self.all_text_features[idx]
        #print("text_feature:", text_feature)
        original_text = self.all_raw_texts[idx]
        #print("original_text",original_text)

        tokenized_text = [self.tokenizer.tokenize(text) for text in original_text]
        flattened_tokenized_text = list(chain(*tokenized_text))

        # 输出tokenized_texts
        tokenized_target = self.tokenizer.convert_tokens_to_ids(flattened_tokenized_text)
        target_tensor_padded, mask = self.pad_tokens(tokenized_target)

        return {"image":image_feature, "text":text_feature, "original_text":target_tensor_padded, "mask": mask}
         

model_path = "clip-vit-base-patch16"
root_dir = ""
img_dir = "imgs"
txt_dir = "output.json"
img_path = os.path.join(root_dir, img_dir)
txt_path = os.path.join(root_dir, txt_dir)


# 从JSON文件加载数据集
with open(txt_path, "r") as f:
    json_content = f.read()

# 解析JSON内容
dataset = json.loads(json_content)
all_image_features, all_text_features, all_raw_texts = get_clip_outputs()
print(f"原始文本数: {len(all_raw_texts)}")

# 初始化logging
logging.basicConfig(filename='text_log.txt', level=logging.INFO, format='%(asctime)s, -%(levelname)s - %(message)s')
prefix_length = 77
clip_dataset = CLIPDataset(all_image_features, all_text_features, all_raw_texts, prefix_length)
dataloader = DataLoader(clip_dataset, batch_size=32, shuffle=True)

model = ClipCaptionPrefix(prefix_length)  
#pretrained_weights_path = ""  
#model.load_state_dict(torch.load(pretrained_weights_path))
model = model.to(device)
epochs = 200
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=epochs * len(dataloader)
    )
for epoch in range(epochs):
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    model.train()
        
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        model.zero_grad()

        images = batch["text"].to(device) 
        target_text = batch["original_text"].to(device)
        mask = batch["mask"].to(device)
        #print("images", images.shape)
        #print("target_text", target_text.shape)
        #print("mask",mask.shape)

        outputs = model(target_text, images, mask)
        logits = outputs.logits[:, prefix_length - 1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_text.flatten(), ignore_index=0)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    logging.info(f"Epoch {epoch + 1}/{epochs}, average loss: {loss}")
    if (epoch + 1) % 25 == 0:
        torch.save(model.state_dict(), f"/loss_epoch_{epoch + 1}_{loss:.6f}.pth")
