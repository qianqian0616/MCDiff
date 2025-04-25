import torch
import torch.nn as nn
from seed import worker_seeds
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from imgdecoder import Decoder
from patchgan import NLayerDiscriminator
from clip_img import get_clip_outputs
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import logging
import os, hashlib
import requests
from tqdm import tqdm
from torchvision import models
from collections import namedtuple
import json
import numpy as np

model_path = "clip-vit-base-patch16"
root_dir = "data/CUHK-PEDES/"
img_dir = "imgs"
txt_dir = "reid_raw"
img_path = os.path.join(root_dir, img_dir)
txt_path = os.path.join(root_dir, txt_dir)

worker_seeds()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        #print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        
        lins = [lin.to(device) for lin in lins]
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp.to(device) - self.shift.to(device)) / self.scale.to(device)


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x].to(device))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X.to(device))
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    x = x.to(device)
    return x.mean([2,3],keepdim=keepdim)


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2])

    def nll(self, sample, dims=[1,2]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class YourDataset(Dataset):
    def __init__(self, all_image_features, all_text_features, all_image_paths):
        self.all_image_features = all_image_features
        self.all_text_features = all_text_features
        self.all_image_paths = all_image_paths

        self.batch_size = len(all_text_features)

    def __len__(self):
        return self.batch_size
    

    def __getitem__(self, idx):
        image_feature = self.all_image_features[idx]
        text_feature = self.all_text_features[idx]
        image_paths = self.all_image_paths[idx]

        image = Image.open(image_paths).convert('RGB') 
        image = transforms.Compose([
            transforms.Resize((96, 32)),
            transforms.ToTensor()    
        ])(image)

        return {'image': image, 'clip_vector_image': image_feature, 'clip_vector_text': text_feature}


# 调用 get_clip_outputs 函数获取 CLIP 输出的图像向量和文本向量
all_image_features, all_text_features, all_image_paths = get_clip_outputs()
# 创建包含图像路径和向量的数据集
custom_dataset = YourDataset(all_image_features, all_text_features, all_image_paths)
# 定义 DataLoader
dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True)
# 初始化logging
logging.basicConfig(filename='img_log.txt', level=logging.INFO, format='%(asctime)s, -%(levelname)s - %(message)s')

class DecoderConfig:
    def __init__(self):
        self.width = 512
        self.heads = 8
        self.layers = 12
        self.output_dim = 576
        self.img_h = 24
        self.img_w = 8
        self.in_channels = 3
        self.ch = 128
        self.out_ch = 3
        self.ch_mult = (1, 2, 2)
        self.num_res_blocks = 2
        self.attn_resolutions = [48, ]
        self.resamp_with_conv = 'False'
        self.dropout = 0.1
        self.tanh_out= 'False'
      

# 实例化模型
config = DecoderConfig()
Decoder = Decoder(config)
# 加载之前保存的权重文件
pretrained_weights_path = ""  
Decoder.load_state_dict(torch.load(pretrained_weights_path))
Decoder.to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

D = NLayerDiscriminator().apply(weights_init)
D.to(device)

# VAE 损失
reconstruction_function = nn.MSELoss()

# GAN 损失
def discriminator_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def generator_loss(logits_fake):
    g_loss = -torch.mean(logits_fake)
    return g_loss

def get_last_layer():
    return Decoder.conv_out.weight

def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * 0.5
        return d_weight

def quant_conv(x):
    conv_layer =  torch.nn.Conv1d(1, 2, 1)
    conv_layer.to(device)
    quant_conv = conv_layer(x)
    return quant_conv

    

def post_quant_conv(x):
    conv_layer = torch.nn.Conv1d(1, 1, 1)
    conv_layer.to(device)
    post_quant_conv = conv_layer(x)
    return post_quant_conv
    
optimizer_Decoder = optim.Adam(Decoder.parameters(), lr=0.00001, betas=(0.5, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.9))

epochs = 100
#模型训练
for epoch in range(epochs):
    Decoder.train()
    D.train()
    total_loss_Decoder = 0.0
    total_loss_d = 0.0
    if epoch < 80:
        disc_factor = 0
    else:
        disc_factor = 1

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        # 把像素值、向量值归一化到[-1, 1]
        im = batch["image"].to(device)
        bs = im.shape[0]
        im = im * 2 - 1

        clip_vectors_text = batch['clip_vector_text'].to(device)
        recon_im = Decoder(clip_vectors_text)
        # 像素级重构loss
        rec_loss = reconstruction_function(recon_im.contiguous(), im.contiguous())
        # 感知loss
        perceptual_loss = LPIPS().eval()
        p_loss = perceptual_loss(im.contiguous(), recon_im.contiguous())
        rec_loss = rec_loss + 1.0 * p_loss

        # 生成器loss
        g_logits_fake = D(recon_im.contiguous())
        g_loss = generator_loss(g_logits_fake)

        logvar1 = nn.Parameter(torch.ones(size=()) * 0)
        nll_loss = rec_loss / torch.exp(logvar1) + logvar1
        weighted_nll_loss = nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        last_layer = get_last_layer()
        d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        loss_Decoder = weighted_nll_loss + d_weight * disc_factor * g_loss
        #disc_factor_g = adopt_weight(1.0, global_step=50000, threshold=50001)
        #loss_Decoder = weighted_nll_loss + 0.000001 * KLD + d_weight * disc_factor * g_loss
        #print("Decoder loss:", loss_Decoder.item())

        # 反向传播和优化
        optimizer_Decoder.zero_grad()
        loss_Decoder.backward()
        optimizer_Decoder.step()

        total_loss_Decoder += loss_Decoder.item()
        average_loss_Decoder = total_loss_Decoder / len(dataloader)

        # 判别网络
        logits_real = D(im.contiguous().detach())
        logits_fake = D(recon_im.contiguous().detach())
        #disc_factor_d = adopt_weight(1.0, global_step=50002, threshold=50001)
        loss_d = disc_factor * discriminator_loss(logits_real, logits_fake)   #判别器的loss 
        #print("Discriminator loss:", loss_d.item())

        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()  

        total_loss_d += loss_d.item()
        average_loss_d = total_loss_d / len(dataloader)
    
    print("average Decoder loss:", average_loss_Decoder)
    print("average d loss:", average_loss_d)
    logging.info(f"Epoch {epoch + 1}/{epochs}, average Decoder loss: {average_loss_Decoder}, average discriminator loss: {average_loss_d}")
    # 保存权重
    if (epoch + 1) % 10 == 0:
        torch.save(Decoder.state_dict(), f"")
