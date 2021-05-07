import torch
from torchvision import transforms

from modules.unet import UNet, UNetReshade

import PIL
from PIL import Image

import argparse
import os.path
from pathlib import Path
import glob
import sys
import cv2
import numpy as np

import pdb



root_dir = './models/'
trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p assets/")

models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
model = models[1]

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')

img_path_o = "assets\\Chair.jpg"
img_path = Path(img_path_o)

img = Image.open(img_path)
img_tensor = trans_totensor(img)[:3].unsqueeze(0)

# compute consistency output
path = './models/rgb2depth_consistency.pth'
model_state_dict = torch.load(path, map_location=map_location)
model.load_state_dict(model_state_dict)
baseline_output = model(img_tensor).clamp(min=0, max=1)
output_img = baseline_output[0].detach().numpy()[0]

index1 = img_path_o.find("\\")
index2 = img_path_o.find(".")
depth_img_path = "depthImages\\"+img_path_o[index1:index2]+"Depth.jpg"
cv2.imwrite(depth_img_path, output_img)