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

import compression
import lbg

from LZW import LZW


def thresh_depth_img(img_path_o, percentile):
    root_dir = './models/'
    trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor()])
    trans_topil = transforms.ToPILImage()

    os.system(f"mkdir -p assets/")

    models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
    model = models[1]

    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')

    img_path = Path(img_path_o)

    img = Image.open(img_path)
    img_tensor = trans_totensor(img)[:3].unsqueeze(0)

    # compute consistency output
    path = './models/rgb2depth_consistency.pth'
    model_state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(model_state_dict)
    baseline_output = model(img_tensor).clamp(min=0, max=1)
    output_img = baseline_output[0].detach().numpy()[0]

    thresh_img = output_img.copy()
    thresh = np.percentile(thresh_img, percentile)
    for i in range(thresh_img.shape[0]):
        for j in range(thresh_img.shape[1]):
            if thresh_img[i, j] < thresh:
                thresh_img[i, j] = 0
            elif thresh_img[i, j] >= thresh:
                thresh_img[i, j] = 255
    index1 = img_path_o.find("\\")
    index2 = img_path_o.find(".")
    thresh_img_path = "binaryImages" + img_path_o[index1:index2] + "_BinaryDepth.jpg"
    cv2.imwrite(thresh_img_path, thresh_img)
    return thresh_img_path

def lbg_compressing(img_path, cb_size, block, epsilon=0.00005):
    """
    This function needed for doing simulation for different scenario.
    """
    index1 = img_path.find("\\")
    img_name = img_path[index1+1:]
    imgO = cv2.imread(img_path)
    imgRes = np.zeros((imgO.shape[0]//block[0], imgO.shape[1]//block[1], imgO.shape[2]))
    list_cb = np.zeros((imgO.shape[2], cb_size, block[0]*block[1]))
    for i in range(3):
        img = imgO[:, :, i]
        train_X = compression.generate_training(img, block)
        cb, cb_abs_w, cb_rel_w = lbg.generate_codebook(train_X, cb_size, epsilon)
        cb_n = np.array(cb)
        list_cb[i] = cb_n
        result = compression.encode_image(img, cb_n, block)
        imgRes[:, :, i] = result
    index = img_name.find(".")
    encoded_file_path = "CompressedFiles\\" +img_name[:index]+"_Compressed_LBG.npz"
    np.savez_compressed(file=encoded_file_path, imgComp=imgRes, list_cb=list_cb, blockList=np.array(block))
    return encoded_file_path

def lbg_decompressing(encoded_file_path):
    """
    This function needed for doing simulation for different scenario.
    """
    index1 = encoded_file_path.find("\\")
    file_name = encoded_file_path[index1+1:]
    data = np.load(encoded_file_path)
    imgC = data['imgComp']
    list_cb = data['list_cb']
    blockList = data['blockList']
    block = (blockList[0], blockList[1])
    data.close()
    imgFinRes = np.zeros((imgC.shape[0]*block[0], imgC.shape[1]*block[1], imgC.shape[2]))
    for i in range(3):
        comp = imgC[:, :, i]
        cb_n = list_cb[i]
        final_result = compression.decode_image(cb_n, comp, block)
        imgFinRes[:, :, i] = final_result
    index = file_name.find("_")
    imgComp_path = "DecompressedFiles\\"+file_name[:index]+"_Decompressed_LBG.jpg"
    cv2.imwrite(imgComp_path, imgFinRes)
    return imgComp_path

def lzw_compression(img_path):
    LZW(img_path).compress()

def lzw_decompression(compressed_file_path):
    LZW(compressed_file_path).decompress()

thresh_depth_img("Images\\Chair.jpg", 50)
thresh_depth_img("Images\\DogPortrait.jpg", 50)
thresh_depth_img("Images\\DogPortrait2.jpg", 50)

lbg_compressing("Images\\Chair.jpg", 4, (4, 10))
lbg_compressing("Images\\DogPortrait.jpg", 4, (5,6))
lbg_compressing("Images\\DogPortrait2.jpg", 4, (6, 8))

lbg_decompressing("CompressedFiles\\Chair_Compressed_LBG.npz")
lbg_decompressing("CompressedFiles\\DogPortrait_Compressed_LBG.npz")
lbg_decompressing("CompressedFiles\\DogPortrait2_Compressed_LBG.npz")

lzw_compression("Images\\Chair.jpg")
lzw_compression("Images\\DogPortrait.jpg")
lzw_compression("Images\\DogPortrait2.jpg")

lzw_decompression("CompressedFiles\\Chair_Compressed_LZW.lzw")
lzw_decompression("CompressedFiles\\DogPortrait_Compressed_LZW.lzw")
lzw_decompression("CompressedFiles\\DogPortrait2_Compressed_LZW.lzw")
