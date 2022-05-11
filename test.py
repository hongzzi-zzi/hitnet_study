from __future__ import division, print_function

import argparse
import gc
import json
import os
import pdb
import time
import warnings
from datetime import datetime
from re import I

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import __datasets__
from loss.propagation_loss import prop_loss
from loss.total_loss import global_loss
from models.HITNet import HITNet
from utils import *




from utils.saver import Saver

# warning ignore
gc.collect()
torch.cuda.empty_cache()

warnings.filterwarnings('ignore')
plt.ion()   # 반응형 모드
cudnn.benchmark = True

pklpath='./logdir/experiment_1/model.pickle'
model=torch.load(pklpath)
print("Model load finish")
model.eval()
print("Model eval finish")

# choose dataset
StereoDataset = __datasets__["test"]
test_dataset=StereoDataset('./testdata', './filenames/test.txt', False)
TestImgLoader = DataLoader(test_dataset, batch_size=16,shuffle=False, num_workers=4, drop_last=False)

# StereoDataset = __datasets__["kitti"]
# test_dataset=StereoDataset('./kittidata', './filenames/kt.txt', False)
# TestImgLoader = DataLoader(test_dataset, batch_size=8,shuffle=False, num_workers=4, drop_last=False)

def draw_disparity(disparity_map):

	disparity_map = disparity_map.astype(np.uint8)
	norm_disparity_map = (255*((disparity_map-np.min(disparity_map))/(np.max(disparity_map) - np.min(disparity_map))))
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)


for batch_idx, sample in enumerate(TestImgLoader):
    imgL, imgR=sample['left'], sample['right']
    # 64의 배수로 패딩할것 !
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    
    gc.collect()
    torch.cuda.empty_cache()

    outputs=model(imgL, imgR)
    
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    outputs=outputs.get('prop_disp_pyramid')[0]
    outputs=outputs.cpu().detach().numpy()
    print(outputs)
    outputs = np.array(outputs[0][0])
    # print(outputs)
    # print(outputs.shape)
    
    outputs.tofile('./test_output/'+str(time.time())+'.raw')
    color_disparity=draw_disparity(outputs)
    cv2.imwrite('./test_output/'+str(time.time())+'.png',color_disparity)
    
    print("Save image and raw data")
##### print(outputs.dtype)# float32
##### 원본이미지는 메모리에러남
##### raw data: 3.3MB
##### 이미지(패딩 후 768*1088)로 나누면 4.000153186