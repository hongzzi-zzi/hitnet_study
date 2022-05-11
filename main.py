from __future__ import division, print_function

import argparse
import gc
import json
import os
import pdb
import time
# warning ignore
import warnings
from datetime import datetime
from re import I

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

warnings.filterwarnings('ignore')



cudnn.benchmark = True

parser = argparse.ArgumentParser(description='HITNet')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--fea_c', type=list, default=[32, 24, 24, 16, 16], help='feature extraction channels')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--ckpt_start_epoch', type=int, default=0, help='the epochs at which the program start saving ckpt')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', type=str, help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
saver = Saver(args)
print("creating new summary file")
logger = SummaryWriter(saver.experiment_dir)


logfilename = saver.experiment_dir + '/log.txt'

with open(logfilename, 'a') as log:  # wrt running information to log
    log.write('\n\n\n\n')
    log.write('-------------------NEW RUN-------------------\n')
    log.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    log.write('\n')
    json.dump(args.__dict__, log, indent=2)
    log.write('\n')

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
print(StereoDataset)
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)
# # print(type(StereoDataset))
# # print(type(test_dataset))
# # print(type(TestImgLoader))
# print(type(test_dataset))#datasets.kitti_dataset.KITTIDataset
# print(type(args.testlist))#str
# print(type(args.datapath))#str
# # print(test_dataset)
# print(type(__datasets__))#dict
# print(type(args.dataset))# str

# model, optimizer
model = HITNet(args)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    print("loading the lastest model in logdir: {}".format(args.resume))
    state_dict = torch.load(args.resume)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def train():
    val_losses = []
    train_losses = []

    min_EPE = args.maxdisp
    min_D1 = 1
    min_Thres3 = 1
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            # print(sample)
            # if batch_idx == 2:
            #     break
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            
            if batch_idx+1==len(TrainImgLoader):
                train_losses.append(loss)

            
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # print("save_scalars fin")
                save_images(logger, 'train', image_outputs, global_step)
                # print("save_images fin")
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
            with open(logfilename, 'a') as log:
                log.write('Epoch {}/{}, Iter {}/{}, train loss = {}, time = {:.3f}\n'.format(epoch_idx, args.epochs,
                                                                                                 batch_idx,
                                                                                                 len(TrainImgLoader),
                                                                                                 loss,
                                                                                                 time.time() - start_time))

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0 and epoch_idx >= args.ckpt_start_epoch:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(saver.experiment_dir, epoch_idx))
            
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            print(batch_idx)
            print(sample)

            # if batch_idx == 2:
            #     break
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            
            if batch_idx+1==len(TestImgLoader):
                val_losses.append(loss)
            
            
            
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)

            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                 batch_idx,
                                                                                 len(TestImgLoader), loss,
                                                                                 time.time() - start_time))
            with open(logfilename, 'a') as log:
                log.write('Epoch {}/{}, Iter {}/{}, test loss = {}, time = {:.3f}\n'.format(epoch_idx, args.epochs,
                                                                                            batch_idx,
                                                                                            len(TestImgLoader),
                                                                                            loss,
                                                                                            time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        if avg_test_scalars['EPE'][-1] < min_EPE:
            min_EPE = avg_test_scalars['EPE'][-1]
            minEPE_epoch = epoch_idx
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/bestEPE_checkpoint.ckpt".format(saver.experiment_dir))
        if avg_test_scalars['D1'][-1] < min_D1:
            min_D1 = avg_test_scalars['D1'][-1]
            minD1_epoch = epoch_idx
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/bestD1_checkpoint.ckpt".format(saver.experiment_dir))
        if avg_test_scalars['Thres3'][-1] < min_Thres3:
            min_Thres3 = avg_test_scalars['Thres3'][-1]
            minThres3_epoch = epoch_idx
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/bestThres3_checkpoint.ckpt".format(saver.experiment_dir))
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        with open(logfilename, 'a') as log:
            js = json.dumps(avg_test_scalars)
            log.write(js)
            log.write('\n')
        gc.collect()
    with open(logfilename, 'a') as log:
        log.write('min_EPE: {}/{}; min_D1: {}/{}'.format(min_EPE, minEPE_epoch, min_D1, minD1_epoch))
    
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_losses,label="val")
    plt.plot(train_losses,label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('Training and Validation Loss.png')
    torch.save(model, saver.experiment_dir+"/model.pickle")
    
    torch.save(model.state_dict(), 'model_weights.pth')
    # state_dict(): 학습가능한 매개변수가 담겨있는 딕셔너리 ㅇㅇ
    # 매개변수 이외에는 정보 담겨이ㅆ지 않음 ->코드상으로 모델이 구현되어있는 경우에만 로드할 수 있음




# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt, dx_gt, dy_gt = sample['left'], sample['right'], sample['disparity'], sample['dx_gt'], sample['dy_gt']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda().unsqueeze(1)#언스퀴즈(Unsqueeze) - 특정 위치에 1인 차원을 추가한다.
    dx_gt = dx_gt.cuda().unsqueeze(1)
    dy_gt = dy_gt.cuda().unsqueeze(1)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    optimizer.zero_grad()

    outputs = model(imgL, imgR)
    init_cv_pyramid = outputs["init_cv_pyramid"]
    prop_disp_pyramid = outputs["prop_disp_pyramid"]
    dx_pyramid = outputs["dx_pyramid"]
    dy_pyramid = outputs["dy_pyramid"]
    w_pyramid = outputs["w_pyramid"]
    loss = global_loss(init_cv_pyramid, prop_disp_pyramid, dx_pyramid, dy_pyramid, w_pyramid,
                       disp_gt, dx_gt, dy_gt, args.maxdisp)
    scalar_outputs = {"weighted_loss_sum": loss}
    image_outputs = {"disp_est": prop_disp_pyramid, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR, "dx_gt": dx_gt, "dy_gt": dy_gt,
                     "dx_pyramid": dx_pyramid, "dy_pyramid": dy_pyramid, "w_pyramid": w_pyramid,
                     }
    
    # print(image_outputs)
    # pdb.set_trace()

    if compute_metrics:
        with torch.no_grad():
            ################## 문제는 disp_error_image_func() 이거다 !!!!!!!!!!!!!!!!!!! 
            ##### make_nograd_func()도 이상함
            # RuntimeError: Legacy autograd function with non-static forward method is deprecated.
            # Please use new-style autograd function with static forward method. (Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)
            
            image_outputs["errormap"] = [disp_error_image_func().apply(disp_est.squeeze(1), disp_gt.squeeze(1)) for disp_est in prop_disp_pyramid]
            scalar_outputs["EPE"] = [EPE_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
            scalar_outputs["D1"] = [D1_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 1.0) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 2.0) for disp_est in prop_disp_pyramid]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 3.0) for disp_est in prop_disp_pyramid]
   
    loss.backward()
    optimizer.step()
    
    # print(image_outputs)
    # print(type(loss))# torch.Tensor
    # print(type(scalar_outputs))# dict
    # print(type(image_outputs))# dict

    return tensor2float(loss), tensor2float(scalar_outputs),image_outputs
    # return loss, scalar_outputs, image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda().unsqueeze(1)

    outputs = model(imgL, imgR)
    prop_disp_pyramid = outputs['prop_disp_pyramid']
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = torch.mean((prop_loss(torch.abs(prop_disp_pyramid[0] - disp_gt)))[mask])
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": prop_disp_pyramid, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
    scalar_outputs["EPE"] = [EPE_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1)) for disp_est in prop_disp_pyramid]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 1.0) for disp_est in prop_disp_pyramid]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 2.0) for disp_est in prop_disp_pyramid]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est.squeeze(1), disp_gt.squeeze(1), mask.squeeze(1), 3.0) for disp_est in prop_disp_pyramid]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func().apply(disp_est.squeeze(1), disp_gt.squeeze(1)) for disp_est in prop_disp_pyramid]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
    