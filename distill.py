import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import socket
import torch.multiprocessing as mp
import torch.distributed as dist
from models.resnet import InsResNet18
from models.LinearModel import LinearClassifierResNet
from torchvision import transforms, datasets
def parse_option():
    parser = argparse.ArgumentParser('argument for distillation')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','resnet50', 'resnet50x2', 'resnet50x4'])
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    args = parser.parse_args()
    return args

def still():
    args = parse_option()
    if args.model == 'resnet18':
        model = InsResNet18()
        classifier= network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=2048).cuda()
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    teacher_model = 
