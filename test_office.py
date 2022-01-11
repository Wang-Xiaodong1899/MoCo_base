import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse


from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from models.resnet import InsResNet50,InsResNet18,ResBase

from dataset import  ImageList_idx
import os.path as osp

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_sour31 = open(args.s_dset_path).readlines()
    txt_tar31 = open(args.t_dset_path).readlines()
    txt_test = open(args.t_dset_path).readlines()
    txt_sour = [i for i in txt_sour31 if float(i.split()[1])<10]
    txt_tar = [i for i in txt_tar31 if float(i.split()[1])<10]
    s_root = args.root
    t_root = args.root
    dsets["source"] = ImageList_idx(txt_sour,root=s_root, transform=image_test())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, root = t_root, transform=image_test())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, root = t_root, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def test_office(args):
    dset_loaders = data_load(args)
    feature_extractor = ResBase(res_name=args.net).cuda()
    feature_extractor.eval()
    start_test = True
    with torch.no_grad():
        iter_sour = iter(dset_loaders['source'])
        for i in range(len(dset_loaders['source'])):
            data = iter_sour.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feat = feature_extractor(inputs)##1000 dim
            if start_test:
                all_feat = feat.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feat = torch.cat((all_feat, feat.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders['target'])
        for i in range(len(dset_loaders['target'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feat = feature_extractor(inputs)##128 dim
            if start_test:
                all_feat_t = feat.float().cpu()
                all_label_t = labels.float()
                start_test = False
            else:
                all_feat_t = torch.cat((all_feat_t, feat.float().cpu()), 0)
                all_label_t = torch.cat((all_label_t, labels.float()), 0)
    ###cpu temp
    print(all_feat.shape)
    print(all_feat_t.shape)
    feat_s_norm = all_feat/all_feat.norm(dim=-1, keepdim=True)#(Ns, 1000)
    feat_t_norm = all_feat_t/all_feat_t.norm(dim=-1, keepdim=True)#(Nt, 1000)
    dis_matrix = feat_t_norm.mm(feat_s_norm.t())#(Nt, Ns)
    dis_matrix = torch.cdist(all_feat_t, all_feat,p=2)
    # dis_matrix = torch.cdist(feat_t_norm, feat_s_norm,p=2)
    _, idx_list = torch.topk(dis_matrix, k=10, dim=1, largest=False)###(Nt, 50)
    acc=0
    for ii in range(idx_list.shape[0]):
        D={}
        P = all_label[idx_list[ii]].int()###idxs->labels
        for jj in range(args.class_num):##one sample labels
            D[jj]=0
        for jj in P.numpy().tolist():
            D[jj]+=1

        p = P.numpy().tolist()[0]###
        for k,v in D.items():
            if v > D[p]:
                p=k
        if p==all_label_t[ii].numpy().item():
            acc+=1
    print(acc/idx_list.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test office')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office', choices=['visda', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--net', type=str, default='resnet18', help="vgg16, resnet50, resnet101")
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        args.root = '/data1/junbao3/xdwang/dsets/office'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.folder = './data/txt/'
    args.s_dset_path = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.s] + '.txt'
    args.t_dset_path = args.folder + args.dset + '/' + 'labeled_source_images_'+names[args.t] + '.txt'
    print(names[args.s]+'->'+names[args.t])
    test_office(args)
