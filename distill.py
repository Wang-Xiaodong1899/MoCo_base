import os
import sys
import time
import torch
import random
import numpy as np
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
import network
import os.path as osp
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
from util import image_train, image_test, op_copy, lr_scheduler
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)

def cal_acc(loader, netF,netC, flag=True):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    logstr= 'Total Accuracy = {:.2f}%'.format(accuracy*100)
    print(logstr)
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if netB is not None:
                feas = netB(netF(inputs))
            else:
                feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>0)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    print(log_str+'\n')

    return pred_label.astype('int')


def parse_option():
    parser = argparse.ArgumentParser('argument for distillation')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18','resnet50', 'resnet50x2', 'resnet50x4'])
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--teacher_backbone',type=str,default='resnet101')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--class_num', type=int, default=12)
    parser.add_argument('--dset',type=str,default='visda')
    parser.add_argument('--resume',type=str,default='./visda-T_models/MoCo0.999_softmax_16384_resnet18_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/ckpt_epoch_240.pth')
    parser.add_argument('--worker', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    return args

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.t_dset_path).readlines()
    if args.dset=='visda':
        s_root = osp.join(args.root, 'train')
        t_root = osp.join(args.root, 'validation')
    dsets["source"] = ImageList_idx(txt_src, root = s_root, transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["target"] = ImageList_idx(txt_tar, root = t_root, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["test"] = ImageList_idx(txt_test, root = t_root, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def test_moco_model():
    args = parse_option()
    folder = './data/txt/'
    args.s_dset_path = folder + args.dset + '/' + 'train' + '.txt'
    args.t_dset_path = folder + args.dset + '/' + 'validation' + '.txt'
    args.root = '/data1/junbao3/xdwang/dsets/visda/'
    dset_loaders = data_load(args)
    model = InsResNet18().cuda()
    classify_head= network.feat_classifier(type='wn', class_num = args.class_num, bottleneck_dim=128).cuda()
    print("phase begins => loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'])
    param_group=[]
    for k, v in model.named_parameters():
        v.requires_grad = False
    for k, v in classify_head.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    optimizer_moco = optim.SGD(param_group, lr = 0.01, momentum=0.9,weight_decay=0.0001)
    optimizer_moco = op_copy(optimizer_moco)
    model.eval()
    classify_head.train()
    max_iter = 20 * len(dset_loaders["target"])
    iter_num = 0
    pbar = tqdm(total = max_iter)
    test_interval = max_iter//10
    while iter_num < max_iter:
        try:
            inputs_test, labels_test, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["source"])
            inputs_test, labels_test, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue
        iter_num = iter_num+1
        inputs_test,labels_test = inputs_test.cuda(),labels_test.cuda()
        lr_scheduler(optimizer_moco, iter_num=iter_num, max_iter=max_iter)
        stu_logit = classify_head(model(inputs_test))
        fintune_loss = nn.CrossEntropyLoss()(stu_logit, labels_test)
        optimizer_moco.zero_grad()
        fintune_loss.backward()
        optimizer_moco.step()
        if iter_num%test_interval==0 or iter_num==0 or iter_num==max_iter:
            model.eval()
            classify_head.eval()
            print('test begins=>')
            acc_s_te, acc_list = cal_acc(dset_loaders['test'], model, classify_head, True)
            log_str = 'Epoch:{}/{}; Accuracy = {:.2f}%'.format(iter_num//test_interval, 20, acc_s_te) + '\n' + acc_list
            print(log_str)
            model.eval()
            classify_head.train()
            moco_save_path = osp.join('./ontarget_ckpts', 'moco_head_epoch_{}.pth'.format(iter_num//test_interval))
            torch.save(classify_head.state_dict(),moco_save_path)
        pbar.update(1)
    moco_save_path = osp.join('./ontarget_ckpts', 'moco_head_final.pth')
    torch.save(classify_head.state_dict(),moco_save_path)
def still():
    args = parse_option()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    ontar_folder = './ontarget_ckpts'
    args.out_file = open(osp.join(ontar_folder, 'log_' + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    teacher_model_path = './shot_ckpt'
    netF = network.ResBase(res_name=args.teacher_backbone).cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=256).cuda()
    netC = network.feat_classifier(type='wn', class_num = args.class_num, bottleneck_dim=256).cuda()

    param_name = 'par_0.3_seed_2020'
    modelpath = osp.join(teacher_model_path,'target_F_'+param_name+'.pt')
    netF.load_state_dict(torch.load(modelpath))
    modelpath = osp.join(teacher_model_path,'target_B_'+param_name+'.pt') 
    netB.load_state_dict(torch.load(modelpath))
    modelpath = osp.join(teacher_model_path,'target_C_'+param_name+'.pt') 
    netC.load_state_dict(torch.load(modelpath))

    folder = './data/txt/'
    args.s_dset_path = folder + args.dset + '/' + 'train' + '.txt'
    args.t_dset_path = folder + args.dset + '/' + 'validation' + '.txt'
    args.root = '/data1/junbao3/xdwang/dsets/visda/'
    dset_loaders = data_load(args)
    model = InsResNet18().cuda()
    classify_head= network.feat_classifier(type='wn', class_num = args.class_num, bottleneck_dim=128).cuda()
    teach_model = InsResNet18().cuda()
    teach_classify_head= network.feat_classifier(type='wn', class_num = args.class_num, bottleneck_dim=128).cuda()
    for i in range(1):##phase 0 
        # phase_path = osp.join(ontar_folder,'ckpt_phase_0_head.pth')
        # if os.path.isfile(phase_path):
        #     save_model = os.path.join('./ontarget_ckpts', 'ckpt_phase_phase_0_model.pth')
        #     teach_model.load_state_dict(torch.load(save_model))
        #     save_model = os.path.join('./ontarget_ckpts', 'ckpt_phase_phase_0_head.pth')
        #     teach_classify_head.load_state_dict(torch.load(save_model))
        #     print('phase 0 resume successfully!\n')
        #     # break
        print("phase begins => loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        moco_save_path = osp.join('./ontarget_ckpts', 'moco_head_final.pth')
        classify_head.load_state_dict(torch.load(moco_save_path))
        param_group=[]
        for k, v in model.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        for k, v in classify_head.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        optimizer_moco = optim.SGD(param_group, lr = 0.01, momentum=0.9,weight_decay=0.0001)
        optimizer_moco = op_copy(optimizer_moco)
        model.train()
        classify_head.train()
        max_iter = 10 * len(dset_loaders["target"])
        iter_num = 0
        pbar = tqdm(total = max_iter)
        test_interval = max_iter//10
        netF.eval()
        netB.eval()
        netC.eval()
        print("generates pseudo labels=>")
        mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)###原来的预测是不变的，可以离线
        mem_label = torch.from_numpy(mem_label).cuda()
        while iter_num < max_iter:
            try:
                inputs_test, _, tar_idx = iter_test.next()
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = iter_test.next()

            if inputs_test.size(0) == 1:
                continue
            inputs_test = inputs_test.cuda()
            pred = mem_label[tar_idx]
            iter_num = iter_num+1
            lr_scheduler(optimizer_moco, iter_num=iter_num, max_iter=max_iter)
            stu_logit = classify_head(model(inputs_test))
            distill_loss = nn.CrossEntropyLoss()(stu_logit, pred)
            optimizer_moco.zero_grad()
            distill_loss.backward()
            optimizer_moco.step()
            pbar.update(1)
            if iter_num%test_interval==0 or iter_num==0 or iter_num==max_iter:
                model.eval()
                classify_head.eval()
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], model, classify_head, True)
                log_str = 'Phase {}; Epoch:{}/{}; Accuracy = {:.2f}%'.format(i, iter_num//test_interval, 10, acc_s_te) + '\n' + acc_list
                print(log_str)
                args.out_file.write(log_str+'\n')
                args.out_file.flush()
                model.train()
                classify_head.train()
        # moment_update(model,teach_model,0)
        # moment_update(classify_head,teach_classify_head,0)
        ##teacher model 在phase结束后更新为最新的student model
        print('==> Saving...')
        if not os.path.isdir(ontar_folder):
            os.makedirs(ontar_folder)
        save_model = os.path.join('./ontarget_ckpts', 'ckpt_phase_0_model.pth')
        torch.save(model.state_dict(), save_model)
        teach_model.load_state_dict(model.state_dict())
        save_model = os.path.join('./ontarget_ckpts', 'ckpt_phase_0_head.pth')
        torch.save(classify_head.state_dict(), save_model)
        teach_classify_head.load_state_dict(classify_head.state_dict())
    for i in range(1,3):##phase 1-2
        ###开始时student恢复原始的moco model，classify head重新初始化
        classify_head= network.feat_classifier(type='wn', class_num = args.class_num, bottleneck_dim=128).cuda()
        moco_save_path = osp.join('./ontarget_ckpts', 'moco_head_final.pth')
        classify_head.load_state_dict(torch.load(moco_save_path))
        print("phase begins => loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        param_group=[]
        for k, v in model.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        for k, v in classify_head.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        optimizer_moco = optim.SGD(param_group, lr = 0.01, momentum=0.9,weight_decay=0.0001)
        optimizer_moco = op_copy(optimizer_moco)
        model.train()
        classify_head.train()
        max_iter = 10 * len(dset_loaders["target"])
        test_interval = max_iter//10
        iter_num = 0
        pbar = tqdm(total = max_iter)
        teach_model.eval()
        teach_classify_head.eval()
        print("generates pseudo labels=>")
        mem_label = obtain_label(dset_loaders['test'], teach_model, None, teach_classify_head, args)###原来的预测是不变的，可以离线
        mem_label = torch.from_numpy(mem_label).cuda()
        while iter_num < max_iter:
            try:
                inputs_test, _, tar_idx = iter_test.next()
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = iter_test.next()
            inputs_test = inputs_test.cuda()
            if inputs_test.size(0) == 1:
                continue
            iter_num = iter_num+1
            lr_scheduler(optimizer_moco, iter_num=iter_num, max_iter=max_iter)
            stu_logit = classify_head(model(inputs_test))
            pred = mem_label[tar_idx]
            distill_loss = nn.CrossEntropyLoss()(stu_logit, pred)
            optimizer_moco.zero_grad()
            distill_loss.backward()
            optimizer_moco.step()
            pbar.update(1)
            if iter_num%test_interval==0 or iter_num==0 or iter_num==max_iter:
                model.eval()
                classify_head.eval()
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], model, classify_head, True)
                log_str = 'Phase {}; Epoch:{}/{}; Accuracy = {:.2f}%'.format(i, iter_num//test_interval, 10, acc_s_te) + '\n' + acc_list
                print(log_str)
                args.out_file.write(log_str+'\n')
                args.out_file.flush()
                model.train()
                classify_head.train()
        # moment_update(model,teach_model,0)
        # moment_update(classify_head,teach_classify_head,0)
        print('==> Saving...')
        save_model = os.path.join('./ontarget_ckpts', 'ckpt_phase_{}_model.pth'.format(i))
        torch.save(model.state_dict(), save_model)
        teach_model.load_state_dict(model.state_dict())
        save_model = os.path.join('./ontarget_ckpts', 'ckpt_phase_{}_head.pth'.format(i))
        torch.save(classify_head.state_dict(), save_model)
        teach_classify_head.load_state_dict(classify_head.state_dict())


if __name__ == '__main__':
    # still()
    test_moco_model()
