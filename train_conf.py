import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.backends.cudnn as cudnn

#from utils.dataloader_cmplt import CompletionDataset, MultiViewDataset
#from models import SegNet, ConfNet, ResNet, Bottleneck, DeconvBottleneck
from utils.dataloader_cmplt import MultiViewDataset
from models import ResNet, Bottleneck, DeconvBottleneck
from utils.utils import d3_41_colors_rgb

import argparse
import json
from tensorboardX import SummaryWriter
import os
import shutil
import time
import cv2
import random
import csv
from tqdm import tqdm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def validate(device, cd, fd, test_loader, writer, batch_size,
        save_dir, epoch, CLoss, ce, mask, oh, ratio):
    c_losses = [] 
    fd.eval()
    with torch.no_grad():
        for vid, sample in enumerate(test_loader):
            # read sample
            sem_in = sample['sem_in'].float().to(device)
            sem_gt = sample['sem_gt'].float().to(device)

            pred = cd(sem_in).detach()
            pred = torch.nn.functional.softmax(pred, dim=1)
            if oh:
                max_idx = torch.argmax(pred, 1, keepdim=True)
                pred = torch.FloatTensor(pred.shape).to(device)
                pred.zero_()
                pred.scatter_(1, max_idx, 1)

            if mask:

                valid = (sem_gt != sem_in.shape[1]-1) & (torch.argmax(sem_in,
                dim=1) == sem_in.shape[1]-1)
                valid = valid.float()
            sem_gt = (sem_gt == torch.argmax(pred,
                dim=1))
            
                
            output = fd(torch.cat((sem_in, pred),
                dim=1))

            if output.shape[1] == 1:
                output = output.squeeze(1)
            # calculate loss

            if not ce:

                c_loss = CLoss(output, sem_gt.float())
            else:
                c_loss = CLoss(output, sem_gt.long())


            if mask:
                c_loss_unseen = c_loss * valid * ratio
                c_loss_unseen = torch.sum(c_loss_unseen) / torch.sum(valid)
                c_loss_seen = c_loss * (1. - valid) * (1. - ratio)
                c_loss_seen = torch.sum(c_loss_seen) / torch.sum(1.-valid)
                c_loss = c_loss_unseen + c_loss_seen
            c_losses.append(float(c_loss))

            # calculate accuracy

    writer.add_scalar("Test_Loss", sum(c_losses)/len(c_losses), epoch)
    fd.train()
# get experiment id
parser = argparse.ArgumentParser(description="confidence")
parser.add_argument('conf_id', help='which experiment to pick')
args = parser.parse_args()
conf_id = args.conf_id

# start to log this process
log_dir = os.path.join('../run', conf_id)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# save path for this process
save_dir = os.path.join('../result', conf_id)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

# read hyperparameters
config_path = 'configs/config.json'
with open(config_path, 'r') as f:
    data = json.load(f)
    setting = data[conf_id]
    seed = setting['seed']
    benchmark = setting['benchmark']
    deterministic = setting['deterministic']
    enabled = setting['enabled']
    root_dirs = setting['root_dirs']
    limit = setting['limit']
    test_dirs = setting['test_dirs']
    batch_size = setting['batch_size']
    shuffle = setting['shuffle']
    num_workers = setting['num_workers']
    pin_memory = setting['pin_memory']
    weight = setting['weight']
    num_channels = setting['num_channels']
    lr = setting['lr']
    weight_decay = setting['weight_decay']
    lr_decay_rate = setting['lr_decay_rate']
    lr_epoch_per_decay = setting['lr_epoch_per_decay']
    epochs = setting['epochs']
    save_interval = setting['save_interval']
    vis_train = setting['vis_train']
    vis_test = setting['vis_test']
    vis_freq = setting['vis_freq']
    layer_infos = setting['layer_infos']
    cd_path = setting['cd_path']
    ce = setting['ce']
    mask = setting['mask']
    oh = setting['oh']
    ratio = setting['ratio']
'''
# test pipeline

num_workers = 2
save_interval = 10
'''


# set random seed
random.seed(seed)
torch.manual_seed(seed)

# cudnn setting
cudnn.benchmark = benchmark
cudnn.deterministic = deterministic
cudnn.enabled = enabled

# get cuda device
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# prepare training data
train_data = MultiViewDataset(root_dirs, num_channels, limit=limit)
train_loader = DataLoader(train_data, \
        batch_size=batch_size,\
        shuffle=shuffle,\
        num_workers=num_workers,\
        pin_memory=pin_memory)
# prepare testing data
test_data = MultiViewDataset(test_dirs, num_channels)
test_loader = DataLoader(test_data,\
        batch_size=batch_size,\
        shuffle=shuffle,\
        num_workers=num_workers,\
        pin_memory=pin_memory)

# prepare models
#cd = SegNet(n_classes=num_channels, in_channels=num_channels,\
#       add_last=add_last, dilation=dilation, configures=configures).to(device)
cd = ResNet(Bottleneck, DeconvBottleneck, layer_infos, num_channels).to(device)
if cd_path != "":
    cd.load_state_dict(torch.load(cd_path))
cd = nn.DataParallel(cd)
cd.eval()
layer_infos = [
        
              [64, 7, 2, 3],
              [3, 2, 1],
              [12, 64, 3, 2, 1, 1],
              [16, 128, 3, 1, 1, 1],
              [24, 256, 3, 1, 1, 1],
              [12, 512, 3, 1, 1, 1],
              [12, 512, 3, 1, 1, 0, 1],
              [24, 256, 3, 1, 1, 0, 1],
              [16, 128, 3, 1, 1, 0, 1],
              [12, 64, 3, 2, 1, 1, 1],
              [4, 64, 3, 2, 1, 1, 1]
        ]
fd = ResNet(Bottleneck, DeconvBottleneck, layer_infos, 41, inp=1+int(ce)).to(device)
#fd = ConfNet(41, ce).to(device)
fd = nn.DataParallel(fd)
fd.train()


# loss definition

if ce:
    if mask:
        CLoss = nn.CrossEntropyLoss(reduction='none')
    else:
        CLoss = nn.CrossEntropyLoss()
elif mask:
    CLoss = nn.MSELoss(reduction='none')
else:
    CLoss = nn.MSELoss()
# optimizer
optimizer = torch.optim.Adam(fd.module.parameters(),\
        lr=lr, weight_decay=weight_decay)

# lr decay
#lr_decay_lambda = lambda epoch: lr_decay_rate ** (epoch // lr_epoch_per_decay)
#scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
def adjust_learning_rate(optimizer, base_lr, max_iters, 
                cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr

birth = time.time()
length = int(len(train_data) / batch_size)
if len(train_data) % batch_size != 0:
    length += 1
anchors = []


for anchor in range(vis_freq):
    anchors.append(int(length * (anchor+1) /vis_freq))
for epoch in range(epochs):
    pbar = tqdm(total = length)
    for batch_idx, sample in enumerate(train_loader):
        # start time
        start = time.time()
        pbar.update(1)
        # read sample
        sem_in = sample['sem_in'].float().to(device)
        sem_gt = sample['sem_gt'].float().to(device)
        # loss and optimize
        optimizer.zero_grad()
        pred = cd(sem_in).detach()
        pred = torch.nn.functional.softmax(pred, dim=1)
        if oh:
            max_idx = torch.argmax(pred, 1, keepdim=True)
            pred = torch.FloatTensor(pred.shape).to(device)
            pred.zero_()
            pred.scatter_(1, max_idx, 1)


        if mask:
            valid = (sem_gt != sem_in.shape[1]-1) & (torch.argmax(sem_in,
                dim=1) == sem_in.shape[1]-1)
            valid = valid.float()
        sem_gt = (sem_gt == torch.argmax(pred,
            dim=1))



        output = fd(torch.cat((sem_in, pred), dim=1))
        if output.shape[1] == 1:
            output = output.squeeze(1)
        if not ce:
            loss = CLoss(output, sem_gt.float())
        else:
            loss = CLoss(output, sem_gt.long())
        if mask:

            loss_unseen = loss * valid * ratio
            loss_unseen = torch.sum(loss_unseen) / torch.sum(valid)
            loss_seen = loss * (1. - valid) * (1. - ratio)
            loss_seen = torch.sum(loss_seen) / torch.sum(1.-valid)
            loss = loss_unseen + loss_seen


        loss.backward()
        optimizer.step()

        # update learning rate
        adjust_learning_rate(optimizer, lr, length * epochs, epoch * length
                + batch_idx)

        # end time
        end = time.time()
        
        # visualization
        writer.add_scalar('Loss', float(loss), epoch * length + batch_idx)
         
        #print("Time:%s, Epoch#:%s/%s, Iteration#:%s/%s, Loss:%s" % (end-start,
        #   epoch, epochs, batch_idx, length, float(loss)))
        '''
        cv2.imwrite(os.path.join(save_dir, "sem_in.png"),\
                d3_41_colors_rgb[torch.argmax(sem_in[0], dim=0).cpu().numpy()])
        cv2.imwrite(os.path.join(save_dir, "output.png"),\
                d3_41_colors_rgb[torch.argmax(output[0],\
                    dim=0).detach().cpu().numpy()]) 
        cv2.imwrite(os.path.join(save_dir, "sem_gt.png"),\
                d3_41_colors_rgb[sem_gt[0].int().cpu().numpy()])
        '''     
        # visualize certain pairs  

        if (batch_idx + 1) not in anchors:
            continue
        
        nam = epoch * vis_freq + anchors.index(batch_idx + 1)
        fd.eval()

        # get a batch of training sample to visualize
        sem_in = None
        sem_gt = None
        for idx in vis_train:
            sample = train_data[idx]
            if sem_in is None:
                sem_in = sample['sem_in'].unsqueeze(0).float().to(device)
                sem_gt = sample['sem_gt'].unsqueeze(0).float().to(device)
            else:
                sem_in = torch.cat((sem_in,
                    sample['sem_in'].unsqueeze(0).float().to(device)), dim=0)
                sem_gt = torch.cat((sem_gt,
                    sample['sem_gt'].unsqueeze(0).float().to(device)), dim=0)
        with torch.no_grad():
            output = cd(sem_in)
            output = torch.nn.functional.softmax(output, dim=1)
            if not ce:
                foutput = fd(torch.cat((sem_in, output), dim=1)).squeeze(1)
            else:
                foutput = fd(torch.cat((sem_in, output), dim=1))
                foutput = torch.nn.functional.softmax(foutput, dim=1)[:, 1, ...]

        for s in range(len(vis_train)):
            # save sample imgs
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_in.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(sem_in[s], dim=0).cpu().numpy()])
            cv2.imwrite(os.path.join(save_dir, "%s_%s_output.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(output[s], dim=0).detach().cpu().numpy()]) 
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_gt.png" % (nam,s)),\
                d3_41_colors_rgb[sem_gt[s].int().cpu().numpy()])
            cv2.imwrite(os.path.join(save_dir, "%s_%s_conf.png" % (nam, s)),\
                    foutput[s].cpu().numpy() * 255.)
        sem_in = None
        sem_gt = None
        for idx in vis_test:
            sample = test_data[idx]
            if sem_in is None:
                sem_in = sample['sem_in'].unsqueeze(0).float().to(device)
                sem_gt = sample['sem_gt'].unsqueeze(0).float().to(device)
            else:
                sem_in = torch.cat((sem_in,
                    sample['sem_in'].unsqueeze(0).float().to(device)), dim=0)
                sem_gt = torch.cat((sem_gt,
                    sample['sem_gt'].unsqueeze(0).float().to(device)), dim=0)

        with torch.no_grad():
            output = cd(sem_in)
            output = torch.nn.functional.softmax(output, dim=1)
            if not ce:
                foutput = fd(torch.cat((sem_in, output), dim=1)).squeeze(1)
            else:
                foutput = fd(torch.cat((sem_in, output), dim=1))
                foutput = torch.nn.functional.softmax(foutput, dim=1)[:, 1, ...]
        for s in range(len(vis_test)):
            # save sample imgs
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_in_test.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(sem_in[s], dim=0).cpu().numpy()])
            cv2.imwrite(os.path.join(save_dir, "%s_%s_output_test.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(output[s], dim=0).detach().cpu().numpy()]) 
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_gt_test.png" % (nam,s)),\
                d3_41_colors_rgb[sem_gt[s].int().cpu().numpy()])
            cv2.imwrite(os.path.join(save_dir, "%s_%s_conf_test.png" % (nam, s)),\
                    foutput[s].cpu().numpy() * 255.)
        fd.train()


    pbar.close()

    # save model
    if (epoch+1) % save_interval == 0:
        nam = int((epoch+1) / save_interval)
        torch.save(fd.module.state_dict(), os.path.join(save_dir, "%s_fd.pth" % nam))



    ## update learning rate
    #scheduler.step(epoch + 1)
    
    # evaluate this epoch's model on test set
    validate(device, cd, fd, test_loader, writer, batch_size, save_dir, epoch,
            CLoss, mask=mask, ce=ce, oh=oh, ratio=ratio) 
