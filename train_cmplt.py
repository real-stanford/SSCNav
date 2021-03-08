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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def validate(device, cd, test_loader, writer, batch_size,
        save_dir, epoch, CLoss,
        classes=['wall','floor','chair','door','table','picture','cabinet','cushion','window','sofa','bed','curtain','chest_of_drawers','plant','sink','stairs','ceiling','toilet','stool','towel','mirror','tv_monitor','shower','column','bathtub','counter','fireplace','lighting','beam','railing','shelving','blinds','gym_equipment','seating','board_panel','furniture','appliances','clothes','objects','misc']):
    num_channel = len(classes)
    accs = {}
    acc_actives = {}
    ious = {}
    iou_actives = {}
    for ch in range(num_channel):
        accs[ch] = 0.
        acc_actives[ch] = 0.
        ious[ch] = 0.
        iou_actives[ch] = 0.
    c_losses = []
    cd.eval()
    with torch.no_grad():
        for vid, sample in enumerate(test_loader):
            # read sample
            sem_in = sample['sem_in'].float().to(device)
            sem_gt = sample['sem_gt'].float().to(device)

            output = cd(sem_in)
            
            # calculate loss
            c_loss = CLoss(output, sem_gt.long())
            c_losses.append(float(c_loss))

            # calculate accuracy
            output = torch.argmax(output, dim=1)
            output = output.detach().view(sem_in.shape[0], -1)
            sem_gt = sem_gt.long().view(sem_in.shape[0], -1)
            sem_in = torch.argmax(sem_in, dim=1).view(sem_in.shape[0], -1) 
            for batch in range(batch_size):
                for ch in range(num_channel):
                    try:
                        sem_gt[batch]
                    except:
                        continue
                    gt_region = (sem_gt[batch] == ch) & (sem_in[batch] == num_channel)
                    out_region = (output[batch] == ch) & (sem_in[batch] == num_channel)

                    acc = torch.sum((gt_region & out_region).float())\
                        / torch.sum(gt_region.float())\
                        if torch.sum(gt_region.float()) != 0. else 0.
                    acc_active = (torch.sum(gt_region.float()) != 0.).float()
                    iou = torch.sum((gt_region & out_region).float())\
                        / torch.sum((gt_region | out_region).float())\
                        if torch.sum((gt_region | out_region).float()) != 0. \
                        else 0.
                    iou_active = (torch.sum((gt_region | out_region).float()) != 0.).float()


                    accs[ch] += acc
                    acc_actives[ch] += acc_active
                    ious[ch] += iou
                    iou_actives[ch] += iou_active
            


    with open("%s/%s.csv"% (save_dir, epoch), 'w') as csv_file:
        fieldnames = ['class', 'accuracy', 'iou']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for ch in range(num_channel):
            csv_writer.writerow({'class': classes[ch],
                    'accuracy': str(float(accs[ch]/acc_actives[ch])) if acc_actives[ch] != 0. else "null", 
                    'iou': str(float(ious[ch]/iou_actives[ch])) if iou_actives[ch] != 0. else "null"})
    mean_accuracies = []
    mean_ious = []
    for ch in range(num_channel):
        if ch == 1:
            continue
        if acc_actives[ch] != 0.:
            mean_accuracies.append(float(accs[ch]/acc_actives[ch]))
        if iou_actives[ch] != 0.:
            mean_ious.append(float(ious[ch]/iou_actives[ch]))

    if len(mean_accuracies) != 0:        
        writer.add_scalar('Test_Accuracy',
            sum(mean_accuracies)/len(mean_accuracies), epoch)
    if len(mean_ious) != 0:
        writer.add_scalar('Test_IOU',
            sum(mean_ious)/len(mean_ious), epoch)
    writer.add_scalar("Test_CrossEntropyLoss", sum(c_losses)/len(c_losses), epoch)

    cd.train()
# get experiment id
parser = argparse.ArgumentParser(description="completion")
parser.add_argument('cmplt_id', help='which experiment to pick')
args = parser.parse_args()
cmplt_id = args.cmplt_id

# start to log this process
log_dir = os.path.join('./run', cmplt_id)
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# save path for this process
save_dir = os.path.join('./result', cmplt_id)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir)

# read hyperparameters
config_path = 'configs/config.json'
with open(config_path, 'r') as f:
    data = json.load(f)
    setting = data[cmplt_id]
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

cd = ResNet(Bottleneck, DeconvBottleneck, layer_infos, num_channels).to(device)
if cd_path != "":
    cd.load_state_dict(torch.load(cd_path))
cd = nn.DataParallel(cd)
cd.train()

# loss definition
CLoss = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(device),\
        ignore_index=num_channels-1)


# optimizer
optimizer = torch.optim.Adam(cd.module.parameters(),\
        lr=lr, weight_decay=weight_decay)

# lr decay

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

    for batch_idx, sample in enumerate(train_loader):
        # start time
        start = time.time()
        
        # read sample
        sem_in = sample['sem_in'].float().to(device)
        sem_gt = sample['sem_gt'].float().to(device)
        # loss and optimize
        optimizer.zero_grad()
        output = cd(sem_in)

        loss = CLoss(output, sem_gt.long())
        loss.backward()
        optimizer.step()

        # update learning rate
        adjust_learning_rate(optimizer, lr, length * epochs, epoch * length
                + batch_idx)

        # end time
        end = time.time()
        
        # visualization
        writer.add_scalar('CrossEntropyLoss', float(loss), epoch * length + batch_idx)
         
        print("Time:%s, Epoch#:%s/%s, Iteration#:%s/%s, Loss:%s" % (end-start,
            epoch, epochs, batch_idx, length, float(loss)))

        cv2.imwrite(os.path.join(save_dir, "sem_in.png"),\
                d3_41_colors_rgb[torch.argmax(sem_in[0], dim=0).cpu().numpy()])
        cv2.imwrite(os.path.join(save_dir, "output.png"),\
                d3_41_colors_rgb[torch.argmax(output[0],\
                    dim=0).detach().cpu().numpy()]) 
        cv2.imwrite(os.path.join(save_dir, "sem_gt.png"),\
                d3_41_colors_rgb[sem_gt[0].int().cpu().numpy()])
        
        # visualize certain pairs  

        if (batch_idx + 1) not in anchors:
            continue
        
        nam = epoch * vis_freq + anchors.index(batch_idx + 1)
        cd.eval()

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

        for s in range(len(vis_train)):
            # save sample imgs
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_in.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(sem_in[s], dim=0).cpu().numpy()])
            cv2.imwrite(os.path.join(save_dir, "%s_%s_output.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(output[s], dim=0).detach().cpu().numpy()]) 
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_gt.png" % (nam,s)),\
                d3_41_colors_rgb[sem_gt[s].int().cpu().numpy()])

        # get a batch of testing sample to visualize
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

        for s in range(len(vis_test)):
            # save sample imgs
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_in_test.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(sem_in[s], dim=0).cpu().numpy()])
            cv2.imwrite(os.path.join(save_dir, "%s_%s_output_test.png" % (nam,s)),\
                d3_41_colors_rgb[torch.argmax(output[s], dim=0).detach().cpu().numpy()]) 
            cv2.imwrite(os.path.join(save_dir, "%s_%s_sem_gt_test.png" % (nam,s)),\
                d3_41_colors_rgb[sem_gt[s].int().cpu().numpy()])
        cd.train()

    
    # save model
    if (epoch+1) % save_interval == 0:
        nam = int((epoch+1) / save_interval)
        torch.save(cd.module.state_dict(), os.path.join(save_dir, "%s_cd.pth" % nam))



    ## update learning rate

    
    # evaluate this epoch's model on test set
    validate(device, cd, test_loader, writer, batch_size, save_dir, epoch,
            CLoss) 
