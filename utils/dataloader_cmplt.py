import torch
from torch.utils.data import Dataset, DataLoader
import time
import h5py
import numpy as np
import os
import cv2
import random

# the dataset used to train completion model
class BaseDataset(Dataset):
    # self.samples: record directory of each hdf5 file
    def __init__(self, root_dirs, num_channel, limit=None):
        self.samples = []
        self.num_channel = num_channel
        for root_dir in root_dirs:
            for sample in sorted(os.listdir(root_dir)):
                if sample.endswith('.hdf5'):
                    file_dir = os.path.join(root_dir, sample)
                    try:
                        f = h5py.File(file_dir, 'r')
                        self.samples.append(file_dir)
                    except:
                        continue

        if limit is not None and int(len(self.samples) * 4 / limit) >= 1:
            ratio = int(len(self.samples)*4 / limit)
            self.samples = [sample for i, sample in enumerate(self.samples) if
                    i % ratio == 0]
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self):
        raise NotImplementedError

class CompletionDataset(BaseDataset):
    def __init__(self, root_dirs, num_channel, limit=None):
        super(CompletionDataset, self).__init__(root_dirs, num_channel, limit)
    # for each hdf5 file
    # there are four rotations thus four training samples
    def __len__(self):
        return 4*len(self.samples)

    def __getitem__(self, idx):
        fid = int(idx / 4) # find which sample
        rid = idx - fid * 4 # find which rotation

        with h5py.File(self.samples[fid], 'r') as f:
            sem_in = f['single_height_sems'][str(rid)][()]
            sem_in = np.eye(self.num_channel)[sem_in].transpose(2, 0, 1)
            sem_gt = torch.from_numpy(np.rot90(f['height_sem'][()], rid+1).copy())
            unknown = (sem_gt < 0) | (sem_gt >= self.num_channel)
            sem_gt[unknown] = self.num_channel - 1
            sample = {
                    "sem_in": torch.from_numpy(sem_in.copy()),
                    "sem_gt": sem_gt
                    }
        return sample

class MultiViewDataset(BaseDataset):
    def __init__(self, root_dirs, num_channel, choicess = None, limit=None):
        super(MultiViewDataset, self).__init__(root_dirs, num_channel, limit)
        self.choicess = choicess
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        if self.choicess is None:
            return self.getitem(idx)
        else:
            return self.getitem_fix(idx, self.choicess[idx])
    def getitem_fix(self, idx, choices):
        with h5py.File(self.samples[idx], 'r') as f:
            # 0 -> right
            # 1 -> back
            # 2 -> left
            # 3 -> front
            #rid = random.choice(range(4))
            rid = choices[0]
            sem_in = f['single_height_sems'][str(rid)][()]
            sem_in = np.eye(self.num_channel)[sem_in].transpose(2, 0, 1)
            unknown = (sem_in[self.num_channel - 1] == 1)
            for i in choices:
                print(i)
            #if random.random() < 0.5:
            if choices[1] < 0.5:
                sem_left = f['single_height_sems'][str((rid-1) % 4)][()]
                sem_left = np.rot90(sem_left, k=1)
                sem_left = np.eye(self.num_channel)[sem_left].transpose(2, 0, 1)
                new = (sem_left[self.num_channel-1] != 1) 
                sem_in[:, new] = sem_left[:, new]
                unknown = (sem_in[self.num_channel - 1] == 1)

            #if random.random() < 0.5:
            if choices[2] < 0.5:
                sem_right = f['single_height_sems'][str((rid + 1) % 4)][()]
                sem_right = np.rot90(sem_right, k=3)
                sem_right = np.eye(self.num_channel)[sem_right].transpose(2, 0, 1)
                new = (sem_right[self.num_channel - 1] != 1)
                sem_in[:, new] = sem_right[:, new]
                unknown = (sem_in[self.num_channel - 1] == 1)
            #if random.random() < 0.5:
            if choices[3] < 0.5:
                sem_back = f['single_height_sems'][str((rid + 2) % 4)][()]
                sem_back = np.rot90(sem_back, k=2)
                sem_back = np.eye(self.num_channel)[sem_back].transpose(2, 0, 1)
                new = (sem_back[self.num_channel - 1] != 1)
                sem_in[:, new] = sem_back[:, new]
                unknown = (sem_in[self.num_channel - 1] == 1)
            # height_sem -> front

            sem_gt = torch.from_numpy(np.rot90(f['height_sem'][()], rid+1).copy())
            unknown = (sem_gt < 0) | (sem_gt >= self.num_channel)
            sem_gt[unknown] = self.num_channel - 1
            sample = {
                    "sem_in": torch.from_numpy(sem_in.copy()),
                    "sem_gt": sem_gt
                    }
            return sample
            
    def getitem(self, idx):
        with h5py.File(self.samples[idx], 'r') as f:
            # 0 -> right
            # 1 -> back
            # 2 -> left
            # 3 -> front
            rid = random.choice(range(4))
            
            sem_in = f['single_height_sems'][str(rid)][()]
            sem_in = np.eye(self.num_channel)[sem_in].transpose(2, 0, 1)
            unknown = (sem_in[self.num_channel - 1] == 1)
            
            if random.random() < 0.5:
                sem_left = f['single_height_sems'][str((rid-1) % 4)][()]
                sem_left = np.rot90(sem_left, k=1)
                sem_left = np.eye(self.num_channel)[sem_left].transpose(2, 0, 1)
                new = (sem_left[self.num_channel-1] != 1) 
                sem_in[:, new] = sem_left[:, new]
                unknown = (sem_in[self.num_channel - 1] == 1)

            if random.random() < 0.5:
                sem_right = f['single_height_sems'][str((rid + 1) % 4)][()]
                sem_right = np.rot90(sem_right, k=3)
                sem_right = np.eye(self.num_channel)[sem_right].transpose(2, 0, 1)
                new = (sem_right[self.num_channel - 1] != 1)
                sem_in[:, new] = sem_right[:, new]
                unknown = (sem_in[self.num_channel - 1] == 1)
            if random.random() < 0.5:
                sem_back = f['single_height_sems'][str((rid + 2) % 4)][()]
                sem_back = np.rot90(sem_back, k=2)
                sem_back = np.eye(self.num_channel)[sem_back].transpose(2, 0, 1)
                new = (sem_back[self.num_channel - 1] != 1)
                sem_in[:, new] = sem_back[:, new]
                unknown = (sem_in[self.num_channel - 1] == 1)
            # height_sem -> front

            sem_gt = torch.from_numpy(np.rot90(f['height_sem'][()], rid+1).copy())
            unknown = (sem_gt < 0) | (sem_gt >= self.num_channel)
            sem_gt[unknown] = self.num_channel - 1
            sample = {
                    "sem_in": torch.from_numpy(sem_in.copy()),
                    "sem_gt": sem_gt
                    }
            return sample
            

if __name__ == '__main__':

    from utils import d3_41_colors_rgb
    root_dirs \
    = ['../../buffer/test_data_gen_1','../../buffer/test_data_gen_2',
        '../../buffer/test_data_gen_3']
    CData = MultiViewDataset(root_dirs, 41)
    print(len(CData))
    
