import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision

import h5py
import numpy as np
import os
import cv2
import skimage.transform
import random
import matplotlib


#from utils import d3_41_colors_rgb

class BaseDataset(Dataset):
    def __init__(self,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    def multi_scale_aug(self, image, label=None,
                        rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label


    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

     
        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )

        return image, label


                
                
# segmentation dataset for HRNet
class HRNetDataset(BaseDataset):
    # self.samples: record directory of each hdf5 file
    def __init__(self, root_dirs, num_channel, \
            multi_scale, flip,\
            base_size, crop_size, downsample_rate,\
            scale_factor, \
            mode="train",\
            mean=[0.485, 0.456, 0.406], \
            std=[0.229, 0.224, 0.225],
            limit=None):
        super(HRNetDataset, self).__init__(num_channel-1,\
                base_size, crop_size, downsample_rate,\
                scale_factor, mean, std,)

        self.samples = []
        self.num_channel = num_channel
        
        self.multi_scale = multi_scale
        self.flip = flip
        
        self.mode = mode
        
        for root_dir in root_dirs:
            for sample in os.listdir(root_dir):
                if sample.endswith('.hdf5'):
                    file_dir = os.path.join(root_dir, sample)
                    try:
                        f = h5py.File(file_dir, 'r')
                        self.samples.append(file_dir)
                    except:
                        continue
        if limit is not None and int(len(self.samples) * 4 / limit) >= 1:
            ratio = int(len(self.samples)*4 / limit)
            self.samples = [sample for i, sample in enumerate(self.samples) if i % ratio == 0]
    # for each hdf5 file
    # there are four rotations thus four training samples
    def __len__(self):
        return 4*len(self.samples)

    def __getitem__(self, idx):
        fid = int(idx / 4) # find which sample
        rid = idx - fid * 4 # find which rotation
        with h5py.File(self.samples[fid], 'r') as f:
            rgb = f['rgb_imgs'][str(rid)][()][..., [2, 1, 0]]#rgb -> bgr
            sem = f['sem_imgs'][str(rid)][()]
            unknown = (sem < 0) | (sem >= self.num_channel)
            sem[unknown] = self.num_channel - 1
            
            if self.mode == 'test':
                image = self.input_transform(rgb)# get bgr image
                image = image.transpose((2, 0, 1))
                label = self.label_transform(sem)
                sample = {'rgb': rgb.copy(),
                        "image": image.copy(),
                        "label": label.copy()}

            elif self.mode == 'train':
              #  from .utils import d3_41_colors_rgb
                
                image, label = self.gen_sample(rgb, sem, \
                        self.multi_scale, self.flip)
                image = image[:, :, ::-1]
              #  cv2.imwrite("rgb.png", rgb)
              #  cv2.imwrite("sem.png", d3_41_colors_rgb[sem])
               # cv2.imwrite("label.png", d3_41_colors_rgb[label])

                rgb_ = image.copy()
                rgb_[0:1, ...] *= self.std[0]
                rgb_[1:2, ...] *= self.std[1]
                rgb_[2:3, ...] *= self.std[2]
                rgb_[0:1, ...] += self.mean[0]
                rgb_[1:2, ...] += self.mean[1]
                rgb_[2:3, ...] += self.mean[2]
                rgb_ *= 255.
                rgb_ = rgb_[:, :, ::-1] # rgb -> bgr
             #   cv2.imwrite('image.png', rgb_.transpose(1, 2, 0))
                sample = {"rgb": rgb_.copy(),
                    "image": image.copy(),
                    "label": label.copy()}
            return sample



# the dataset used to train segmentation model for ACNet
class SegmentationDataset(Dataset):
    # self.samples: record directory of each hdf5 file
    def __init__(self, root_dirs, num_channel, transform=None, limit=None):
        self.transform = transform
        self.samples = []
        self.num_channel = num_channel
        for root_dir in root_dirs:
            for sample in os.listdir(root_dir):
                if sample.endswith('.hdf5'):
                    file_dir = os.path.join(root_dir, sample)
                    try:
                        f = h5py.File(file_dir, 'r')
                        self.samples.append(file_dir)
                    except:
                        continue
        if limit is not None and int(len(self.samples) * 4 / limit) >= 1:
            ratio = int(len(self.samples)*4 / limit)
            self.samples = [sample for i, sample in enumerate(self.samples) if i % ratio == 0]
    # for each hdf5 file
    # there are four rotations thus four training samples
    def __len__(self):
        return 4*len(self.samples)

    def __getitem__(self, idx):
        fid = int(idx / 4) # find which sample
        rid = idx - fid * 4 # find which rotation
        with h5py.File(self.samples[fid], 'r') as f:
            rgb = f['rgb_imgs'][str(rid)][()]
            dep = f['dep_imgs'][str(rid)][()]
            sem = f['sem_imgs'][str(rid)][()]
           # cv2.imwrite('rgb.png', rgb)
          #  cv2.imwrite("dep.png", dep[..., 0]*25.5)
            unknown = (sem < 0) | (sem >= self.num_channel)
            sem[unknown] = self.num_channel - 1
          #  cv2.imwrite("sem.png", d3_41_colors_rgb[sem])
            sample = {'image': rgb[..., [2, 1, 0]], 'depth': dep[..., 0], 'label': sem}
            if self.transform:
                sample = self.transform(sample)
            
        return sample

class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        image_h, image_w = depth.shape
        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        image_h, image_w = depth.shape
        i = random.randint(0, image_h - self.th)
        j = random.randint(0, image_w - self.tw)

        return {'image': image[i:i + self.th, j:j + self.tw, :],
                'depth': depth[i:i + self.th, j:j + self.tw],
                'label': label[i:i + self.th, j:j + self.tw]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        image = image / 255
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}

if __name__ == '__main__':

    root_dirs \
    = ['/local/crv/yiqing/buffer/data_gen_1','/local/crv/yiqing/buffer/data_gen_2',
        '/local/crv/yiqing/buffer/data_gen_3']
    SData = HRNetDataset(root_dirs, num_channel=41, \
            multi_scale=True, flip=True,\
            base_size=2048, crop_size=(480, 640), downsample_rate=1,\
            scale_factor=16, \
            mode="train",\
            mean=[0.485, 0.456, 0.406], \
            std=[0.229, 0.224, 0.225])
    
    from utils import d3_41_colors_rgb

    SData = SegmentationDataset(root_dirs, 41, transform=transforms.Compose([scaleNorm(),
                                                                   RandomScale((1.0, 1.4)),
                                                                   RandomHSV((0.9, 1.1),
                                                                             (0.9, 1.1),
                                                                             (25, 25)),
                                                                   RandomCrop(480, 640),
                                                                   RandomFlip(),
                                                                   ToTensor(),
                                                                   Normalize()]),)
    
    print(len(SData))
    #for i in range(len(SData)):
    #    image, label = SData[6003]
    #    print(image.shape)
    #    print(label.shape)
       # print(sample['image'].shape)
      #  print(sample['depth'].shape)
      #  print(sample['label'].shape)
        #print(sample['sem_in'].shape)
        #print(sample['sem_gt'].shape)
        #cv2.imwrite('sem_in.png', d3_41_colors_rgb[sample['sem_in']])
       # cv2.imwrite('sem_gt.png', d3_41_colors_rgb[sample['sem_gt']])
    #    break
    
