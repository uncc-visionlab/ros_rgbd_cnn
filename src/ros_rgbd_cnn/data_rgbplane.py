import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
from ros_rgbd_cnn.utils import depth2plane
#from train_rgbplane import image_h, image_w, fittingSize
from test_rgbplane import image_h, image_w, fittingSize

img_dir_train_file = './data/img_dir_train.txt'
depth_dir_train_file = './data/depth_dir_train.txt'
label_dir_train_file = './data/label_train.txt'
ext_dir_train_file = './data/ext_dir_train.txt'
int_dir_train_file = './data/int_dir_train.txt'
plane_dir_train_file = './data/plane_dir_train.txt'
plane_label_dir_train_file = './data/plane_label_train.txt'

img_dir_test_file = './data/img_dir_test.txt'
depth_dir_test_file = './data/depth_dir_test.txt'
label_dir_test_file = './data/label_test.txt'
ext_dir_test_file = './data/ext_dir_test.txt'
int_dir_test_file = './data/int_dir_test.txt'
plane_dir_test_file = './data/plane_dir_test.txt'
plane_label_dir_test_file = './data/plane_label_test.txt'

class SUNRGBD(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):

        self.phase_train = phase_train
        self.transform = transform

        with open(img_dir_train_file, 'r') as f:
            self.img_dir_train = f.read().splitlines()
        with open(depth_dir_train_file, 'r') as f:
            self.depth_dir_train = f.read().splitlines()
        with open(label_dir_train_file, 'r') as f:
            self.label_dir_train = f.read().splitlines()
        with open(ext_dir_train_file, 'r') as f:
            self.ext_dir_train = f.read().splitlines()
        with open(int_dir_train_file, 'r') as f:
            self.int_dir_train = f.read().splitlines()
        with open(plane_dir_train_file, 'r') as f:
            self.plane_dir_train = f.read().splitlines()
        with open(plane_label_dir_train_file, 'r') as f:
            self.plane_label_dir_train = f.read().splitlines()
        with open(img_dir_test_file, 'r') as f:
            self.img_dir_test = f.read().splitlines()
        with open(depth_dir_test_file, 'r') as f:
            self.depth_dir_test = f.read().splitlines()
        with open(label_dir_test_file, 'r') as f:
            self.label_dir_test = f.read().splitlines()
        with open(ext_dir_test_file, 'r') as f:
            self.ext_dir_test = f.read().splitlines()
        with open(int_dir_test_file, 'r') as f:
            self.int_dir_test = f.read().splitlines()
        with open(plane_dir_test_file, 'r') as f:
            self.plane_dir_test = f.read().splitlines()
        with open(plane_label_dir_test_file, 'r') as f:
            self.plane_label_dir_test = f.read().splitlines()

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            # rgb here
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            ext_dir = self.ext_dir_train
            int_dir = self.int_dir_train
            plane_dir = self.plane_dir_train
            plane_label_dir = self.plane_label_dir_train
        else:
            # rgb here
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test
            ext_dir = self.ext_dir_test
            int_dir = self.int_dir_test
            plane_dir = self.plane_dir_test
            plane_label_dir = self.plane_label_dir_test
        '''
        #label = np.load(label_dir[idx])
        labelPath = label_dir[idx]
        depth = imageio.imread(depth_dir[idx])
        extrinsicPath = ext_dir[idx]
        intrinsicPath = int_dir[idx]
        plane = depth2plane(depth, extrinsicPath, intrinsicPath, labelPath, fittingSize)
        planelabel = plane.getPlaneLabel()
        plane = plane.getPlaneImage()
        '''
        # rgb here
        image = imageio.imread(img_dir[idx])
        planelabel = np.load(plane_label_dir[idx])
        plane = imageio.imread(plane_dir[idx])
        plane = plane[:,:,[1,2,3]]  # channel b, c, d
        sample = {'image': image, 'plane': plane, 'planelabel': planelabel}

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

        return {'image': img_new, 'plane': sample['plane'], 'planelabel': sample['planelabel']}


class scaleNorm(object):
    def __call__(self, sample):
        # rgb here
        image, plane, planelabel = sample['image'], sample['plane'], sample['planelabel']

        # Bi-linear
        # rgb here
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        plane = skimage.transform.resize(plane, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        planelabel = skimage.transform.resize(planelabel, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'plane': plane, 'planelabel': planelabel}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        # rgb here
        image, plane, planelabel = sample['image'], sample['plane'], sample['planelabel']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * plane.shape[0]))
        target_width = int(round(target_scale * plane.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        plane = skimage.transform.resize(plane, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        planelabel = skimage.transform.resize(planelabel, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'plane': plane, 'planelabel': planelabel}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, plane, planelabel = sample['image'], sample['plane'], sample['planelabel']
        h = plane.shape[0]
        w = plane.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'plane': plane[i:i + image_h, j:j + image_w],
                'planelabel': planelabel[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, plane, planelabel = sample['image'], sample['plane'], sample['planelabel']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            plane = np.fliplr(plane).copy()
            planelabel = np.fliplr(planelabel).copy()

        return {'image': image, 'plane': plane, 'planelabel': planelabel}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, plane = sample['image'], sample['plane']
        image = image / 255
        #plane[:, :, 3] *= 0.15
        #plane *= 10000
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        #plane = torchvision.transforms.Normalize(mean=[0, 0, 0, 19050], std=[5000, 5000, 5000, 9650])(plane)  # need to recalculate the max and min over the dataset
        plane = torchvision.transforms.Normalize(mean=[0, 0, 5], std=[1, 1, 10])(plane)     # only 3 channel for training
        sample['image'] = image
        sample['plane'] = plane

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, plane, planelabel = sample['image'], sample['plane'], sample['planelabel']

        # Generate different label scales
        planelabel2 = skimage.transform.resize(planelabel, (planelabel.shape[0] // 2, planelabel.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        planelabel3 = skimage.transform.resize(planelabel, (planelabel.shape[0] // 4, planelabel.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        planelabel4 = skimage.transform.resize(planelabel, (planelabel.shape[0] // 8, planelabel.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        planelabel5 = skimage.transform.resize(planelabel, (planelabel.shape[0] // 16, planelabel.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        plane = plane.transpose((2, 0, 1))
        #plane = np.expand_dims(plane, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'plane': torch.from_numpy(plane).float(),
                'label': torch.from_numpy(planelabel).float(),
                'label2': torch.from_numpy(planelabel2).float(),
                'label3': torch.from_numpy(planelabel3).float(),
                'label4': torch.from_numpy(planelabel4).float(),
                'label5': torch.from_numpy(planelabel5).float()}
