import numpy as np
from torch import nn
import torch
import os
import sys
from six.moves.urllib import request
import shutil
import contextlib
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imageio


med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]

'''
def download_trained_weights(model_name, model_path, verbose=1):
    """
    Download trained weights from previous training on depth images or rgbd images.
    """
    # depth_model_path: local path of depth trained weights
    
    if verbose > 0:
        print("Downloading pretrained model to " + model_path + " ...")
    if model_name == 'depth':
        with contextlib.closing(request.urlopen(DEPTH_TRAINED_MODEL)) as resp, open(model_path, 'wb') as out:
            shutil.copyfileobj(resp, out)
    elif model_name == 'rgbd':
        with contextlib.closing(request.urlopen(REDNET_PRETRAINED_MODEL)) as resp, open(model_path, 'wb') as out:
            shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")
'''         

'''
#An example of using depth2plane:
fittingSize = 2
import imageio
from utils.utils import depth2plane
depth = imageio.imread('./data/SUNRGBD/kv1/NYUdata/NYU0002/depth_bfx/NYU0002.png')
plane = depth2plane(depth, extrinsic, intrinsic, fittingSize)
planeImage = plane.getPlaneImage()
plane.visualizePlaneImage(planeImage)
'''


class depth2plane:
    def __init__(self, depth, extrinsic, intrinsic, fittingSize=5):
        self.depthImage = depth
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.fittingSize = fittingSize

    def getPlaneImage(self):
        planeImage = self.estimate_planes()
        return planeImage

    def matrix_from_txt(self, file):
        f = open(file)
        l = []
        for line in f.readlines():
            line = line.strip('\n')
            for j in range(len(list(line.split()))):
                l.append(line.split()[j])
        matrix = np.array(l, dtype=np.float32)
        return matrix

    def getCameraInfo(self):
        K = self.intrinsic
        ifocal_length_x = 1.0/K[0]
        ifocal_length_y = 1.0/K[4]
        center_x = K[2]
        center_y = K[5]
        camera_pose = self.extrinsic
        camera_pose = camera_pose.reshape(3, 4)
        Rtilt = camera_pose[0:3, 0:3]
        #A = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float32)
        #A = A.reshape(3, 3)
        #B = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0], dtype=np.float32)
        #B = B.reshape(3, 3)
        #Rtilt = A*Rtilt*B
        #Rtilt = np.matmul(np.matmul(A, Rtilt), B)
        return ifocal_length_x, ifocal_length_y, center_x, center_y, Rtilt

    def bitShiftDepthMap(self, depthImage):
        depthVisData = np.asarray(depthImage, np.uint16)
        depthInpaint = np.bitwise_or(np.right_shift(depthVisData, 3), np.left_shift(depthVisData, 16 - 3))
        depthInpaint = depthInpaint.astype(np.single) / 1000
        depthInpaint[depthInpaint > 8] = 8
        return depthInpaint

    def depthImage2ptcloud(self, depth_img):
        [ifocal_length_x, ifocal_length_y, center_x, center_y, Rtilt] = self.getCameraInfo()
        ptCloud = np.zeros(shape=(int(depth_img.shape[0]), int(depth_img.shape[1]), 3), dtype=np.float32)        
        for y in xrange(0, depth_img.shape[0]):
            for x in xrange(0, depth_img.shape[1]):
                depth = depth_img[y, x]
                if (~np.isnan(depth)):
                    #print(depth)
                    ptCloud[y, x, 0] = ifocal_length_x * (x - center_x) * depth
                    ptCloud[y, x, 1] = ifocal_length_y * (y - center_y) * depth
                ptCloud[y, x, 2] = depth
        points3d = ptCloud.reshape(-1, 3)   # [height*width, 3]
        ptCloud = (np.matmul(Rtilt, points3d.T)).T
        ptCloud.astype(np.float32)
        ptCloud = ptCloud.reshape(depth_img.shape[0], depth_img.shape[1], 3)   # [height, width, 3]
        return ptCloud

    def estimate_planes(self):
        winsize = self.fittingSize
        depthImage = self.depthImage
        depthImage = self.smoothing(depthImage, filtersize=3)  
        #depth_img = self.bitShiftDepthMap(depthImage) # no need to shift in real scenerio
        depth_img = depthImage 
        ptCloud = self.depthImage2ptcloud(depth_img)
        #ptCloud = self.smoothing(ptCloud, filtersize=5)  #  smoothing the inpainted depth image will cause inconvergence in fitting
        planes_img = np.zeros(shape=(int(depth_img.shape[0] / winsize), int(depth_img.shape[1] / winsize), 4), dtype=np.float32)
        for y in range(0, depth_img.shape[0]-winsize, winsize):
            for x in range(0, depth_img.shape[1]-winsize, winsize):
                windowDepths = depth_img[y:(y + winsize + 1), x:(x + winsize + 1)]
                # print(windowDepths)
                numValidPoints = np.count_nonzero(~np.isnan(windowDepths))
                # print(numValidPoints)
                if (numValidPoints < 3):
                    plane3 = np.array([0, 0, 0, 0])
                else:
                    pts3D = np.empty(shape=(numValidPoints, 3), dtype=np.float32)
                    offset = 0
                    # print(pts3D)
                    for ywin in range(0, winsize + 1):
                        for xwin in range(0, winsize + 1):
                            if (~np.isnan(ptCloud[y + ywin, x + xwin, 2])):
                                pts3D[offset, :] = ptCloud[y + ywin, x + xwin, :]
                                offset += 1
                    plane3 = self.fitPlaneImplicitLeastSquares(pts3D)
                planes_img[int(y/winsize), int(x/winsize), :] = plane3
        return planes_img

    def fitPlaneImplicitLeastSquares(self, points):
        from numpy import linalg as LA
        plane3 = np.empty(shape=(4), dtype=np.float32)
        centroid = np.mean(points, 0)
        demeaned_pts3D = points - centroid
        _MtM = np.matmul(demeaned_pts3D.transpose(), demeaned_pts3D)
        [_, v] = LA.linalg.eigh(_MtM)
        plane3[0:3] = v[:, 0]
        plane3[3] = -np.dot(plane3[0:3], centroid[:])
        if (abs(plane3[3]) > abs(plane3[2])):  # use d as criterion
            if (plane3[3] < 0):
                plane3 = -plane3
        elif (plane3[2] > 0):   # use c as criterion
                plane3 = -plane3
        return plane3

    def smoothing(self, img, filtersize):
        img = cv2.GaussianBlur(img, (filtersize, filtersize), cv2.BORDER_CONSTANT)
        return img

    def visualizePlaneImage(self, plane_img):
        #imageio.imwrite('0002.tiff', plane_img)
        #cv2.namedWindow("planeImage", cv2.WINDOW_NORMAL)
        vis_plane_img_c = np.zeros(shape=(plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8)
        for y in range(0, plane_img.shape[0]):
            for x in range(0, plane_img.shape[1]):
                vis_plane_img_c[y, x, :] = plane_img[y, x, 2] * 255
        cv2.imshow("c channel", vis_plane_img_c)
        cv2.imshow("abs_planeImage", abs(plane_img[:, :, 0:3]))
        cv2.imshow("ori_planeImage", plane_img[:, :, 0:3])
        cv2.waitKey()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           reduction='none')

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 1e-3  # should be larger than 0 but tensor seems to treat 0 as a very small number like 2e-9 then
            targets_m = targets.clone()  # when we subtract 1 from 2e-9 it gives us -1 which causes an IndexError: target -1 out of bounds
            targets_m[mask] -= 1  # map the label from [1, 37] to [0, 36]
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)
