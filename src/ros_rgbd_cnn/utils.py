import numpy as np
from torch import nn
import torch
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import sys
from six.moves.urllib import request
import shutil
import contextlib


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

DEPTH_TRAINED_MODEL = "https://drive.google.com/uc?export=download&id=1ezJOTP8KGSbj8M3Fdldws1-XWAMK9pr5"
REDNET_PRETRAINED_MODEL = "https://drive.google.com/uc?export=download&id=18E0hAYEvCPAIPGnwXpqi2Im-wAdGPN7_"

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
#An example of using depth2plane:
intrinsicPath = '$dataPath$/data/SUNRGBD/kv1/NYUdata/NYU0002/intrinsics.txt'
fittingSize = 2
extrinsicPath = '$dataPath$/data/SUNRGBD/kv1/NYUdata/NYU0002/extrinsics/20150118235913.txt'
import imageio
from utils import depth2plane
depth = imageio.imread('$dataPath$/data/SUNRGBD/kv1/NYUdata/NYU0002/depth_bfx/NYU0002.png')
#labelPath = '$dataPath$/data/SUNRGBD/kv1/NYUdata/NYU0002/label/label.npy'
plane = depth2plane(depth, extrinsicPath, intrinsicPath, fittingSize)
newLabel = plane.getPlaneLabel()
planeImage = plane.getPlaneImage()
#plane.visualizePlaneImage(planeImage)
#rgbPath = '$dataPath$/data/SUNRGBD/kv1/NYUdata/NYU0002/image/NYU0002.jpg'
#plane.visualizePointCloud(rgbPath)
'''


class depth2plane:
    def __init__(self, depth, extrinsicPath, intrinsicPath, fittingSize=5):  # add labelPath as one argument if needed
        self.depthImage = depth
        self.extrinsicPath = extrinsicPath
        self.intrinsicPath = intrinsicPath
        self.fittingSize = fittingSize
        #self.labelPath = labelPath

    def getPlaneImage(self):
        planeImage = self.estimate_planes()
        return planeImage

    def getPlaneLabel(self):
        label = np.load(self.labelPath)
        winsize = self.fittingSize
        #halfwinsize = int(0.5 * winsize)
        newLabel = np.zeros(shape=(int(label.shape[0] / winsize), int(label.shape[1] / winsize)))
        #for y in range(halfwinsize, label.shape[0] - halfwinsize, winsize):
        for y in range(0, label.shape[0]-winsize, winsize):
            for x in range(0, label.shape[1]-winsize, winsize):
                windowLabels = label[y:(y + winsize + 1), x:(x + winsize + 1)]

                newLabel[int(y / winsize), int(x / winsize)] = np.max(windowLabels)
        return newLabel

    def matrix_from_txt(self, file):
        contents = open(file).read()
        matrix = [item.split() for item in contents.split('\n')[:-1]]
        matrix = np.array(matrix, dtype=np.float32)
        return matrix

    def getCameraInfo(self):
        contents = open(self.intrinsicPath).read()
        K = [item.split() for item in contents.split(' ')[:-1]]
        K = np.array(K, dtype=np.float32)
        ifocal_length_x = 1.0/K[0]
        ifocal_length_y = 1.0/K[4]
        center_x = K[2]
        center_y = K[5]
        camera_pose = self.matrix_from_txt(self.extrinsicPath)
        Rtilt = camera_pose[0:3, 0:3]
        A = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float32)
        A = A.reshape(3, 3)
        B = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0], dtype=np.float32)
        B = B.reshape(3, 3)
        #Rtilt = A*Rtilt*B
        Rtilt = np.matmul(np.matmul(A, Rtilt), B)
        return ifocal_length_x, ifocal_length_y, center_x, center_y, Rtilt

    def bitShiftDepthMap(self, depthImage):
        depthVisData = np.asarray(depthImage, np.uint16)
        depthInpaint = np.bitwise_or(np.right_shift(depthVisData, 3), np.left_shift(depthVisData, 16 - 3))
        depthInpaint = depthInpaint.astype(np.single) / 1000
        depthInpaint[depthInpaint > 8] = 8
        return depthInpaint

    def depthImage2ptcloud(self, depth_img):
        [ifocal_length_x, ifocal_length_y, center_x, center_y, Rtilt] = self.getCameraInfo()
        '''
        ptCloud = np.zeros(shape=(int(depth_img.shape[0]), int(depth_img.shape[1]), 3), dtype=np.float32)
        for y in range(0, depth_img.shape[0]):
            for x in range(0, depth_img.shape[1]):
                depth = depth_img[y, x]
                if (~np.isnan(depth)):
                    # print(depth)
                    ptCloud[y, x, 0] = ifocal_length_x * (x - center_x) * depth
                    ptCloud[y, x, 1] = ifocal_length_y * (y - center_y) * depth
                ptCloud[y, x, 2] = depth
        '''
        #invalid = depth_img == 0
        x, y = np.meshgrid(np.arange(depth_img.shape[1], dtype=np.float32), np.arange(depth_img.shape[0], dtype=np.float32))
        xw = (x - center_x) * depth_img * ifocal_length_x
        yw = (y - center_y) * depth_img * ifocal_length_y
        zw = depth_img
        points3dMatrix = np.stack((xw, zw, -yw), axis=2)
        #points3dMatrix[np.stack((invalid, invalid, invalid), axis=2)] = np.nan  # no zero-depth pixels when using depth_bfx
        points3d = points3dMatrix.reshape(-1, 3)   # [height*width, 3]
        ptCloud = (np.matmul(Rtilt, points3d.T)).T
        ptCloud.astype(np.float32)
        ptCloud = ptCloud.reshape(depth_img.shape[0], depth_img.shape[1], 3)   # [height, width, 3]
        return ptCloud, points3d

    def estimate_planes(self):
        winsize = self.fittingSize
        depthImage = self.depthImage
        #depthImage = self.smoothing(depthImage, filtersize=3)  # smoothing the inpainted depth image will cause bad fitting result
        depth_img = self.bitShiftDepthMap(depthImage)
        # size of window is 2*halfwinsize+1
        [ptCloud, _] = self.depthImage2ptcloud(depth_img)
        #ptCloud = self.smoothing(ptCloud, filtersize=5)
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
        #_MtM = np.dot(demeaned_pts3D.transpose(), demeaned_pts3D)
        _MtM = np.matmul(demeaned_pts3D.transpose(), demeaned_pts3D)
        [_, v] = LA.linalg.eigh(_MtM)
        plane3[0:3] = v[:, 0]
        plane3[3] = -np.dot(plane3[0:3], centroid[:])
        if (plane3[2] > 0):
            plane3 = -plane3
        return plane3

    def smoothing(self, img, filtersize=3):
        img = cv2.GaussianBlur(img, (filtersize, filtersize), cv2.BORDER_CONSTANT)
        return img

    def visualizePlaneImage(self, plane_img):
        vis_plane_img_a = np.zeros(shape=(plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8)
        vis_plane_img_b = np.zeros(shape=(plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8)
        vis_plane_img_c = np.zeros(shape=(plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8)
        vis_planes_img_d = np.zeros(shape=(plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8)
        for y in range(0, plane_img.shape[0]):
            for x in range(0, plane_img.shape[1]):
                vis_plane_img_c[y, x, :] = plane_img[y, x, 2] * 255
        cv2.imshow("c", vis_plane_img_c)
        cv2.imshow("planeImage", abs(plane_img[:,:,0:3]))
        cv2.imshow("abs_planeImage", plane_img[:,:, 0:3])
        cv2.waitKey()

    def visualizePointCloud(self, rgbpath):
        depthImage = self.depthImage
        depth_img = self.bitShiftDepthMap(depthImage)
        [_, points3d] = self.depthImage2ptcloud(depth_img)
        im = np.asarray(Image.open(rgbpath))
        rgb = im.astype(np.double) / np.iinfo(im.dtype).max
        rgb = rgb.reshape(-1, 3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c=rgb)
        plt.show()


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
