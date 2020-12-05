#!/usr/bin/env python2
#encoding: UTF-8
import os
import threading
import numpy as np

import cv2
from cv_bridge import CvBridge
import rospy
import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray

import argparse
import torch
import imageio
import skimage.transform
import torchvision
import torch.optim

import sys
sys.path.append('../src/')
from ros_rgbd_cnn import RedNet_model_depth
from ros_rgbd_cnn import RedNet_model
from ros_rgbd_cnn import RedNet_model_rgbplane
from ros_rgbd_cnn import utils
from ros_rgbd_cnn.utils import load_ckpt

import skimage.io
import glob

from ros_rgbd_cnn_core import RGBD_CNN_Core
from ros_rgbd_cnn.msg import Result

ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))
#DEPTH_MODEL_PATH = os.path.join(ROS_HOME, 'ckpt_epoch_depth_640*480_150.00.pth')
PLANENET_MODEL_PATH = os.path.join(ROS_HOME, 'ckpt_epoch_10.00_rgbdplane.pth')

CLASS_NAMES =  ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
    'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books',
    'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
    'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

# Local path to trained weights file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
device = torch.device("cpu")
if device.type == 'cuda':
    print('Using '+ torch.cuda.get_device_name(0))

image_w = 320
image_h = 256

class PlaneNet(RGBD_CNN_Core):
    def __init__(self):
        RGBD_CNN_Core.__init__(self)
        self._plane_img = None
        self._ifocal_len = 1.0/530.0
        self._center_x = 320
        self._center_y = 240
        self._model = RedNet_model.RedNet(pretrained=False)

        # Load weights trained on depth only or RGBD
        model_path = rospy.get_param('~rgbplane_model_path', PLANENET_MODEL_PATH)

        # Download trained weights from Releases if needed
        if model_path == PLANENET_MODEL_PATH and not os.path.exists(PLANENET_MODEL_PATH):
            utils.download_trained_weights('rgbd', PLANENET_MODEL_PATH)      # this util method needs to be modified later  

        load_ckpt(self._model, None, model_path, device)
        self._model.eval()
        self._model.to(device)

    
    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        self._segimg_pub = rospy.Publisher('segimg', Image, queue_size=1)
        rgb_sub = message_filters.Subscriber('~rgb', Image, queue_size=1)
        depth_sub = message_filters.Subscriber('~depth', Image, queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self._sync_callback)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                self._estimate_planes(msg)
                self._segment_image(msg)

            rate.sleep()

    def _estimate_planes(self, msg):
        rgb_img = msg._rgb
        depth_img = msg._depth
        # size of window is 2*halfwinsize+1
        halfwinsize = 1;
        Y = np.empty(shape=(2*halfwinsize+1, 2*halfwinsize+1), dtype=np.float32)
        for x in xrange(halfwinsize,depth_img.shape[0]-halfwinsize,1):
            if (x % 10 == 0):
                print x
            for y in xrange(halfwinsize,depth_img.shape[0]-halfwinsize,1):
                Y = depth_img[np.ix_([y - halfwinsize, y + halfwinsize], [x-halfwinsize, x+halfwinsize])]
                Y = Y.flatten()
                for p in xrange(len(Y)):
                    x3d = self._ifocal_len * (Y[p] - self._center_x)
                    y3d = self._ifocal_len * (Y[p] - self._center_y)
                    z3d = Y[p];
                    
                a=1                
    def fitPlaneImplicitLeastSquares(self, points):
        from numpy import linalg as LA
        plane3 = np.empty(shape=(4,1), dtype=np.float32)

        _M = np.zeros(shape=(num_points, 3), dtype=float32)        
        centroid = np.zeros(shape(3,1), dtype=float32)
        for ptIdx in xrange(num_points):
            centroid += points;
        centroid /= num_points;
        #for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
        #    size_t pt_begin = stride*ptIdx;
        #    M[3 * ptIdx] = points[pt_begin] - centroid.x;
        #    M[3 * ptIdx + 1] = points[pt_begin + 1] - centroid.y;
        #    M[3 * ptIdx + 2] = points[pt_begin + 2] - centroid.z;
        _MtM = np.zeros(shape=(4,4), dtype=float32)        
        _MtM = np.dot(_M.transpose(), _M)
        [w, v] = LA.linalg.eigh(_MtM)
        # v[:, 0]
        #cv::Mat _planeCoeffs = eigVecs.row(2).t();

        #plane3.x = _planeCoeffs.at<scalar_t>(0);
        #plane3.y = _planeCoeffs.at<scalar_t>(1);
        #plane3.z = _planeCoeffs.at<scalar_t>(2);
        #plane3.d = -(plane3.x * centroid.x + plane3.y * centroid.y + plane3.z * centroid.z);
        #plane3.scale((plane3.z > 0) ? -1.0 : 1.0);

        #return plane3;

                
    
    def _segment_image(self, msg):
        return
        rgb_img = msg._rgb
        depth_img = msg._depth
        depth_img = np.nan_to_num(depth_img)
        #depth_img = depth_img*10000
        #print("rgb_(min,max) 0: " + str(rgb_img[..., 0].min()) + "," + str(rgb_img[..., 0].max()))
        #print("rgb_(min,max) 1: " + str(rgb_img[..., 1].min()) + "," + str(rgb_img[..., 1].max()))
        #print("rgb_(min,max) 2: " + str(rgb_img[..., 2].min()) + "," + str(rgb_img[..., 2].max()))
        #print("depth_(min,max): " + str(depth_img[...].min()) + "," + str(depth_img[...].max()))
        # Run detection
        # Bi-linear
        image = skimage.transform.resize(rgb_img, (image_h, image_w), order=1,
                                     mode='reflect', preserve_range=True)
        #depth = skimage.transform.resize(depth_img, (image_h, image_w), order=0,
        #                             mode='reflect', preserve_range=True)
        
        #plane goes here. Need to be resized to (image_h, image_w)
        plane = plane                        
        image = image / 255
        image = torch.from_numpy(image).float()
        #depth = torch.from_numpy(depth).float()
        plane = torch.from_numpy(depth).float()
        image = image.permute(2, 0, 1)
        plane = plane.permute(2, 0, 1)
        #depth = torch.unsqueeze(depth, 0)
        
        plane *= 10000
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        #depth = torchvision.transforms.Normalize(mean=[19050],
        #                                         std=[9650])(depth)
        plane = torchvision.transforms.Normalize(mean=[0, 0, 0, 19050],
                                                 std=[5000, 5000, 5000, 9650])(plane)

        image = image.to(device).unsqueeze_(0)
        #depth = depth.to(device).unsqueeze_(0)
        plane = plane.to(device).unsqueeze_(0)
        result = self._model(image, plane)

        color_label = utils.color_label(torch.max(result, 1)[1] + 1)[0]
        img = color_label.cpu().numpy().transpose((1, 2, 0))
        img = img.astype(np.uint8)
        img_msg = self._cv_bridge.cv2_to_imgmsg(img, 'bgr8')
        self._segimg_pub.publish(img_msg)

        result_msg = self._build_result_msg(msg, result)
        self._result_pub.publish(result_msg)

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg._header
        
        label_tensor = torch.max(result, 1)[1] + 1
        label_tensor = label_tensor.int()
        label = label_tensor.cpu().data.numpy()
        result_msg.labels.append(label)

        color_label = utils.color_label(label_tensor)[0]
        img = color_label.cpu().numpy().transpose(1, 2, 0)
        img = img.astype(np.uint8)
        result_image = Image()
        result_image.header = result_msg.header
        result_image.height = img.shape[0]
        result_image.width = img.shape[1]
        result_image.encoding = "bgr8"
        result_image.is_bigendian = False
        result_image.step = result_image.width
        result_image.data = img.tobytes()
        result_msg.image.append(result_image)

        return result_msg
    
if __name__ == '__main__':
    main()
