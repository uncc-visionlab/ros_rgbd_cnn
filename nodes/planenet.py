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
#from ros_rgbd_cnn import RedNet_model_rgbplane
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
        self._model = RedNet_model.RedNet(pretrained=False)
        
        # Load weights trained on depth only or RGBD
        model_path = rospy.get_param('~model_path', PLANENET_MODEL_PATH)

        # Download trained weights from Releases if needed
        if model_path == PLANENET_MODEL_PATH and not os.path.exists(PLANENET_MODEL_PATH):
            utils.download_trained_weights('rgbd', PLANENET_MODEL_PATH)      # this util method needs to be modified later  

        load_ckpt(self._model, None, model_path, device)
        self._model.eval()
        self._model.to(device)

    
    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        self._segimg_pub = rospy.Publisher('segimg', Image, queue_size=1)
        self._planeimg_pub = rospy.Publisher('planeimg', Image, queue_size=1)
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
                plane_img = self._estimate_planes(msg, 3, 1)
                plane_img_msg = self._visualizePlaneImage(msg, plane_img)
                self._planeimg_pub.publish(plane_img_msg)
                #self._segment_image(msg)

            rate.sleep()
            
    def _visualizePlaneImage(self, msg, plane_img):
        vis_planes_img = np.zeros(shape=(plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8);
        for y in xrange(0, plane_img.shape[0], 1):
            for x in xrange(0, plane_img.shape[1], 1):
                if (~np.isnan(plane_img[y,x,2])):
                    vis_planes_img[y, x, :] = plane_img[y, x, 2]*255                
        image_msg = self._cv_bridge.cv2_to_imgmsg(vis_planes_img, encoding="bgr8")
        return image_msg
            
    def _depthImage2ptcloud(self, msg, winsize = 1, stride = 1):        
        depth_img = msg._depth
        # smooth the depth image using winsize here
        ptCloud = np.zeros(shape=(int(depth_img.shape[0]/stride), int(depth_img.shape[1]/stride), 3), dtype=np.float32);        
        for y in xrange(0, depth_img.shape[0], 1):
            for x in xrange(0, depth_img.shape[1], 1):
                depth = depth_img[y, x]
                if (~np.isnan(depth)):
                    #print(depth)
                    ptCloud[y, x, 0] = self._ifocal_length_x * (x - self._center_x) * depth
                    ptCloud[y, x, 1] = self._ifocal_length_y * (y - self._center_y) * depth
                ptCloud[y, x, 2] = depth
        return ptCloud

    def _estimate_planes(self, msg, winsize = 3, stride = 1):
        rgb_img = msg._rgb
        depth_img = msg._depth
        # size of window is 2*halfwinsize+1        
        halfwinsize = int(0.5*(winsize-1))
        #winsize = 2*halfwinsize + 1
        
        ptCloud = self._depthImage2ptcloud(msg, winsize, stride)        
        planes_img = np.zeros(shape=(int(depth_img.shape[0]/stride), int(depth_img.shape[1]/stride), 4), dtype=np.float32);
        windowDepths = np.empty(shape=(2*halfwinsize+1, 2*halfwinsize+1), dtype=np.float32)
        for y in xrange(halfwinsize, depth_img.shape[0]- halfwinsize, 1):
            for x in xrange(halfwinsize, depth_img.shape[1]- halfwinsize, 1):
                windowDepths = depth_img[(y - halfwinsize):(y + halfwinsize + 1), (x - halfwinsize):(x + halfwinsize + 1)]
                #print(windowDepths)
                numValidPoints = np.count_nonzero(~np.isnan(windowDepths))
                #print(numValidPoints)
                if (numValidPoints < 3):
                    plane3 = np.array([0,0,0,0]);
                else:                    
                    pts3D = np.empty(shape=(numValidPoints,3), dtype=np.float32);                    
                    offset = 0;
                    #print(pts3D)
                    for ywin in xrange(-halfwinsize, halfwinsize + 1):
                        for xwin in xrange(-halfwinsize, halfwinsize + 1):
                            if (~np.isnan(ptCloud[y + ywin, x + xwin, 2])):
                                pts3D[offset,:] = ptCloud[y + ywin, x + xwin, :]
                                offset += 1                                                      

                    #print("pts3D = " + str(pts3D))
                    plane3 = self._fitPlaneImplicitLeastSquares(pts3D)
                    
                planes_img[y,x,:] = plane3
        return planes_img
                    
                
    def _fitPlaneImplicitLeastSquares(self, points):
        from numpy import linalg as LA
        plane3 = np.empty(shape=(4), dtype=np.float32)
        _M = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)        
        centroid = np.mean(points, 0)
        demeaned_pts3D = points - centroid        
        _MtM = np.dot(demeaned_pts3D.transpose(), demeaned_pts3D)
        [w, v] = LA.linalg.eigh(_MtM)
        plane3[0:3] = v[:,0]
        plane3[3] = -np.dot(plane3[0:3],centroid[:])
        if (plane3[2] > 0):
            plane3 = -plane3
        #print(plane3)
        return plane3
    
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
