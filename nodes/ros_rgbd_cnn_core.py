#!/usr/bin/env python2
#encoding: UTF-8
import os
import threading
import numpy as np

import cv2
from cv_bridge import CvBridge
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import UInt8MultiArray

import argparse
import torch
import imageio
import skimage.transform
import torchvision
import torch.optim

import sys
sys.path.append('../src/')
from ros_rgbd_cnn import model_rgbd
from ros_rgbd_cnn import utils
from ros_rgbd_cnn.utils import load_ckpt

import skimage.io
import glob

class rgbd_msg(object):
    def __init__(self):
        self._header = None;
        self._rgb = None;
        self._depth = None;

class RGBD_CNN_Core(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)     

        print("Waiting for CameraInfo message ....")
        camera_info_msg = rospy.wait_for_message('~depth/camera_info', CameraInfo, timeout=None)
        self.intrinsic = np.array(camera_info_msg.K)
        self.extrinsic = np.array(camera_info_msg.P)
        self._ifocal_length_x = 1.0/camera_info_msg.K[0]
        self._ifocal_length_y = 1.0/camera_info_msg.K[4]
        self._center_x = camera_info_msg.K[2]
        self._center_y = camera_info_msg.K[5]
        print("Received CameraInfo message.")

    def _sync_callback(self, ros_rgb_img, ros_depth_img):
        rospy.logdebug("Got a synchronized RGB & depth image")
        if self._msg_lock.acquire(False):
            #print("Got a synchronized RGB & depth image")
            self._last_msg = rgbd_msg()
            self._last_msg._header = ros_rgb_img.header
            self._last_msg._rgb = self._cv_bridge.imgmsg_to_cv2(ros_rgb_img, 'bgr8')  # 32FC1 for depth images, bgr8 for color images
            self._last_msg._depth = self._cv_bridge.imgmsg_to_cv2(ros_depth_img, '32FC1')  # 32FC1 for depth images, bgr8 for color images
            self._msg_lock.release()

