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
from ros_rgbd_cnn import model_rgbd
from ros_rgbd_cnn import utils
from ros_rgbd_cnn.utils import load_ckpt

import skimage.io
import glob

from ros_rgbd_cnn_core import RGBD_CNN_Core
from ros_rgbd_cnn.msg import Result

CLASS_NAMES =  ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
    'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes', 'ceiling', 'books',
    'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
    'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
device = torch.device("cpu")
if device.type == 'cuda':
    print('Using '+ torch.cuda.get_device_name(0))

RGBD_PRETRAINED_MODEL_PATH = ""
image_w = 640
image_h = 480

class RGBDNet(RGBD_CNN_Core):
    def __init__(self):
        RGBD_CNN_Core.__init__(self)
        self._model = model_rgbd.model(pretrained=False)

        # Load weights trained on RGBD
        model_path = rospy.get_param('~model_path', RGBD_PRETRAINED_MODEL_PATH)

        # Download trained weights from Releases if needed (method not working)
        #if model_path == RGBD_PRETRAINED_MODEL_PATH and not os.path.exists(RGBD_PRETRAINED_MODEL_PATH):
        #    utils.download_trained_weights('rgbd', RGBD_PRETRAINED_MODEL_PATH)        

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
                self._segment_image(msg)

            rate.sleep()

    def _segment_image(self, msg):
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
        depth = skimage.transform.resize(depth_img, (image_h, image_w), order=0,
                                     mode='reflect', preserve_range=True)
        image = image / 255
        image = torch.from_numpy(image).float()
        depth = torch.from_numpy(depth).float()
        image = image.permute(2, 0, 1)
        depth = torch.unsqueeze(depth, 0)

        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        depth = torchvision.transforms.Normalize(mean=[19050],
                                                 std=[9650])(depth)

        image = image.to(device).unsqueeze_(0)
        depth = depth.to(device).unsqueeze_(0)
        #print("running inference")
        result = self._model(image, depth)

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
