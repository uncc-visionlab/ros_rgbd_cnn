#!/usr/bin/env python
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
from planenet import PlaneNet
from rgbdnet import RGBDNet

def main():
    rospy.init_node('ros_rgbd_cnn_node')

    algorithm = rospy.get_param('~algorithm', 'rgbdnet')     

    if (algorithm == 'rgbdnet'):
        node = RGBDNet()
    elif (algorithm == 'planenet'):
        node = PlaneNet()

    node.run()

if __name__ == '__main__':
    main()
