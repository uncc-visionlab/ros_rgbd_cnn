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
import math

class rgbdFastFitting:    
    def __init__(self, intrinsic, extrinsic, rows = 480, cols = 640, winsize_x = 5, winsize_y = 5):
        self.winsize_x = winsize_x
        self.winsize_y = winsize_y
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        ifocal_length_x = 1.0/intrinsic[0]
        ifocal_length_y = 1.0/intrinsic[4]
        center_x = intrinsic[2]
        center_y = intrinsic[5]
        camera_pose = self.extrinsic
        camera_pose = camera_pose.reshape(3, 4)
        self.initializeFastFittingIntegralImages(center_x, center_y, ifocal_length_x, ifocal_length_y, rows, cols)
        self.precomputeFastFittingMatrixElements()


    def isFastFittingMatricesInitialized(self):
        return ((self.sum_tanX is not None) and (self.fastFittingMatrixValues is not None))
    
    def initializeFastFittingIntegralImages(self, center_x, center_y, ifocal_length_x, ifocal_length_y, rows, cols):
        self.tanX = np.zeros((rows, cols), np.float64)
        self.tanY = np.zeros((rows, cols), np.float64)
        self.tanX2 = np.zeros((rows, cols), np.float64)
        self.tanY2 = np.zeros((rows, cols), np.float64)
        self.tanXtanY = np.zeros((rows, cols), np.float64)
        for y in xrange(0, rows):
            for x in xrange(0, cols):
                self.tanX[y,x] = (x - center_x) * ifocal_length_x
                self.tanY[y,x] = (y - center_y) * ifocal_length_y
                self.tanX2[y,x] = self.tanX[y,x] * self.tanX[y,x]
                self.tanY2[y,x] = self.tanY[y,x] * self.tanY[y,x]
                self.tanXtanY[y,x] = self.tanX[y,x] * self.tanY[y,x]
        self.sum_tanX = np.zeros((rows,cols), np.float64) 
        self.sum_tanY = np.zeros((rows,cols), np.float64) 
        self.sum_tanX2 = np.zeros((rows,cols), np.float64) 
        self.sum_tanY2 = np.zeros((rows,cols), np.float64) 
        self.sum_tanXtanY = np.zeros((rows,cols), np.float64) 
        
        self.idepth = np.zeros((rows,cols), np.float64)
        self.idepth2 = np.zeros((rows,cols), np.float64)
        self.tanX_idepth = np.zeros((rows,cols), np.float64)
        self.tanY_idepth = np.zeros((rows,cols), np.float64)
        self.sum_idepth = np.zeros((rows,cols), np.float64)
        self.sum_idepth2 = np.zeros((rows,cols), np.float64)
        self.sum_tanX_idepth = np.zeros((rows,cols), np.float64)
        self.sum_tanY_idepth = np.zeros((rows,cols), np.float64)
        
        cv2.integral(self.tanX, self.sum_tanX)
        cv2.integral(self.tanY, self.sum_tanY)
        cv2.integral(self.tanX2, self.sum_tanX2)
        cv2.integral(self.tanY2, self.sum_tanY2)
        cv2.integral(self.tanXtanY, self.sum_tanXtanY)
        
    def precomputeFastFittingMatrixElements(self):
        half_winsize_x = np.floor(self.winsize_x/2)
        half_winsize_y = np.floor(self.winsize_y/2)
        rows = self.sum_tanX.shape[0]
        cols = self.sum_tanX.shape[1]
        self.fastFittingMatrixValues = np.zeros((5,rows*cols), np.float32)
        for y in xrange(0, rows):
            for x in xrange(0, cols):
                offset = cols*y + x
                x_start = int(max(x-half_winsize_x, 0))
                x_end = int(min(x+half_winsize_x, cols-1))
                y_start = int(max(y-half_winsize_y, 0))
                y_end = int(min(y+half_winsize_y, rows-1))
                self.fastFittingMatrixValues[0,offset] = self.sum_tanX[y_end, x_end] - self.sum_tanX[y_start, x_start]
                self.fastFittingMatrixValues[1,offset] = self.sum_tanY[y_end, x_end] - self.sum_tanY[y_start, x_start]
                self.fastFittingMatrixValues[2,offset] = self.sum_tanX2[y_end, x_end] - self.sum_tanX2[y_start, x_start]
                self.fastFittingMatrixValues[3,offset] = self.sum_tanY2[y_end, x_end] - self.sum_tanY2[y_start, x_start]
                self.fastFittingMatrixValues[4,offset] = self.sum_tanXtanY[y_end, x_end] - self.sum_tanXtanY[y_start, x_start]

    def fastFitPlanes(self, depth_img=[]):
        from numpy import linalg as LA

        rows = self.sum_tanX.shape[0]
        cols = self.sum_tanX.shape[1]
        print("size = %s %s" % (self.sum_tanX.shape[0], self.sum_tanX.shape[1]))
        if (not self.isFastFittingMatricesInitialized()):
            print("Cannot perform plane fitting. Fast fitting matrices has not been initialized...")
            return     
        # MORE CODE GOES HERE
        planes_img = np.zeros(shape=(int(rows), int(cols), 4), dtype=np.float32)        
        #planes_img = np.zeros(shape=(int(rows / self.winsize_x), int(cols / self.winsize_y), 4), dtype=np.float32)
        self.idepth = 1.0/depth_img
        np.nan_to_num(self.idepth, copy=False)
        self.idepth2 = np.multiply(self.idepth, self.idepth)
        self.tanX_idepth = np.multiply(self.tanX, self.idepth)
        self.tanY_idepth = np.multiply(self.tanY, self.idepth)
        cv2.integral(self.idepth, self.sum_idepth)
        cv2.integral(self.idepth2, self.sum_idepth2)
        cv2.integral(self.tanX_idepth, self.sum_tanX_idepth)
        cv2.integral(self.tanY_idepth, self.sum_tanY_idepth)
        sMatrix = np.zeros((4,4),np.float32)
        half_winsize_x = np.floor(self.winsize_x/2)
        half_winsize_y = np.floor(self.winsize_y/2)
        #for y in xrange(int(half_winsize_y), self.winsize_y, rows):
        #    for x in xrange(int(half_winsize_x), self.winsize_x, cols):
        #x = np.arange(0, cols, 1)
        for y in xrange(0, rows, 1):
            for x in xrange(0, cols, 1):
                offset = cols*y + x
#                matrixValues = self.fastFittingMatrixValues[:,offset]
#                sMatrix[0,0] = matrixValues[2]
#                sMatrix[1,1] = matrixValues[3]
#                sMatrix[0,1] = matrixValues[4]
#                sMatrix[1,0] = matrixValues[4]
#                sMatrix[0,2] = matrixValues[0]
#                sMatrix[2,0] = matrixValues[0]
#                sMatrix[1,2] = matrixValues[1]
#                sMatrix[2,1] = matrixValues[1]
#                sMatrix[2,2] = self.winsize_x*self.winsize_y
                x_start = int(max(x-half_winsize_x, 0))
                x_end = int(min(x+half_winsize_x, cols-1))
                y_start = int(max(y-half_winsize_y, 0))
                y_end = int(min(y+half_winsize_y, rows-1))
#                for xx in xrange(0,4,1):
#                   for yy in xrange(0,4,1):
#                       sMatrix[yy,xx] = 1;

                sMatrix[3,0] = self.sum_tanX_idepth[y_end, x_end] - self.sum_tanX_idepth[y_start, x_start]
#                sMatrix[0,3] = sMatrix[0,3]
                sMatrix[3,1] = self.sum_tanY_idepth[y_end, x_end] - self.sum_tanY_idepth[y_start, x_start]
#                sMatrix[1,3] = sMatrix[1,3]
                sMatrix[3,2] = self.sum_idepth[y_end, x_end] - self.sum_idepth[y_start, x_start]
#                sMatrix[2,3] = sMatrix[1,3]
#                sMatrix[3,3] = self.sum_idepth2[y_end, x_end] - self.sum_idepth2[y_start, x_start]
#                if (y == 240 and x == 320):
#                    print("sMatrix(%s,%s) = %s" % (y,x,str(sMatrix)))
#                [_, v] = LA.linalg.eigh(sMatrix)
#                coeffs = v[:,0]
#                scalef = math.sqrt(np.dot(coeffs[0:3],coeffs[0:3])) # Hessian normal form
#                coeffs = coeffs/scalef
#                if (abs(coeffs[3]) > abs(coeffs[2])):  # use d as criterion
#                    if (coeffs[3] < 0):
#                        coeffs = -coeffs
#                    elif (coeffs[2] > 0):   # use c as criterion
#                        coeffs = -coeffs
#                #planes_img[int(y/self.winsize_y), int(x/self.winsize_x), :] = coeffs
#                planes_img[int(y), int(x), :] = coeffs
                
        return planes_img
                
    '''
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
    '''

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

