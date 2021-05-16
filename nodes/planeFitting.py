import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import cv_uncc
import sys

os.chdir('/home/jzhang72/PycharmProjects/PlaneNet')

directory = [['./data/depth_dir_train.txt',
              './data/img_dir_train.txt',
              './data/label_train.txt',
              './data/int_dir_train.txt'],

             ['./data/depth_dir_test.txt',
              './data/img_dir_test.txt',
              './data/label_test.txt',
              './data/int_dir_test.txt']]
              

def matrix_from_txt(file):
    f = open(file)
    l = []
    for line in f.readlines():
        line = line.strip('\n')
        for j in range(len(list(line.split()))):
            l.append(line.split()[j])
    matrix = np.array(l, dtype=np.float32)
    return matrix

windowSize = (7, 7)
log_path = "/home.md1/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/log_plane.txt"
log = open(log_path, "w")
log.close()

for i in range(len(directory)):
    if i == 0:
        log = open(log_path, "a")
        log.write("Processing Training Set...\n")
        log.close()
        print("Processing Training Set...")
    else:
        log = open(log_path, "a")
        log.write("Processing Testing Set...\n")
        log.close()
        print("Processing Testing Set...")
    with open(directory[i][0], 'r') as f0:
        depth_dir = f0.read().splitlines()
    with open(directory[i][1], 'r') as f1:
        img_dir = f1.read().splitlines()
    with open(directory[i][2], 'r') as f2:
        label_dir = f2.read().splitlines()
    with open(directory[i][3], 'r') as f3:
        int_dir = f3.read().splitlines()
    
#depth_dir = ["/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv1/NYUdata/NYU0013/depth_bfx/NYU0013.png", "/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/depth_bfx/0000121.png"]
#img_dir = ["/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv1/NYUdata/NYU0013/image/NYU0013.jpg", "/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/image/0000121.jpg"]
#ext_dir = ["/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv1/NYUdata/NYU0013/intrinsics.txt", "/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/intrinsics.txt"]
#int_dir = ext_dir
#label_dir = ["/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv1/NYUdata/NYU0013/label/label.npy", "/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv2/kinect2data/000065_2014-05-16_20-14-38_260595134347_rgbf000121-resize/label/label.npy"]
    for idx in range(len(depth_dir)):
        labelPath = label_dir[idx]
        depth = imageio.imread(depth_dir[idx])
        rgb = imageio.imread(img_dir[idx])
        intrinsicPath = int_dir[idx]
        planeSavePath = labelPath.split('label/label.npy')
        log = open(log_path, "a")
        log.write("Processing <" + planeSavePath[0] + ">\n")
        log.close()
        print("Processing <" + planeSavePath[0] + ">")        

        K = matrix_from_txt(intrinsicPath)
        focal_length_x = K[0]
        focal_length_y = K[4]
        center_x = K[2]
        center_y = K[5]

        depth = np.asarray(depth, np.uint16)
        depthInpaint = np.bitwise_or(np.right_shift(depth, 3), np.left_shift(depth, 16 - 3))
        depthInpaint = depthInpaint.astype(np.single) / 1000
        depthInpaint[depthInpaint > 8] = 8
        rgbd = cv_uncc.rgbd.RgbdImage(rgb, depthInpaint, center_x, center_y, focal_length_x)        
        #rgbd.initializeIntegralImages(windowSize)
        #duration = rgbd.computeNormals()
        #normals_img = rgbd.getNormals()
        duration = rgbd.computePlanes(4)  # method #4 is the opencv FALS. EXPL_IMPL is not working here when image size changes inside the for loop
        #duration = rgbd.computePlanes(0)
        planeImage = rgbd.getPlanes()

        #plane = depth2plane(depth, extrinsicPath, intrinsicPath, labelPath, fittingSize)
        #planeLabel = plane.getPlaneLabel()
        #planeImage = plane.getPlaneImage()
        planeImgSavePath = planeSavePath[0] + 'planeImage/' + 'planeImage.tiff'
        normalsImgSavePath = planeSavePath[0] + 'planeImage/' + 'normals.jpg'
        #planeTxtPath = planeSavePath[0] + 'planeImage/' + 'planecoeffs.txt'
        #planeLabelSavePath = planeSavePath[0] + 'planeLabel/' + 'label.npy'
        imageio.imwrite(planeImgSavePath, planeImage)
        cv2.imwrite(normalsImgSavePath, np.floor(255*abs(planeImage[:,:,0:3])))
        cv2.imshow("normals", np.uint8(np.floor(255*abs(planeImage[:,:,0:3]))))
        cv2.waitKey()
        #if os.path.exists(planeTxtPath):
        #    os.remove(planeTxtPath)
        #with open(planeLabelSavePath, 'wb') as f:
        #    np.save(f, planeLabel)

log = open(log_path, "a")
log.write("All Images and Labels Generated!\n")
log.close()
print("All Images and Labels Generated!")


#rgb_cmp = imageio.imread("/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv1/NYUdata/NYU0014/image/NYU0014.jpg")
#depth_cmp = imageio.imread("/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv1/NYUdata/NYU0014/depth_bfx/NYU0014.png")
##depthVisData = np.asarray(depth_cmp, np.uint16)
##depthInpaint = np.bitwise_or(np.right_shift(depthVisData, 3), np.left_shift(depthVisData, 16 - 3))
##depthInpaint = depthInpaint.astype(np.single) / 1000
##depthInpaint[depthInpaint > 8] = 8
##rgbd_cmp = cv_uncc.rgbd.RgbdImage(rgb_cmp, depthInpaint, 284.582449, 208.736166, 518.857901);
###rgbd.computeNormals()
###normals_img = rgbd.getNormals()
##rgbd_cmp.computePlanes()
##planes_img_cmp = rgbd_cmp.getPlanes()
##print(planes_img_cmp.dtype)
##print(planes_img_cmp.shape)
#
#intrinsicPath = "/home/jzhang72/PycharmProjects/PlaneNet/data/SUNRGBD/kv1/NYUdata/NYU0014/intrinsics.txt";
#K = matrix_from_txt(intrinsicPath)
#focal_length_x = K[0]
#focal_length_y = K[4]
#center_x = K[2]
#center_y = K[5]
#
#depthInpaint = np.bitwise_or(np.right_shift(depth_cmp, 3), np.left_shift(depth_cmp, 16 - 3))
#depthInpaint = depthInpaint.astype(np.single) / 1000
#depthInpaint[depthInpaint > 8] = 8
#rgbd = cv_uncc.rgbd.RgbdImage(rgb_cmp, depthInpaint, center_x, center_y, focal_length_x)
#print(center_x)
#print(center_y)
#print(focal_length_x)
##rgbd.computeNormals()
##normals_img = rgbd.getNormals()
#rgbd.computePlanes()
#planes_img_cmp = rgbd.getPlanes()
#
#
#imageio.imwrite('planes.tiff', planes_img_cmp)
#cv2.imwrite('normals.jpg', np.floor(255*abs(planes_img_cmp[:,:,0:3])))