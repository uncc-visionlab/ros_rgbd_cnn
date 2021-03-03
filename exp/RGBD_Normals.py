import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import cv_uncc
import numpy as np
import imageio
import scipy.io
import datetime
import csv


def LineIntersectPlane(P1, P2, planecoeffs):
    p0 = np.array([0, 0, np.asscalar(-planecoeffs[3]/planecoeffs[2])]).reshape(3, 1)
    d = np.asscalar(np.matmul((p0 - P2).transpose(), planecoeffs[0:3]) / np.matmul((P1-P2).transpose(), planecoeffs[0:3]))
    iP1 = P2 + (P1-P2) * d
    return iP1


def LineIntersectSphere(P1,P2,C,r):
    #
    # [iP1,iP2] = LineIntersectSphere(P1,P2,C,r)
    #
    # Intersect a line that passes through the 2 3D points (P1,P2) with the sphere of
    # radius r centered at the 3D point C.
    #
    # Andrew Willis
    # UNC Charlotte
    # September 7, 2006
    #
    dist, nullPt = LineToPointDistance(P1,P2,C)
    if (dist > r):
        iP1 = np.Inf
        iP2 = np.Inf
        return iP1,iP2
    P3 = C
    a = np.matmul((P2-P1).transpose(), (P2-P1))
    b = 2 * (np.matmul((P2-P1).transpose(), (P1-P3)))
    c = np.matmul(P3.transpose(), P3)+np.matmul(P1.transpose(), P1)-2*(np.matmul(P1.transpose(), P3))-r ** 2
    u1 = np.real(-b + np.sqrt(b ** 2 - 4*a*c))/(2*a)
    u2 = np.real(-b - np.sqrt(b ** 2 - 4*a*c))/(2*a)
    iP1 = P1+u1*(P2-P1)
    iP2 = P1+u2*(P2-P1)
    return iP1,iP2


def LineToPointDistance(P1,P2,C):
    #
    # [r,P] = LineToPointDistance(P1,P2,C)
    #
    # Find the distance r of the point C to the line that passes through the points (P1,P2).
    # The point P is the point on the line closest to the point C.
    #
    # Andrew Willis
    # UNC Charlotte
    # September 7, 2006
    #
    u = np.asscalar(np.matmul((C-P1).transpose(), ((P2-P1))))/(np.linalg.norm((P2-P1)) ** 2)
    P = P1 + u*(P2-P1)
    r = np.linalg.norm(C-P)
    return r, P


def computeNormals(dims, K, GEOMETRY, NOISE_TYPE, noisemag):
    
    dx = np.zeros((1, dims[1]))
    dy = np.zeros((1, dims[0]))
    for x  in range(dims[1]):
        dx[:, x] = (x-K[0,2])/K[0,0]
    for y in range(dims[0]):
        dy[:, y] = (y-K[1,2])/K[1,1]
    Ix = np.matmul(np.ones((dims[0], 1)), dx)
    Iy = np.matmul(dy.transpose(), np.ones((1, dims[1])))
    Iz = np.ones((dims[0], dims[1]))

    C = np.array([0, 0, 3]).reshape(3, 1)
    r = 1
    P1 = np.array([0, 0, 0]).reshape(3, 1)
    planecoeffs = np.array([0, 3, -2, 6]).reshape(4, 1)  # np.array([0, 1, -1, 2]).reshape(4, 1)
    planecoeffs = planecoeffs/np.linalg.norm(planecoeffs[0:3])
    #print(planecoeffs)
    depth = np.zeros((dims[0], dims[1]), np.float32) 
    rgb = np.zeros((dims[0], dims[1], 3), np.uint8)
    normals = np.zeros((dims[0], dims[1],3))
    normalsIdx = [] 
    ipt1 = np.Inf
    ipt2 = np.Inf
    for row in range(dims[0]): 
        for col in range(dims[1]):
            P2 = np.array([Ix[row,col], Iy[row,col], Iz[row,col]])
            P2 = P2.reshape(3, 1)
            if (GEOMETRY == 1):
                #print(P1)
                #print(P2)
                ipt1, ipt2 = LineIntersectSphere(P1, P2, C, r)
            else:
                ipt1 = LineIntersectPlane(P1, P2, planecoeffs)
            if (np.any(np.isinf(ipt1)) and np.any(np.isinf(ipt2))):
                depth[row,col] = 4
                rgb[row,col,0] = 0
                rgb[row,col,1] = 0
                rgb[row,col,2] = 0
                normals[row,col,:] = [0, 0, -1]                
            else:
                if ((not np.any(np.isinf(ipt2))) and (ipt1[2] > ipt2[2])):
                    ipt1 = ipt2
                depth[row,col] = round(ipt1[2], 3)
                rgb[row,col,0] = 50
                rgb[row,col,1] = 168
                rgb[row,col,2] = 82                
                if (GEOMETRY == 1):
                    normals[row,col,:] = ((ipt1-C)/np.linalg.norm(ipt1-C)).squeeze()
                else:
                    normals[row,col,:] = (planecoeffs[0:3]/np.linalg.norm(planecoeffs[0:3])).squeeze()
                normalsIdx.append([row, col])
            # noise constant from
            # Khoshelham, K.; Elberink, S.O. Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications. Sensors 2012, 12, 1437-1454. https://doi.org/10.3390/s12020143            
            if (NOISE_TYPE == 1):
                depth[row, col] = depth[row, col] + np.random.normal(0, noisemag[row]*depth[row,col] ** 2, 1)
            elif (NOISE_TYPE == 2): None
            elif (NOISE_TYPE == 0): None
            else:
                print("Invalid NOISE_TYPE argument!")

    #xyzPoints = depth.flatten('F') * np.array([Ix.flatten('F'), Iy.flatten('F'), Iz.flatten('F')]) # [height*width, 3]
    #color = rgb.astype(np.double) / np.max(rgb)
    #color = color.reshape(-1, 3)
    #points3D = xyzPoints.transpose()
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c=color, s=0.01)
    #plt.show()
    return rgb, depth, normals, normalsIdx


if __name__ == "__main__":
    
    np.set_printoptions(precision=4, suppress=True)   # array printing precision 
    
    #SPHERE = 1, PLANE = 2

    # NOISE_TYPE = 0 --> ZERO NOISE OR PERFECT MEASUREMENTS
    # NOISE_TYPE = 1 --> NOISE IS IN THE DIRECTION OF THE DEPTH MEASUREMENT
    # NOISE_TYPE = 2 --> NOISE IS IN THE DIRECTION OF THE SURFACE NORMAL
    
    accuracyTest = 1
    accuracyVSdepthTest = 0
    timeTest = 0
    CSVheader = ([" ", "Explicit", "Implicit", "SRI", "FALS"]) 
    if (accuracyTest):
        with open('err results.csv', 'w') as file:
            writer = csv.writer(file) 
            writer.writerow(CSVheader)
        with open('std results.csv', 'w') as file:
            writer = csv.writer(file) 
            writer.writerow(CSVheader)
    elif (accuracyVSdepthTest):
        headerWriterCnt = 1;    # only write header once 
        with open('deptherr results.csv', 'w') as file:
            None
    elif (timeTest):
        with open('time results.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(CSVheader)
           
    noiseType = 1
    for GEOMETRY in [1]:
        for SCALE in [1]: #np.arange(0.2, 1.05, 0.05):
            for noisefactor in np.arange(0, 3.1, 0.1):                                  
                dims = np.array((480/SCALE, 640/SCALE), np.int)
                K = np.array([567.84/SCALE, 0, (dims[1]-1)/2.0, 0, 567.84/SCALE, (dims[0]-1)/2.0, 0, 0, 1], dtype=float)                
                K = K.reshape(3, 3)                
                windowSize = (5, 5)
                noisemag = np.full(dims[0], 0.001425) * noisefactor    #  generate noise along the depth 
                if(accuracyVSdepthTest and GEOMETRY==2):
                    noisemag = np.multiply(np.linspace(1, 2, num=dims[0]), noisemag).transpose()  #  generate noise along the depth (depth increases top-down)
                rgb, depth, normals, normalsIdx = computeNormals(dims, K, GEOMETRY, noiseType, noisemag)
                #plt.imsave("depth.png", depth, cmap='Greys')
                #cv2.imwrite("rgb.jpg", rgb)                
                rgbd = cv_uncc.rgbd.RgbdImage(rgb, depth, K[0, 2], K[1, 2], K[0, 0])
                rgbd.initializeIntegralImages(windowSize)
                #rgbd = cv_uncc.rgbd.RgbdImage(rgb, depth, self._center_x, self._center_y, 1.0/self._ifocal_length_y);
                
                errArr = []
                stdArr = []
                errdepthArr = []
                durationArr = []
                for method in range(5):
                    err = 0
                    std = 0
                    errdepth = 0
                    duration = 0
                    if method == 2:  #  lineMod not working
                        None
                    else:
                        for i in range(10):
                            duration += rgbd.computeNormals(method)
                            normalsImg = rgbd.getNormals()                                                            
                            errImg = np.zeros([dims[0], dims[1], 3])
                            validCnt = 0
                            for coor in normalsIdx:
                                row = coor[0]
                                col = coor[1]
                                if (np.any(np.isnan(normalsImg[row, col, :])) or np.any(np.isinf(normalsImg[row, col, :]))):
                                    None
                                    #print("An invalid value at " + str([row, col]))
                                else:
                                    errImg[row, col, :] = abs(normalsImg[row, col, :] - normals[row, col, :]) 
                                    validCnt += 1   # most of time valiCnt = len(normalsIdx) except ~40 estimates are invalid in planar depth image with SRI method 
                            std += np.array([np.std(errImg[:,:,0]), np.std(errImg[:,:,1]), np.std(errImg[:,:,2])])
                            err += np.sum(np.sum(errImg, axis = 0), axis = 0) / validCnt
                            if (accuracyVSdepthTest == 1 and GEOMETRY == 2):
                                errImgNorm = np.linalg.norm(errImg, axis=2)
                                errdepth += np.sum(errImgNorm, axis=1) / np.shape(errImgNorm)[1]
#                            if (i == 1):
#                                print("valid pixels in estimated result = " + str(validCnt))
#                                print("valid pixels in ground truth result = " + str(len(normalsIdx)))                                    
                        normerr = np.average(err/10)  # np.linalg.norm(err/10)
                        normstd = np.average(err/10)  # np.linalg.norm(std/10)
                        errArr.append(normerr)
                        stdArr.append(normstd)
                        errdepthArr.append(errdepth/10)
                        durationArr.append(duration*1000/10)  # ms
                        
                if (accuracyTest == 1):
                    with open('err results.csv', 'a') as file: 
                        writer = csv.writer(file)
                        writer.writerow([str(noisefactor), str(errArr[0]), str(errArr[1]), str(errArr[2]), str(errArr[3])])                        
                    with open('std results.csv', 'a') as file:
                        writer = csv.writer(file) 
                        writer.writerow([str(noisefactor), str(stdArr[0]), str(stdArr[1]), str(stdArr[2]), str(stdArr[3])])
                elif (accuracyVSdepthTest == 1 and GEOMETRY==2):                    
                    with open('deptherr results.csv', 'a') as file:
                        writer = csv.writer(file) 
                        if (headerWriterCnt == 1):
                            #depthCol = np.zeros((1, dims[1]))
                            depthCol = np.sum(depth, axis=1) / np.shape(depth)[1]  # depth value along columns
                            writer.writerow([" ", depthCol])
                            headerWriterCnt += 1
                        writer.writerow(["Explicit", errdepthArr[0]]) 
                        writer.writerow(["Implicit", errdepthArr[1]])
                        writer.writerow(["SRI", errdepthArr[2]])
                        writer.writerow(["FALS", errdepthArr[3]])
                elif (timeTest == 1):
                    with open('time results.csv', 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow([str(dims[1])+"x"+str(dims[0]), str(durationArr[0]), str(durationArr[1]), str(durationArr[2]), str(durationArr[3])])                                                           
    print("Finished")