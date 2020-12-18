# ros_rgbd_cnn
RGBD CNN implementations for ROS

## Usage
The fitting size “fittingSize" is defined at the beginning of the plangent.py, along with the image resolution “image_w, image_h” after fitting. Our weights are trained on a datasdt that are fitter by 4x4 blocks so in the file the fittingSize is 4x4. However, it should also be able to run segmentation on images fitted by different size, say 2x2, with the corresponding changes on variables "fittingSize" and “image_w, image_h”. Note that the “image_w, image_h” have to be multiples of 32. For example, 320x256 with fitting size being 2x2 and 160x128 with fitting size 4x4 (assume original datasize is 640x480).

## weights: 
rgbplane: https://drive.google.com/file/d/1EhRZrh4zsC3I8hlBfMc-HLp3xTxu0xe0/view?usp=sharing

rgbd (pretrained by RedNet): https://drive.google.com/file/d/1yU1ruVKqx_6mUERPWloRHhkpWRr515DK/view?usp=sharing
