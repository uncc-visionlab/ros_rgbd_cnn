/* 
 * File:   planenet.hpp
 * Author: arwillis
 *
 * Created on January 16, 2021, 5:30 AM
 */

#ifndef PLANENET_HPP

#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgcodecs.hpp>

class PlaneNet {
    bool model_loaded;
    torch::jit::script::Module model;
public:

    PlaneNet() : model_loaded(false) {
    }

    virtual ~PlaneNet() {
    }

    void loadModel(std::string modelfilepath);
    
    cv::Mat eval(cv::Mat& rgb, cv::Mat& plane);

};
#define PLANENET_HPP



#endif /* PLANENET_HPP */

