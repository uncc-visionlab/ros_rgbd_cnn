/* 
 * File:   planenet.hpp
 * Author: arwillis
 *
 * Created on January 16, 2021, 5:30 AM
 */

#ifndef PLANENET_HPP

#include <boost/make_shared.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgcodecs.hpp>

class PyTorchPlaneClassifier {
public:
    typedef boost::shared_ptr<PyTorchPlaneClassifier> Ptr;
    virtual void loadModel(std::string modelfilepath) = 0;
    virtual cv::Mat eval(const cv::Mat& rgb, const cv::Mat& plane) = 0;
};

class PlaneNet : public PyTorchPlaneClassifier {
    bool model_loaded;
    torch::jit::script::Module model;
public:
    typedef boost::shared_ptr<PlaneNet> Ptr;

    PlaneNet();
    virtual ~PlaneNet();
    
    void loadModel(std::string modelfilepath);

    cv::Mat eval(const cv::Mat& rgb, const cv::Mat& plane);

    static PlaneNet::Ptr create();
};
#define PLANENET_HPP

#endif /* PLANENET_HPP */

