#include <ros_rgbd_surface_tracker/rgbd_image_uncc.hpp>

#include <torch/script.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>

#define image_w 160
#define image_h 128

namespace cv {
    namespace rgbd {
        cv::Mat SemanticSegmantation::inference(const cv::rgbdRgbdImage& rgbd_img) const {
                
            const std::string _MODEL_JIT_FILENAME = "../../../weights/tracing_rgbplane_model.pt";

            cv::Mat img = rgbd_img.getRGBImage();
            rgbd_img.computePlanes();
            cv::Mat planes = rgbd_img.getPlanes();

            // the trained model is not input dimension-sensitive. It was trained on 160*128 dataset 
            // but it can also be used for inference on 640*480, 320*240 or any dimensions that are multiples of 32
            cv::Mat resize_rgb, resize_plane, rgb_norm;
            cv::resize(rgb, resize_rgb, cv::Size(image_h, image_w), cv::INTER_LINEAR);
            cv::resize(plane, resize_plane, cv::Size(image_h, image_w), cv::INTER_NEAREST);
            resize_rgb.convertTo(rgb_norm, CV_32F, 1.0/255, 0); 
            torch::Tensor rgb_tensor = torch::from_blob(rgb_norm.data, {image_h, image_w, 3}, torch::kFloat);
            torch::Tensor plane_tensor = torch::from_blob(resize_plane.data, {image_h, image_w, 3}, torch::kFloat);
            rgb_tensor = rgb_tensor.permute({2, 0, 1});
            plane_tensor = plane_tensor.permute({2, 0, 1});
            
            //  Normalize data
            rgb_tensor[0] = rgb_tensor[0].sub(0.485).div(0.229);
            rgb_tensor[1] = rgb_tensor[1].sub(0.456).div(0.224);
            rgb_tensor[2] = rgb_tensor[2].sub(0.406).div(0.225);
            plane_tensor[0] = plane_tensor[0].sub(0).div(1);
            plane_tensor[1] = plane_tensor[1].sub(0).div(1);
            plane_tensor[2] = plane_tensor[2].sub(5).div(10);
            auto unsqueezed_rgb = rgb_tensor.unsqueeze(0);
            auto unsqueezed_plane = plane_tensor.unsqueeze(0);

            // Create a vector of inputs.
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(unsqueezed_rgb);
            inputs.push_back(unsqueezed_plane);

            // Deserialize the ScriptModule from a file using torch::jit::load().
            torch::jit::script::Module model = torch::jit::load( _MODEL_JIT_FILENAME);

            // Execute the model and turn its output into a tensor.
            auto output = model.forward(inputs).toTuple()->elements()[0].toTensor(); // Now works on CPU. Need to convert to GPU if required 
            std::tuple<at::Tensor, at::Tensor> result = torch::max(output, 1);
            //torch::Tensor result = torch::max(output, 1)

            torch::Tensor label_data = std::get<1>(result) + 1; // 1*image_w*image_h
            label_data = label_data.squeeze().detach(); // image_w*image_h
            label_data = label_data.contiguous();

            //label_data = label_data.to(torch::kCPU);
            cv::Mat label(image_h, image_w, CV_8UC1);
            std::memcpy((void *) label.data, label_data.data_ptr(), sizeof(torch::kU8)*label_data.numel());
            return label;
        }
    }
}