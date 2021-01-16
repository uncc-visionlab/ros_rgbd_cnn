#include <torch/script.h>
#include <iostream>
#include <memory>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgcodecs.hpp>

#define image_w 160
#define image_h 128

cv::Mat ReadMatFromTxt(std::string filename, int rows, int cols, int channel = 0, int type = CV_32F)
{
    double m;
    if (channel == 0)
    {
        cv::Mat out = cv::Mat::zeros(rows, cols, type);//Matrix to store values
        std::ifstream fileStream(filename);
        int cnt = 0;//index starts from 0
        while (fileStream >> m)
        {
            int temprow = cnt / cols;
            int tempcol = cnt % cols;
            out.at<float>(temprow, tempcol) = m;
            cnt++;
        }   
        return out;
    }
    else
    {
        int dims[] = {rows, cols, channel};
        //cv::Mat out(3, dims, type, cv::Scalar(0)); //Matrix to store value
        cv::Mat out(rows, cols, type, cv::Scalar(0));
        std::ifstream fileStream(filename);
        int k = 0;//index starts from 0
        int cnt = 0;
        int temprow, tempcol;
        while (fileStream >> m)
        {   
            if (k % channel == 0)
            {
                temprow = cnt / cols;
                tempcol = cnt % cols;
                cnt++;
            }
            out.ptr<float>(temprow, tempcol)[k%channel] = m;           
            k++;   
        }   
        return out;
    }
}

int main() {
		
    // need a rgb and a plane image here
    const cv::String _RGB_FILENAME = "test_data/NYU0002.jpg";
    const cv::String _PLANE_FILENAME = "test_data/plane.txt";
    const std::string _MODEL_JIT_FILENAME= "model/tracing_rgbplane_model.pt";

    cv::Mat rgb = cv::imread(_RGB_FILENAME);
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
    cv::Mat plane = ReadMatFromTxt(_PLANE_FILENAME, 106, 140, 3, CV_32FC3);

    cv::Mat resize_rgb, resize_plane, rgb_norm;
    cv::resize(rgb, resize_rgb, cv::Size(image_w, image_h), cv::INTER_LINEAR);
    cv::resize(plane, resize_plane, cv::Size(image_w, image_h), cv::INTER_NEAREST);
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
    //cv::Mat{label_data.size(0), label_data.size(1), CV_8U, label_data.data(uchar)};
    std::cout << label.at<int>(20,70) << "\n";
    //std::cout << label_data << "\n";
    return 0;
}


