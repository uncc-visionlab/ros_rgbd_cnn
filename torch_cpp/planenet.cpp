#include <ros_rgbd_cnn/planenet.hpp>
#include <fstream>

#define image_w 160
#define image_h 128

PlaneNet::PlaneNet() : model_loaded(false) {
}

PlaneNet::~PlaneNet() {
}

PlaneNet::Ptr PlaneNet::create() {
    return PlaneNet::Ptr(boost::make_shared<PlaneNet>());
}

void PlaneNet::loadModel(std::string modelfilepath) {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(modelfilepath);
    model_loaded = true;
}

cv::Mat PlaneNet::eval(const cv::Mat& rgb, const cv::Mat& plane) {
    if (!model_loaded) {
        std::cout << "Error Torch model is not loaded!" << std::endl;
    }
    cv::Mat abcd[4];
    cv::split(plane, abcd);
    std::vector<cv::Mat> bcd = {abcd[1],abcd[2],abcd[3]}; 
    cv::Mat plane3(abcd[0].rows, abcd[0].cols, CV_32FC3);
    cv::merge(bcd, plane3); // y, z, d channels of plane image
    cv::Mat resize_rgb, resize_plane, rgb_norm;
    //cv::resize(rgb, resize_rgb, cv::Size(image_w, image_h), cv::INTER_LINEAR);
    //cv::resize(plane3, resize_plane, cv::Size(image_w, image_h), cv::INTER_NEAREST);
    resize_rgb = rgb;
    resize_plane = plane3;
    resize_rgb.convertTo(rgb_norm, CV_32F, 1.0 / 255, 0);
    cv::Mat label(resize_rgb.rows, resize_rgb.cols, CV_32FC2);
    
    std::chrono::steady_clock::time_point begin;
    // ### Model 1 (PlaneNet) ### //
    if (false) {
        begin = std::chrono::steady_clock::now();
        torch::Tensor rgb_tensor = torch::from_blob(rgb_norm.data,{resize_rgb.rows, resize_rgb.cols, 3}, torch::kFloat);
        torch::Tensor plane_tensor = torch::from_blob(resize_plane.data,{resize_plane.rows, resize_plane.cols, 3}, torch::kFloat);
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

        // Execute the model and turn its output into a tensor.
        auto output = model.forward(inputs).toTuple()->elements()[0].toTensor(); // Now works on CPU. Need to convert to GPU if required 
        auto output_softmax = torch::softmax(output, 1);
        std::tuple<at::Tensor, at::Tensor> result = torch::max(output_softmax, 1);
        //torch::Tensor result = torch::max(output, 1)

        torch::Tensor label_data = std::get<1>(result) + 1; // 1*image_w*image_h
        torch::Tensor label_prob = std::get<0>(result);
        label_data = label_data.squeeze().detach().to(torch::kF32);; // image_w*image_h
        label_data = label_data.to(torch::kCPU);
        label_prob = label_prob.squeeze().detach().to(torch::kF32);; // image_w*image_h
        label_prob = label_prob.to(torch::kCPU);

        cv::Mat labelData(resize_rgb.rows, resize_rgb.cols, CV_32FC1);  // Mat(int rows, int cols, int type)
        cv::Mat labelProb(resize_rgb.rows, resize_rgb.cols, CV_32FC1);
        std::memcpy((void *) labelData.data, label_data.data_ptr(), torch::elementSize(torch::kF32)*label_data.numel());
        std::memcpy((void *) labelProb.data, label_prob.data_ptr(), torch::elementSize(torch::kF32)*label_prob.numel());
        //std::cout << label.size() << "\n";  // returns cv:::Size(cols, rows)
        //torch::Tensor label_data_cpu = label_data.cpu();
        //cv::Mat label(image_h, image_w, CV_8UC1, label_data_cpu.data_ptr<long>());
        std::vector<cv::Mat> labelVec = {labelData,labelProb}; 
        cv::merge(labelVec, label);
    }
    else{        
        // ### Model 2 (BiSeNetv1) ###
        cv::Mat resize_bcd[3];
        cv::split(resize_plane, resize_bcd);
        double min, max;
        cv::minMaxLoc(resize_bcd[2], &min, &max);
        std::vector<cv::Mat> bcd_norm = {(resize_bcd[0] + 1) / 2, (resize_bcd[1] + 1) / 2, (resize_bcd[2] - (float)min) / ((float)max - (float)min)}; 
        cv::Mat rgbChannel[3];
        cv::split(rgb_norm, rgbChannel);
        cv::Mat srcImg(resize_plane.rows, resize_plane.cols, CV_32F);
        std::vector<cv::Mat> srcData = {rgbChannel[0], rgbChannel[1], rgbChannel[2], bcd_norm[0], bcd_norm[1], bcd_norm[2]}; 
        cv::merge(srcData, srcImg);
        
        begin = std::chrono::steady_clock::now();
        torch::Tensor srcImg_tensor = torch::from_blob(srcImg.data,{srcImg.rows, srcImg.cols, 6}, torch::kFloat);
        srcImg_tensor = srcImg_tensor.permute({2, 0, 1});

        srcImg_tensor[0] = srcImg_tensor[0].sub(0.485).div(0.229);
        srcImg_tensor[1] = srcImg_tensor[1].sub(0.456).div(0.224);
        srcImg_tensor[2] = srcImg_tensor[2].sub(0.406).div(0.225);
        srcImg_tensor[3] = srcImg_tensor[3].sub(0.485).div(0.229);
        srcImg_tensor[4] = srcImg_tensor[4].sub(0.456).div(0.224);
        srcImg_tensor[5] = srcImg_tensor[5].sub(0.406).div(0.225);

        auto unsqueezed_img = srcImg_tensor.unsqueeze(0);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(unsqueezed_img);

        // Execute the model and turn its output into a tensor.
        auto output = model.forward(inputs).toTuple()->elements()[0].toTensor(); // Now works on CPU. Need to convert to GPU if required 
        auto output_softmax = torch::softmax(output, 1);
        std::tuple<at::Tensor, at::Tensor> result = torch::max(output_softmax, 1);
        //torch::Tensor result = torch::max(output, 1)

        torch::Tensor label_data = std::get<1>(result) + 1; // 1*image_w*image_h
        torch::Tensor label_prob = std::get<0>(result);
        label_data = label_data.squeeze().detach().to(torch::kF32);; // image_w*image_h
        label_data = label_data.to(torch::kCPU);
        label_prob = label_prob.squeeze().detach().to(torch::kF32);; // image_w*image_h
        label_prob = label_prob.to(torch::kCPU);

        cv::Mat labelData(srcImg.rows, srcImg.cols, CV_32FC1);  // Mat(int rows, int cols, int type)
        cv::Mat labelProb(srcImg.rows, srcImg.cols, CV_32FC1);
        std::memcpy((void *) labelData.data, label_data.data_ptr(), torch::elementSize(torch::kF32)*label_data.numel());
        std::memcpy((void *) labelProb.data, label_prob.data_ptr(), torch::elementSize(torch::kF32)*label_prob.numel());
        //std::cout << label.size() << "\n";  // returns cv:::Size(cols, rows)
        //torch::Tensor label_data_cpu = label_data.cpu();
        //cv::Mat label(image_h, image_w, CV_8UC1, label_data_cpu.data_ptr<long>());
        std::vector<cv::Mat> labelVec = {labelData, labelProb};
        cv::merge(labelVec, label);
    }
        
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Classification duration = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
 
    return label;
    
}


