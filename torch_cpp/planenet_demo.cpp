#include <fstream>
#include <iostream>
#include <memory>

#include <ros_rgbd_cnn/planenet.hpp>

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

void writeCSV(std::string fileName, cv::Mat m) {
    std::ofstream myfile;
    myfile.open(fileName.c_str());
    myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
}

int main() {
		
    // need a rgb and a plane image here
    const cv::String _RGB_FILENAME = "test_data/NYU0002.jpg";
    const cv::String _PLANE_FILENAME = "test_data/plane.txt";
    const std::string _MODEL_JIT_FILENAME= "model/tracing_rgbplane_model.pt";
    
    cv::Mat rgb = cv::imread(_RGB_FILENAME);
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
    cv::Mat plane = ReadMatFromTxt(_PLANE_FILENAME, 106, 140, 3, CV_32FC3);
    
    PlaneNet planenet;
    planenet.loadModel(_MODEL_JIT_FILENAME);
    cv::Mat labels = planenet.eval(rgb, plane);

    std::cout << labels.at<cv::Vec2f>(20,70) << "\n";
    //std::string fileName = "label.csv";
    //writeCSV(fileName, labels);
    return 0;
}


