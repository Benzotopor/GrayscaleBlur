#pragma once

#include <iostream>
#include <string>
#include <thread>
#include "ImageProcessing.h"

namespace TestTask {
    
class Task {
public:

    Task(int argc, char** argv);
    void operator() ();

private:
    
    bool parseParams();
    void grayscale(cv::Mat& output);
    void horizontalBlur(cv::Mat& output, const ImageProcessing::GaussianKernel& kernel);
    void verticalBlur(cv::Mat& output, const ImageProcessing::GaussianKernel& kernel);
    
    cv::Mat image_;
    std::vector<std::thread> threads_;
    unsigned threadsCount_;
    int argc_;
    char** argv_;
    
    constexpr static const double GaussianSigma = 2;
};
    
} // namespace TestTask
