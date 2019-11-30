#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace ImageProcessing {

struct GrayScaleTask {

    GrayScaleTask(const cv::Mat& src, cv::Mat& dst, unsigned start, unsigned end);
    void operator() ();

private:
    const cv::Mat& src_;
    cv::Mat& dst_;
    const unsigned start_, end_;
};

struct GaussianKernel {
    
    GaussianKernel(double sigma);
    size_t size() const;
    double at(size_t idx) const;

private:
    const double sigma_;
    const unsigned radius_;
    std::vector<double> coefs_;
};

struct HorizontalBlurTask {
        
    HorizontalBlurTask(cv::Mat& image, const GaussianKernel& blurKernel, unsigned startRow, unsigned endRow);
    void operator() ();
    
private:
    cv::Mat& image_;
    const GaussianKernel& blurKernel_;
    const unsigned startRow_, endRow_;
};

struct VerticalBlurTask {
        
    VerticalBlurTask(cv::Mat& image, const GaussianKernel& blurKernel, unsigned startCol, unsigned endCol);
    void operator() ();

private:
    cv::Mat& image_;
    const GaussianKernel& blurKernel_;
    const unsigned startCol_, endCol_;
};
    
} // namespace ImageProcessing
