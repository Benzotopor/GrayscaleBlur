#include "ImageProcessing.h"

namespace ImageProcessing {
    
GrayScaleTask::GrayScaleTask(const cv::Mat& src, cv::Mat& dst, unsigned start, unsigned end)
    : src_(src), dst_(dst), start_(start), end_(end)
    { }

void GrayScaleTask::operator() () { 
    for (unsigned i = start_; i < end_; i++) {
        auto pixel = src_.at<cv::Vec3b>(i);
        int b = pixel[0], g = pixel[1], r = pixel[2];
        dst_.at<uchar>(i) = 0.289 * r + 0.587 * g + 0.114 * b; // from rgb to gray pixel
    }
}

GaussianKernel::GaussianKernel(double sigma)
    : sigma_(sigma), radius_(sigma * 3), coefs_(radius_ * 2 + 1, 0) {
        const double denom = sqrt(2 * M_PI) * sigma_;
        coefs_[radius_] = 1.0 / denom;
        double sum = coefs_[radius_];
        for (int i = radius_ + 1, j = radius_ - 1, k = 1; i < coefs_.size(); i++, j--, k++) {
            coefs_[i] = std::exp(-(k * k) / (2.0 * sigma_ * sigma_)) / denom;
            coefs_[j] = coefs_[i];
            sum += coefs_[i] * 2;
        }
        for (auto& c : coefs_) {
            c /= sum;
        }
    }

size_t GaussianKernel::size() const {
    return coefs_.size();
}

double GaussianKernel::at(size_t idx) const {
    return coefs_[idx];
}

HorizontalBlurTask::HorizontalBlurTask(cv::Mat& image, const GaussianKernel& blurKernel, unsigned startRow, unsigned endRow)
    : image_(image), blurKernel_(blurKernel), startRow_(startRow), endRow_(endRow)
    { }
    
void HorizontalBlurTask::operator() () {
    std::vector<double> buffer(image_.cols);
    for (unsigned row = startRow_; row < endRow_; row++) {
        for (int i = 0; i < image_.cols; i++) {
            double p = 0;
            for (int j = 0, di = i - blurKernel_.size() / 2; j < blurKernel_.size(); j++, di++) {
                int x = std::min(image_.cols - 1, std::max(0, di));
                p += image_.at<uchar>(row, x) * blurKernel_.at(j);
            }
            buffer[i] = p;
        }
        for (int i = 0; i < image_.cols; i++) {
            image_.at<uchar>(row, i) = buffer[i];
        }
    }
}

VerticalBlurTask::VerticalBlurTask(cv::Mat& image, const GaussianKernel& blurKernel, unsigned startCol, unsigned endCol)
    : image_(image), blurKernel_(blurKernel), startCol_(startCol), endCol_(endCol)
    { }
    
void VerticalBlurTask::operator() () {
    std::vector<double> buffer(image_.rows);
    for (unsigned col = startCol_; col < endCol_; col++) {
        for (int i = 0; i < image_.rows; i++) {
            double p = 0;
            for (int j = 0, di = i - blurKernel_.size() / 2; j < blurKernel_.size(); j++, di++) {
                int y = std::min(image_.rows - 1, std::max(0, di));
                p += image_.at<uchar>(y, col) * blurKernel_.at(j);
            }
            buffer[i] = p;
        }
        for (int i = 0; i < image_.rows; i++) {
            image_.at<uchar>(i, col) = buffer[i];
        }
    }
}
    
} // namespace ImageProcessing
