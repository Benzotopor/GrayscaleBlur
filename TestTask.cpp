#include "TestTask.h"

namespace TestTask {

Task::Task(int argc, char** argv) : argc_(argc), argv_(argv)
    { }

void Task::operator() () {
    if (!parseParams()) {
        return;
    }
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    cv::Mat output(image_.rows, image_.cols, CV_8UC1);
    threads_.reserve(threadsCount_ - 1);
    std::cout << "Start processing image. Threads count: " << threadsCount_ << "." << std::endl;
    
    std::cout << "Converting into grayscale format..." << std::endl;
    grayscale(output);
    std::cout << "Converting into grayscale format is finished." << std::endl;
    const double sigma = Task::GaussianSigma;
    const ImageProcessing::GaussianKernel kernel(sigma);
    
    std::cout << "Horizontal blur is processing..." << std::endl;
    horizontalBlur(output, kernel);
    std::cout << "Horizontal blur is finished." << std::endl;
    
    std::cout << "Vertical blur is processing..." << std::endl;
    verticalBlur(output, kernel);
    std::cout << "Vertical blur is finished." << std::endl;
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " μs." <<  std::endl;
    
    cv::imwrite("output.jpg", output);
    std::cout << "Check output.jpg file for the resulting image." << std::endl;
}

void Task::grayscale(cv::Mat& output) {
    unsigned currentPixel = 0, rem = output.total() % threadsCount_;
    const unsigned pixelsPerThead = output.total() / threadsCount_;
    threads_.clear();
    for (int i = 0; i < threadsCount_ - 1; i++) {
        unsigned begin = currentPixel;
        unsigned end = currentPixel + pixelsPerThead;
        if (rem) {
            end++;
            rem--;
        }
        threads_.emplace_back(ImageProcessing::GrayScaleTask(image_, output, begin, end));
        currentPixel = end;
    }
    ImageProcessing::GrayScaleTask task(image_, output, currentPixel, output.total());
    task();
    for (auto& thread : threads_) {
        thread.join();
    }
}
    
void Task::horizontalBlur(cv::Mat& output, const ImageProcessing::GaussianKernel& kernel) {
    unsigned currentRow = 0, rem = output.rows % threadsCount_;
    const unsigned rowsPerThread = output.rows / threadsCount_;
    threads_.clear();
    for (int i = 0; i < threadsCount_ - 1; i++) {
        unsigned begin = currentRow;
        unsigned end = currentRow + rowsPerThread;
        if (rem) {
            end++;
            rem--;
        }
        threads_.emplace_back(ImageProcessing::HorizontalBlurTask(output, kernel, begin, end));
        currentRow = end;
    }
    ImageProcessing::HorizontalBlurTask task(output, kernel, currentRow, output.rows);
    task();
    for (auto& thread : threads_) {
        thread.join();
    }
}

void Task::verticalBlur(cv::Mat& output, const ImageProcessing::GaussianKernel& kernel) {
    unsigned currentCol = 0, rem = output.cols % threadsCount_;
    const unsigned colsPerThread = output.cols / threadsCount_;
    threads_.clear();
    for (int i = 0; i < threadsCount_ - 1; i++) {
        unsigned begin = currentCol;
        unsigned end = currentCol + colsPerThread;
        if (rem) {
            end++;
            rem--;
        }
        threads_.emplace_back(ImageProcessing::VerticalBlurTask(output, kernel, begin, end));
        currentCol = end;
    }
    ImageProcessing::VerticalBlurTask task(output, kernel, currentCol, output.cols);
    task();
    for (auto& thread : threads_) {
        thread.join();
    }
}
    
bool Task::parseParams() {
    if (argc_ != 2 && argc_ != 3) {
        std::cerr << "Wrong input params: program requires image file name and optional threads count." << std::endl;
        return false;
    }
    
    image_ = cv::imread(argv_[1], cv::IMREAD_COLOR);
    if (!image_.data) {
        std::cerr << "Image with given file name does not exist." << std::endl;
        return false;
    }
    
    if (argc_ == 3) {
        try {
            int parsed = std::stoi(argv_[2]);
            if (parsed < 1) {
                throw std::exception();
            }
            threadsCount_ = parsed;
        } catch (std::exception& e) {
            std::cerr << "Param threads count is incorrect." << std::endl;
            return false;
        }
    } else {
        threadsCount_ = std::thread::hardware_concurrency(); // If the value is not well defined or not computable, returns ​0.
        if (threadsCount_ == 0) {
            threadsCount_ = 1;
        }
    }
    
    return true;
}

} // namespace TestTask
