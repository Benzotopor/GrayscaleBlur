# Utility for converting image to grayscale mode and blurring.
Program converts a given image to grayscale mode and blurs it.
## Install
```
git clone https://github.com/Benzotopor/GrayscaleBlur.git
cd GrayscaleBlur
```
## Build via CMake
Build requires installed and linked OpenCV. See [Introduction to OpenCV](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)
```
mkdir build
cd build
cmake ..
make
```
## Usage
```
./grayscaleblur path_to_image threads_count
```
The number of threads is an optional parameter. If it is omitted, the program will use the OS recommendations for the number of threads.\
The resulting image will be created at the program location as the image output.jpg.