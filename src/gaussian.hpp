#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// to generate a 1D Gaussian Kernel
std::vector<float> makeGaussianKernel(int radius, float sigma);

// separable Convolution (horizontal and vertical)
cv::Mat gaussianBlur(const cv::Mat& gray, int radius, float sigma);
