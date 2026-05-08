/**
 * @file gaussian.hpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief Custom implementation of Gaussian Blur. OpenMP parallelization is used to improve performance.
 **/

#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// to generate a 1D Gaussian Kernel
std::vector<float> makeGaussianKernel(int radius, float sigma);

// separable Convolution (horizontal and vertical)
cv::Mat gaussianBlur(const cv::Mat& gray, int radius, float sigma);
