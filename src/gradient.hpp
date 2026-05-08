/**
 * @file gradient.hpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief Sobel operator for gradient magnitude and orientation
 **/

#pragma once
#include <opencv2/core.hpp>

struct GradientResult{
    cv::Mat magnitude;
    cv::Mat orientation;
};

// use sobel to compute gradient
GradientResult computeGradient(const cv::Mat& blurred);
