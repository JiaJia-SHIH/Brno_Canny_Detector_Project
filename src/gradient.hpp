#pragma once
#include <opencv2/core.hpp>

struct GradientResult{
    cv::Mat magnitude;
    cv::Mat orientation;
};

GradientResult computeGradient(const cv::Mat& blurred);
