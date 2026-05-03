#pragma once
#include <opencv2/opencv.hpp>
#include "gradient.hpp"

cv::Mat nonMaximumSuppression(const GradientResult& gradient);