#pragma once
#include <opencv2/opencv.hpp>

enum EdgeClass : uchar {
    SUPPRESSED = 0,
    WEAK = 128,
    STRONG = 255
};

cv::Mat doubleThreshold(const cv::Mat& nms, float lowRatio = 0.05f, float highRatio = 0.15f);