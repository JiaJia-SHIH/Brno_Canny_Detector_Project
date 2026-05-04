#pragma once
#include <opencv2/opencv.hpp>

enum EdgeClass : uchar {
    SUPPRESSED = 0,
    WEAK = 128,
    STRONG = 255
};


// Manual threaholding (original)
cv::Mat doubleThreshold(const cv::Mat& nms, float lowRatio = 0.05f, float highRatio = 0.15f);

// Auto-thresholding based on gradient distribution percentiles
cv::Mat autoThreshold(const cv::Mat& nms);

// Get the auto-computed threahold values
std::pair<float, float> getAutoThresholdValues(const cv::Mat& nms);