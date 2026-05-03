#pragma once
#include <opencv2/opencv.hpp>

cv::Mat normalizeManually(const cv::Mat& input);
cv::Mat hconcatManually(const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b);
cv::Mat grayToBgrManually(const cv::Mat& gray);
cv::Mat bgrToGrayManually(const cv::Mat& bgr);
cv::Mat orientationToColorManually(const cv::Mat& orientation, const cv::Mat& magnitude);