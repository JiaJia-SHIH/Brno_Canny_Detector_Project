#pragma once
#include <opencv2/opencv.hpp>
#include "threshold.hpp"

cv::Mat hysteresisTracking(const cv::Mat& thresholded);