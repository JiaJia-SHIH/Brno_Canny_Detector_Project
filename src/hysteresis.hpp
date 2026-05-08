/**
 * @file hysteresis.hpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief Hysteresis edge tracking by BFS
 **/

#pragma once
#include <opencv2/opencv.hpp>
#include "threshold.hpp"

// implement the hysteresis edge tracking
cv::Mat hysteresisTracking(const cv::Mat& thresholded);