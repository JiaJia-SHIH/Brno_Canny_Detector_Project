/**
 * @file nms.hpp 
 * @author SHIH YUE JIA (xshihyu00)
 * @brief Non-maximum suppression to thin the edges
 **/

#pragma once
#include <opencv2/opencv.hpp>
#include "gradient.hpp"

// implemente non-maximum suppression
cv::Mat nonMaximumSuppression(const GradientResult& gradient);