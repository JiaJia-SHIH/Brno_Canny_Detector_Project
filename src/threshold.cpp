#include "threshold.hpp"
#include <algorithm>

cv::Mat doubleThreshold(const cv::Mat& nms, float lowRatio, float highRatio)
{
    // adaptively find the maximum value in the nms image
    float maxVal = 0.0f;
    for(int i = 0; i < nms.rows; i++)
    {
        for(int j = 0; j < nms.cols; j++)
        {
            maxVal = std::max(maxVal, nms.at<float>(i, j));
        }
    }

    float highThresholdVal = maxVal * highRatio;
    float lowThresholdVal = maxVal * lowRatio;

    cv::Mat dst = cv::Mat::zeros(nms.size(), CV_8U);

    for(int i = 0; i < nms.rows; i++)
    {
        for(int j = 0; j < nms.cols; j++)
        {
            float val = nms.at<float>(i, j);
            if(val >= highThresholdVal)
            {
                dst.at<uchar>(i, j) = STRONG;
            }
            else if( val >= lowThresholdVal)
            {
                dst.at<uchar>(i, j) = WEAK;
            }
        }
    }
    return dst;
}