#include "threshold.hpp"
#include <algorithm>
#include <vector>

using namespace std;

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

static float computePercentile(const cv::Mat& nms, float percentile)
{
    // collect all non-zero gradient value
    vector<float> values;
    values.reserve(nms.rows * nms.cols / 10); // to reduce the memory reallocation

    for(int i = 0; i < nms.rows; i++)
    {
        for(int j = 0; j < nms.cols; j++)
        {
            float val = nms.at<float>(i, j);
            if(val > 0.0f)
            {
                values.push_back(val);
            }
        }
    }

    if(values.empty()) return 0.0f;

    sort(values.begin(), values.end());

    size_t index = static_cast<size_t>(percentile * values.size());
    if(index >= values.size()) index = values.size() - 1;

    return values[index];
}

pair<float, float> getAutoThresholdValues(const cv::Mat& nms)
{
    // ** Use 90% to be the high threshold
    // ** Use 50% to be the low threshold
    float lowThreshold = computePercentile(nms, 0.20f);
    float highThreshold = computePercentile(nms, 0.70f);

    return {lowThreshold, highThreshold};
}

cv::Mat autoThreshold(const cv::Mat& nms)
{
    auto[lowThreshold, highThreshold] = getAutoThresholdValues(nms);

    cv::Mat dst = cv::Mat::zeros(nms.size(), CV_8U);

    for(int i = 0; i < nms.rows; i++)
    {
        for(int j = 0; j < nms.cols; j++)
        {
            float val = nms.at<float>(i, j);
            if(val >= highThreshold)
            {
                dst.at<uchar>(i, j) = STRONG;
            }
            else if(val >= lowThreshold)
            {
                dst.at<uchar>(i, j) = WEAK;
            }
        }
    }

    return dst;
}