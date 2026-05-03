#include "utils.hpp"
#include <algorithm>

cv::Mat normalizeManually(const cv::Mat& src)
{
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();

    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            float v = src.at<float>(i, j);
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
    }

    float range = maxVal - minVal;
    cv::Mat dst(src.size(), CV_8U);

    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            float v = src.at<float>(i, j);
            dst.at<uchar>(i, j) = (range > 0) ? (uchar)(255.f * (v - minVal) / range) : 0;
        }
    }
    return dst;
}

cv::Mat hconcatManually(const cv::Mat& a, const cv::Mat& b)
{
    cv::Mat dst(a.rows, a.cols + b.cols, a.type(), cv::Scalar(0));
    for(int i = 0; i < a.rows; i++)
    {
        for(int j = 0; j < a.cols; j++)
        {
            dst.at<cv::Vec3d>(i, j) = a.at<cv::Vec3b>(i, j);
        }
        for(int j = 0; j < b.cols; j++)
        {
            dst.at<cv::Vec3b>(i, a.cols + j) = b.at<cv::Vec3b>(i, j);
        }
    }
    return dst;
}

cv::Mat grayToBgrManually(const cv::Mat& gray)
{
    cv::Mat dst(gray.size(), CV_8UC3);
    for(int i = 0; i < gray.rows; i++)
    {
        for(int j = 0; j < gray.cols; j++)
        {
            uchar v = gray.at<uchar>(i, j);
            dst.at<cv::Vec3b>(i, j) = {v, v, v};
        }
    }

    return dst;
}

cv::Mat bgrToGrayManually(const cv::Mat& bgr)
{
    cv::Mat dst(bgr.size(), CV_8U);
    for (int y = 0; y < bgr.rows; y++)
        for (int x = 0; x < bgr.cols; x++) {
            auto& p = bgr.at<cv::Vec3b>(y, x);
            dst.at<uchar>(y, x) =
                (uchar)(0.114f*p[0] + 0.587f*p[1] + 0.299f*p[2]);
        }
    return dst;
}

cv::Mat orientationToColorManually(const cv::Mat& orientation, const cv::Mat& mag8) {
    cv::Mat dst(orientation.size(), CV_8UC3);

    for (int y = 0; y < orientation.rows; y++) {
        for (int x = 0; x < orientation.cols; x++) {
            float angle = orientation.at<float>(y, x);
            if (angle < 0) angle += 360.f;
            float h = angle / 2.f;
            float s = 1.f;
            float v = mag8.at<uchar>(y, x) / 255.f;

            int   hi = (int)(h / 60.f) % 6;
            float f  = h / 60.f - (int)(h / 60.f);
            float p  = v * (1 - s);
            float q  = v * (1 - f * s);
            float t  = v * (1 - (1 - f) * s);

            float r, g, b;
            switch (hi) {
                case 0: r=v; g=t; b=p; break;
                case 1: r=q; g=v; b=p; break;
                case 2: r=p; g=v; b=t; break;
                case 3: r=p; g=q; b=v; break;
                case 4: r=t; g=p; b=v; break;
                default:r=v; g=p; b=q; break;
            }
            dst.at<cv::Vec3b>(y, x) = {
                (uchar)(b*255), (uchar)(g*255), (uchar)(r*255)};
        }
    }
    return dst;
}