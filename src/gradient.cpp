#include "gradient.hpp"
#include <cmath>
#include <omp.h>

GradientResult computeGradient(const cv::Mat& blurred)
{
    int rows = blurred.rows;
    int cols = blurred.cols;

    cv::Mat Gx = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat Gy = cv::Mat::zeros(rows, cols, CV_32F);
    GradientResult result;
    result.magnitude = cv::Mat::zeros(rows, cols, CV_32F);
    result.orientation = cv::Mat::zeros(rows, cols, CV_32F);
    
    #pragma omp parallel for
    for(int i = 1; i < rows-1; i++)
    {
        for(int j = 1; j < cols-1; j++)
        {
            float top_left = blurred.at<float>(i-1, j-1);
            float top_middle = blurred.at<float>(i-1, j);
            float top_right = blurred.at<float>(i-1, j+1);

            float middle_left = blurred.at<float>(i, j-1);
            float middle_right = blurred.at<float>(i, j+1);

            float bottom_left = blurred.at<float>(i+1, j-1);
            float bottom_middle = blurred.at<float>(i+1, j);
            float bottom_right = blurred.at<float>(i+1, j+1);
            
            // sobel operator
            // [-1, 0, 1]
            // [-2, 0, 2]
            // [-1, 0, 1]
            float Gx_value = -top_left + top_right - 2*middle_left + 2*middle_right - bottom_left + bottom_right;
            float Gy_value = -top_left - 2*top_middle - top_right + bottom_left + 2*bottom_middle + bottom_right;
            
            Gx.at<float>(i, j) = Gx_value;
            Gy.at<float>(i, j) = Gy_value;

            result.magnitude.at<float>(i, j) = std::sqrt(Gx_value * Gx_value + Gy_value * Gy_value);
            float angle = std::atan2(Gy_value, Gx_value) * 180.0 / CV_PI;
            if(angle < 0)
                angle += 360;
            result.orientation.at<float>(i, j) = angle;
        }
    }

    return result;

}