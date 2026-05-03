#include "nms.hpp"
#include <cmath>
#include <omp.h>

cv::Mat nonMaximumSuppression(const GradientResult& gradient) 
{
    const cv::Mat& magnitude = gradient.magnitude;
    const cv::Mat& orientation = gradient.orientation;
    int rows = magnitude.rows;
    int cols = magnitude.cols;

    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_32F);

    #pragma omp parallel for
    for(int j = 0; j < rows; j++)
    {
        for(int i = 0; i < cols; i++)
        {
            int neighbor1 = 0;
            int neighbor2 = 0;

            float angle = orientation.at<float>(j, i);
            if(angle >= 180.f) angle -= 180.f;

            if(angle < 22.5f || angle >= 157.5) // 0 degrees
            {
                neighbor1 = magnitude.at<float>(j, i-1);
                neighbor2 = magnitude.at<float>(j, i+1);
            }
            else if(angle < 67.5f) // 45 degrees
            {
                neighbor1 = magnitude.at<float>(j+1, i+1);
                neighbor2 = magnitude.at<float>(j-1, i-1);
            }
            else if(angle < 112.5f) // 90 degrees
            {
                neighbor1 = magnitude.at<float>(j+1, i);
                neighbor2 = magnitude.at<float>(j-1, i);
            }
            else if (angle < 157.5f) // 135 degrees
            {
                neighbor1 = magnitude.at<float>(j-1, i-1);
                neighbor2 = magnitude.at<float>(j+1, i+1);
            }
            
            float currentMagnitude = magnitude.at<float>(j, i);
            if(currentMagnitude >= neighbor1 && currentMagnitude >= neighbor2)
            {
                dst.at<float>(j, i) = currentMagnitude;
            }
        }
    }
    return dst;
}