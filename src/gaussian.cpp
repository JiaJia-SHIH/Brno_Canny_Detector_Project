/**
 * @file gaussian.cpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief Custom implementation of Gaussian Blur. OpenMP parallelization is used to improve performance.
 **/


#include "gaussian.hpp"
#include <cmath>
#include <omp.h>

using namespace std;

const int TILE_SIZE = 64; // Tiling size

// 1D kernel to save time
vector<float> makeGaussianKernel(int radius, float sigma)
{
    int size = 2 * radius + 1;
    vector<float> kernel(size);
    float sum = 0.0f;

    // kernel creation
    for(int i = 0; i < size; i++)
    {
        int x = i - radius;
        kernel[i] = exp(- (x*x) / (2*sigma*sigma));
        sum += kernel[i];
    }

    // normalize the kernel to make the sum equal to 1 (important for preserving brightness)
    for(auto& v : kernel)
    {
        v /= sum;
    }
    return kernel;
}

// Convolution - horizontal
static cv::Mat convolveH(const cv::Mat& src, const vector<float>& kernel)
{
    int radius = kernel.size() / 2;
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32F);

    // [Parallel Area]
    // [Tilig Area] use tiling to avoid cache thrashing 
    #pragma omp parallel for collapse(2)
    for(int ty = 0; ty < src.rows; ty += TILE_SIZE)
    {
        for(int tx = 0; tx < src.cols; tx += TILE_SIZE)
        {
            int y_end = min(ty + TILE_SIZE, src.rows);
            int x_end = min(tx + TILE_SIZE, src.cols);
            
            for(int y = ty; y < y_end; y++)
            {
                for(int x = tx; x < x_end; x++)
                {
                    float sum = 0.0f;
                    for(int offset = -radius; offset <= radius; offset++)
                    {
                        int nx = clamp(x + offset, 0, src.cols - 1);
                        sum += src.at<float>(y, nx) * kernel[offset + radius];
                    }
                    dst.at<float>(y, x) = sum;
                } // x
            } // y
        } // tx
    } // ty
    return dst;
}

// Convolution - Vertical
static cv::Mat convolveV(const cv::Mat& src, const vector<float>& kernel)
{
    int radius = kernel.size() / 2;
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32F);

    // [Parallel Area]
    // [Tilig Area] use tiling to avoid cache thrashing 
    #pragma omp parallel for collapse(2)
    for(int ty = 0; ty < src.rows; ty += TILE_SIZE)
    {
        for(int tx = 0; tx < src.cols; tx += TILE_SIZE)
        {
            int y_end = min(ty + TILE_SIZE, src.rows);
            int x_end = min(tx + TILE_SIZE, src.cols);
            
            for(int y = ty; y < y_end; y++)
            {
                for(int x = tx; x < x_end; x++)
                {
                    float sum = 0.0f;
                    for(int offset = -radius; offset <= radius; offset++)
                    {
                        int ny = clamp(y + offset, 0, src.rows - 1);

                        sum += src.at<float>(ny, x) * kernel[offset + radius];
                    }
                    dst.at<float>(y, x) = sum;
                }
            }
        }
    }
    return dst;
}

// implemente Gaussian Blur
cv::Mat gaussianBlur(const cv::Mat& gray, int radius, float sigma)
{
    cv::Mat src32;
    gray.convertTo(src32, CV_32F);

    auto kernel = makeGaussianKernel(radius, sigma);
    cv::Mat temp = convolveH(src32, kernel);
    cv::Mat blurred = convolveV(temp, kernel);

    return blurred;
}