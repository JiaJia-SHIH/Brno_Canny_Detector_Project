#include "hysteresis.hpp"
#include <queue>

cv::Mat hysteresisTracking(const cv::Mat& thresholded)
{
    int rows = thresholded.rows;
    int cols = thresholded.cols;

    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_8U);
    std::queue<std::pair<int, int>> bfsQueue;

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(thresholded.at<uchar>(i, j) == STRONG)
            {
                dst.at<uchar>(i, j) = STRONG;
                bfsQueue.push({i, j});
            }
        }
    }

    const int di[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dj[8] = {1, 1, 1, 0, 0, -1, -1, -1};

    while(!bfsQueue.empty())
    {
        auto [i, j] = bfsQueue.front();
        bfsQueue.pop();

        for(int d = 0; d < 8; d++)
        {
            int ni = i + di[d];
            int nj = j + dj[d];

            if(ni < 0 || ni >= rows || nj < 0 || nj >= cols) continue;
            if(thresholded.at<uchar>(ni, nj) != WEAK) continue;
            if(dst.at<uchar>(ni, nj) == 255) continue;

            if(thresholded.at<uchar>(ni, nj) == WEAK)
            {
                bfsQueue.push({ni, nj});
                dst.at<uchar>(ni, nj) = 255;
            }
        }
    }

    return dst;

}