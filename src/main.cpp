#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.hpp"
#include "gaussian.hpp"
#include "gradient.hpp"
#include "nms.hpp"
#include "threshold.hpp"
#include "hysteresis.hpp"
#include "server.hpp"

using namespace std;

int main(int argc, char** argv)
{
    string imgPath = (argc > 1) ? argv[1] : "../src/img/lenna.jpg";
    cv::Mat bgr = cv::imread(imgPath);
    if(bgr.empty())    {
        cerr << "Error: Could not load image at " << imgPath << endl;
        return -1;
    }

    // Convert to grayscale
    cv::Mat gray = bgrToGrayManually(bgr);

    // Parameter Setting
    int radius = 2; // kernel radius
    float sigma = 1.4f; // standard deviation for Gaussian

    // 1. Gaussian Blur
    cv::Mat blurredResult = gaussianBlur(gray, radius, sigma);
    cv::Mat blurredDisplay = normalizeManually(blurredResult);
    
    // Display results
    cv::imwrite("../eval/01_gaussian_result.jpg", blurredDisplay);

    // 2. Sobel Operation
    // -- Gradient
    GradientResult gradientResult = computeGradient(blurredResult);
    cv::Mat magnitudeDisplay = normalizeManually(gradientResult.magnitude);
    
    // Display results
    cv::imwrite("../eval/02_magnitude_result.jpg", magnitudeDisplay);
    
    // -- Orientation
    cv::Mat orientationDisplay(gray.size(), CV_8UC3);
    orientationDisplay = orientationToColorManually(gradientResult.orientation, magnitudeDisplay);

    cv::imwrite("../eval/02_orientation_result.jpg", orientationDisplay);

    // 3. Non-Maximum Suppression
    cv::Mat nmsResult = nonMaximumSuppression(gradientResult);

    // Display results
    cv::Mat nmsDisplay = normalizeManually(nmsResult);
    cv::imwrite("../eval/03_nms_result.jpg", nmsDisplay);

    // 4. Double Thresholding
    cv::Mat thresholdResult = doubleThreshold(nmsResult);

    cv::Mat thresh_color(thresholdResult.size(), CV_8UC3);
    for (int y = 0; y < thresholdResult.rows; y++)
    {
        for (int x = 0; x < thresholdResult.cols; x++) {
            uchar v = thresholdResult.at<uchar>(y, x);
            if      (v == STRONG) thresh_color.at<cv::Vec3b>(y,x) = {255,255,255};
            else if (v == WEAK)   thresh_color.at<cv::Vec3b>(y,x) = {0,128,255};
            else                  thresh_color.at<cv::Vec3b>(y,x) = {0,0,0};
        }
    }

    // Display results
    cv::imwrite("../eval/04_threshold_result.jpg", thresh_color);

    // 5. Hysteresis Tracking
    cv::Mat hysteresisResult = hysteresisTracking(thresholdResult);

    // Display results
    cv::imwrite("../eval/05_hysteresis_result.jpg", hysteresisResult);

    runServer(imgPath, 9090);

    return 0;
}