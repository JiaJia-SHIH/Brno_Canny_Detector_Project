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

    // 4-1. (Manually) Double Thresholding
    cout << "----------- Manually thresholding -----------";
    cv::Mat thresholdResult = doubleThreshold(nmsResult);

    cv::Mat thresh_color(thresholdResult.size(), CV_8UC3);
    for (int y = 0; y < thresholdResult.rows; y++)
    {
        for (int x = 0; x < thresholdResult.cols; x++) {
            uchar v = thresholdResult.at<uchar>(y, x);
            if(v == STRONG) thresh_color.at<cv::Vec3b>(y,x) = {255,255,255};
            else if (v == WEAK) thresh_color.at<cv::Vec3b>(y,x) = {0,128,255};
            else thresh_color.at<cv::Vec3b>(y,x) = {0,0,0};
        }
    }

    // Display results
    cv::imwrite("../eval/04_threshold_result.jpg", thresh_color);

    // 4-2. (Auto) Double Thresholding by percentile
    cout << "----------- Double Thresholding -----------" << endl;
    cv::Mat thresholdAutoResult = autoThreshold(nmsResult);
    auto [autoLow, autoHigh] = getAutoThresholdValues(nmsResult);
    cout << "Auto computed thresholds: low=" << autoLow << ", high=" << autoHigh << endl;
    cv::Mat thresh_auto_color(thresholdAutoResult.size(), CV_8UC3);
    for (int y = 0; y < thresholdAutoResult.rows; y++)
    {
        for (int x = 0; x < thresholdAutoResult.cols; x++) {
            uchar v = thresholdAutoResult.at<uchar>(y, x);
            if(v == STRONG) thresh_auto_color.at<cv::Vec3b>(y,x) = {255,255,255};
            else if (v == WEAK) thresh_auto_color.at<cv::Vec3b>(y,x) = {0,128,255};
            else thresh_auto_color.at<cv::Vec3b>(y,x) = {0,0,0};
        }
    }

    cv::imwrite("../eval/04_threshold_auto.jpg", thresh_auto_color);

    // 4-Compare. Comparison visualization
    cv::Mat comparison(thresh_color.rows, thresh_color.cols * 2, CV_8UC3);
    for(int i = 0; i < thresh_color.rows; i++)
    {
        for(int j = 0; j < thresh_color.cols; j++)
        {
            comparison.at<cv::Vec3b>(i, j) = thresh_color.at<cv::Vec3b>(i, j);
            comparison.at<cv::Vec3b>(i, j + thresh_color.cols) = thresh_auto_color.at<cv::Vec3b>(i, j);
        }
    }
    cv::imwrite("../eval/04_threshold_comparison.jpg", comparison);
    cout << "Comparison saved: ../eval/04_threshold_comparison.jpg (Left: Manual, Right: Auto)" << endl;

    // 4-Histogram. Visualize histogram with auto thresholds
    cout << "\n--- Histogram Visualization ---" << endl;
    cv::Mat histImage = visualizeHistogramWithThresholds(nmsResult, autoLow, autoHigh);
    if(!histImage.empty())
    {
        cv::imwrite("../eval/04_threshold_histogram.jpg", histImage);
        cout << "Histogram saved: ../eval/04_threshold_histogram.jpg" << endl;
    }

    // 5. Hysteresis Tracking
    cv::Mat hysteresisResult = hysteresisTracking(thresholdResult);

    // Display results
    cv::imwrite("../eval/05_hysteresis_result.jpg", hysteresisResult);

    runServer(imgPath, 9090);

    return 0;
}