#include "Evaluation/evaluate.hpp"
#include "utils.hpp"
#include "gaussian.hpp"
#include "gradient.hpp"
#include "nms.hpp"
#include "threshold.hpp"
#include "hysteresis.hpp"
 
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace fs = filesystem;
using namespace std;

BSDS500Evaluator::BSDS500Evaluator(const string& bsds_root, float tolerance_ratio) : bsds_root_(bsds_root), tolerance_ratio_(tolerance_ratio){}

vector<cv::Mat> BSDS500Evaluator::loadGroundTruth(const string& image_name)
{
    vector<cv::Mat> ground_truths;
    string gt_dir = bsds_root_ + "/data/groundTruth/test_png";
    
    // find all ground_truth for this image
    string base_name = image_name.substr(0, image_name.find_last_of('.'));

    for(const auto& entry : fs::directory_iterator(gt_dir))
    {
        string filename = entry.path().filename().string();

        if(filename.find(base_name + "_gt") == 0 && entry.path().extension() == ".png")
        {
            cv::Mat gt = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if(!gt.empty())
            {
                // Convert to binary: edges should be 255, background should be 0
                cv::Mat binary;
                cv::threshold(gt, binary, 127, 255, cv::THRESH_BINARY);
                ground_truths.push_back(binary);
            }
        }
    }

    return ground_truths;
}

void BSDS500Evaluator::computeMetrics(const cv::Mat& detected,  const cv::Mat& ground_truths, float tolerance, float& precision, float& recall, float& f1)
{
    int rows = detected.rows;
    int cols = detected.cols;
    int num_detected = 0;
    cv::Mat matched_gt = cv::Mat::zeros(rows, cols, CV_8U);

    // count detected edges pixels
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(detected.at<uchar>(i, j) > 0) num_detected++;
        }
    }

    // count groundtruth edges pixels
    int num_gt = 0;
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(ground_truths.at<uchar>(i, j) > 0) num_gt++;
        }
    }

    // if no edges, there is no need to compare
    if(num_detected == 0 || num_gt == 0)
    {
        precision = 0.0f;
        recall = 0.0f;
        f1 = 0.0f;
        return;
    }

    // count true positives: detected edges near groundtruth
    int true_positives = 0;
    int tol = static_cast<int>(ceil(tolerance));

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(detected.at<uchar>(i ,j) == 0) continue;

            bool found = false;
            for(int di = -tol; di <= tol && !found; di++)
            {
                for(int dj = -tol; dj <= tol && !found; dj++)
                {
                    int ni = i + di;
                    int nj = j + dj;

                    if(ni < 0 || ni >= rows || nj < 0 || nj >= cols) continue;

                    if(ground_truths.at<uchar>(ni, nj) > 0 && matched_gt.at<uchar>(ni, nj) == 0)
                    {
                        float dist = sqrt(di*di + dj*dj);
                        if(dist <= tolerance)
                        {
                            found = true;
                            matched_gt.at<uchar>(ni, nj) = 255;
                        }
                    }
                }
            }

            if(found) true_positives++;
        }
    }
    // calculate Metrics
    precision = static_cast<float>(true_positives) / num_detected;
    recall = static_cast<float>(true_positives) / num_gt;

    if(precision + recall > 0)
    {
        f1 = 2.0f * precision * recall / (precision + recall);
    }
    else
    {
        f1 = 0.0f;
    }
}

void BSDS500Evaluator::bestMatch(const cv::Mat& detected, const vector<cv::Mat>& ground_truths, float tolerance, float& avg_precision, float& avg_recall, float& avg_f1)
{
    avg_precision = 0.0f;
    avg_recall = 0.0f;
    avg_f1 = 0.0f;

    if(ground_truths.empty()) return;

    // to be fair, I compute metrics for each annotation and average
    for(const auto& gt : ground_truths)
    {
        float p, r, f;
        computeMetrics(detected, gt, tolerance, p, r, f);

        avg_precision += p;
        avg_recall += r;
        avg_f1 += f;
    }

    int num_annotations = ground_truths.size();
    avg_precision /= num_annotations;
    avg_recall /= num_annotations;
    avg_f1 /= num_annotations;
}

ImageEvalResult BSDS500Evaluator::evaluateImage(const cv::Mat& detected_edges, const string& image_name)
{
    ImageEvalResult result;
    result.image_name = image_name;

    auto ground_truths = loadGroundTruth(image_name);
    result.num_annotations = ground_truths.size();

    
    if(ground_truths.empty())
    {
        cerr << "Warning: No ground truth found for " << image_name << endl;
        result.precision = 0.0f;
        result.recall = 0.0f;
        result.f1_score = 0.0f;
        return result;
    }

    // compute toleracne based on image size
    float tolerance = tolerance_ratio_ * max(detected_edges.rows, detected_edges.cols);

    // compute average metrics
    bestMatch(detected_edges, ground_truths, tolerance, result.precision, result.recall, result.f1_score);

    return result;
}

DatasetEvalResult BSDS500Evaluator::evaluateTestSet(int radius, float sigma, float low_ratio, float high_ratio)
{
    DatasetEvalResult dataset_result;
    string test_dir = bsds_root_ + "/data/images/test";
    vector<string> image_files;
    
    for(const auto& entry : fs::directory_iterator(test_dir))
    {
        if(entry.path().extension() == ".jpg")
        {
            image_files.push_back(entry.path().filename().string());
        }
    }

    sort(image_files.begin(), image_files.end());
    cout << "Evaluating " << image_files.size() << " test images..." << endl;
    cout << "Parameters: radius=" << radius << ", sigma=" << sigma << ", low=" << low_ratio << ", high=" << high_ratio << endl;
    cout << endl;

    // Process each image
    for(const auto& img_file : image_files)
    {
        string img_path = test_dir + "/" + img_file;
        cv::Mat bgr = cv::imread(img_path);
        
        if(bgr.empty())
        {
            cerr << "Failed to load: " << img_file << endl;
            continue;
        }
        
        // run Canny pipeline
        cv::Mat gray = bgrToGrayManually(bgr);
        cv::Mat blurred = gaussianBlur(gray, radius, sigma);
        GradientResult gradient = computeGradient(blurred);
        cv::Mat nms = nonMaximumSuppression(gradient);
        cv::Mat thresholded = doubleThreshold(nms, low_ratio, high_ratio);
        cv::Mat edges = hysteresisTracking(thresholded);
        
        // evaluate against ground truth
        ImageEvalResult img_result = evaluateImage(edges, img_file);
        dataset_result.per_image_results.push_back(img_result);
        
        cout << img_file << ": "
                  << "P=" << img_result.precision << ", "
                  << "R=" << img_result.recall << ", "
                  << "F1=" << img_result.f1_score << " "
                  << "(" << img_result.num_annotations << " annotations)" 
                  << endl;
    }

    // Compute dataset averages
    dataset_result.total_images = dataset_result.per_image_results.size();
    
    float sum_p = 0.0f, sum_r = 0.0f, sum_f1 = 0.0f;
    for(const auto& res : dataset_result.per_image_results)
    {
        sum_p += res.precision;
        sum_r += res.recall;
        sum_f1 += res.f1_score;
    }
    
    if(dataset_result.total_images > 0)
    {
        dataset_result.mean_precision = sum_p / dataset_result.total_images;
        dataset_result.mean_recall = sum_r / dataset_result.total_images;
        dataset_result.mean_f1 = sum_f1 / dataset_result.total_images;
    }
    
    return dataset_result;
}

DatasetEvalResult BSDS500Evaluator::evaluateTestSetAuto(int radius, float sigma)
{
    DatasetEvalResult dataset_result;
    string test_dir = bsds_root_ + "/data/images/test";
    vector<string> image_files;
    
    for(const auto& entry : fs::directory_iterator(test_dir))
    {
        if(entry.path().extension() == ".jpg")
        {
            image_files.push_back(entry.path().filename().string());
        }
    }

    sort(image_files.begin(), image_files.end());
    cout << "Evaluating " << image_files.size() << " test images with AUTO threshold..." << endl;
    cout << "Parameters: radius=" << radius << ", sigma=" << sigma << endl;
    cout << endl;

    // Process each image
    for(const auto& img_file : image_files)
    {
        string img_path = test_dir + "/" + img_file;
        cv::Mat bgr = cv::imread(img_path);
        
        if(bgr.empty())
        {
            cerr << "Failed to load: " << img_file << endl;
            continue;
        }
        
        // run Canny pipeline with auto threshold
        cv::Mat gray = bgrToGrayManually(bgr);
        cv::Mat blurred = gaussianBlur(gray, radius, sigma);
        GradientResult gradient = computeGradient(blurred);
        cv::Mat nms = nonMaximumSuppression(gradient);
        cv::Mat thresholded = autoThreshold(nms);
        cv::Mat edges = hysteresisTracking(thresholded);
        
        // evaluate against ground truth
        ImageEvalResult img_result = evaluateImage(edges, img_file);
        dataset_result.per_image_results.push_back(img_result);
        
        cout << img_file << ": "
                  << "P=" << img_result.precision << ", "
                  << "R=" << img_result.recall << ", "
                  << "F1=" << img_result.f1_score << " "
                  << "(" << img_result.num_annotations << " annotations)" 
                  << endl;
    }

    // Compute dataset averages
    dataset_result.total_images = dataset_result.per_image_results.size();
    
    float sum_p = 0.0f, sum_r = 0.0f, sum_f1 = 0.0f;
    for(const auto& res : dataset_result.per_image_results)
    {
        sum_p += res.precision;
        sum_r += res.recall;
        sum_f1 += res.f1_score;
    }
    
    if(dataset_result.total_images > 0)
    {
        dataset_result.mean_precision = sum_p / dataset_result.total_images;
        dataset_result.mean_recall = sum_r / dataset_result.total_images;
        dataset_result.mean_f1 = sum_f1 / dataset_result.total_images;
    }
    
    return dataset_result;
}

ComparisonResult BSDS500Evaluator::compareThresholdMethods(int radius, float sigma, float low_ratio, float high_ratio)
{
    ComparisonResult comparison;
    
    cout << "------------------------------" << endl;
    cout << "MANUAL THRESHOLD EVALUATION" << endl;
    cout << "------------------------------" << endl;
    comparison.manual_result = evaluateTestSet(radius, sigma, low_ratio, high_ratio);
    
    cout << endl;
    cout << "------------------------------" << endl;
    cout << "AUTO THRESHOLD EVALUATION" << endl;
    cout << "------------------------------" << endl;
    comparison.auto_result = evaluateTestSetAuto(radius, sigma);
    
    return comparison;
}

void BSDS500Evaluator::saveResults(const DatasetEvalResult& result, const string& output_path)
{
    ofstream file(output_path);
    
    file << "image_name,precision,recall,f1_score,num_annotations\n";
    
    for(const auto& img_res : result.per_image_results)
    {
        file << img_res.image_name << ","
             << img_res.precision << ","
             << img_res.recall << ","
             << img_res.f1_score << ","
             << img_res.num_annotations << "\n";
    }
    
    file << "\nMean," 
         << result.mean_precision << ","
         << result.mean_recall << ","
         << result.mean_f1 << ","
         << result.total_images << "\n";
    
    file.close();
    
    cout << "\nResults saved to: " << output_path << endl;
}

void BSDS500Evaluator::saveComparisonResults(const ComparisonResult& result, const string& output_path)
{
    ofstream file(output_path);
    
    file << "image_name,manual_precision,manual_recall,manual_f1,auto_precision,auto_recall,auto_f1,num_annotations\n";
    
    for(size_t i = 0; i < result.manual_result.per_image_results.size(); i++)
    {
        const auto& manual = result.manual_result.per_image_results[i];
        const auto& automatic = result.auto_result.per_image_results[i];
        
        file << manual.image_name << ","
             << manual.precision << ","
             << manual.recall << ","
             << manual.f1_score << ","
             << automatic.precision << ","
             << automatic.recall << ","
             << automatic.f1_score << ","
             << manual.num_annotations << "\n";
    }
    
    file << "\nMean,"
         << result.manual_result.mean_precision << ","
         << result.manual_result.mean_recall << ","
         << result.manual_result.mean_f1 << ","
         << result.auto_result.mean_precision << ","
         << result.auto_result.mean_recall << ","
         << result.auto_result.mean_f1 << ","
         << result.manual_result.total_images << "\n";
    
    file.close();
    
    cout << "\nComparison results saved to: " << output_path << endl;
}