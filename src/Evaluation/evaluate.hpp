/**
 * @file evaluate.hpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief BSDS500 dataset evaluator with precision/recall/F1
 **/

#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

using namespace std;

struct ImageEvalResult
{
    string image_name;
    float precision;
    float recall;
    float f1_score;
    int num_annotations; // in bsds500 dataset, there are more than 1 person to annote the edge for each image, I note it and use all version
};

struct DatasetEvalResult
{
    vector<ImageEvalResult> per_image_results;
    float mean_precision;
    float mean_recall;
    float mean_f1;
    int total_images;
};

struct ComparisonResult
{
    DatasetEvalResult manual_result;
    DatasetEvalResult auto_result;
};

class BSDS500Evaluator
{
public:
    BSDS500Evaluator(const string& bsds_root, float tolerance_ratio = 0.0075f);

    // evaluate single image against all its ground truth annotations
    ImageEvalResult evaluateImage(const cv::Mat& detected_edges, const string& image_name);
    DatasetEvalResult evaluateTestSet(int radius = 2, float sigma = 1.4f, float low_ratio = 0.05f, float high_ratio = 0.15f);
    DatasetEvalResult evaluateTestSetAuto(int radius = 2, float sigma = 1.4f);
    ComparisonResult compareThresholdMethods(int radius = 2, float sigma = 1.4f, float low_ratio = 0.05f, float high_ratio = 0.15f);

    // save result
    void saveResults(const DatasetEvalResult& result, const string& output_path);
    void saveComparisonResults(const ComparisonResult& result, const string& output_path);

private:
    string bsds_root_;
    float tolerance_ratio_;

    // load all ground truth annotations for one image
    vector<cv::Mat> loadGroundTruth(const string& image_name);

    void computeMetrics(const cv::Mat& detected, const cv::Mat& ground_truths, float tolerance, float& precision, float& recall, float& f1);
    void bestMatch(const cv::Mat& detected, const vector<cv::Mat>& ground_truths, float tolerance, float& avg_precision, float& avg_recall, float& avg_f1);
};