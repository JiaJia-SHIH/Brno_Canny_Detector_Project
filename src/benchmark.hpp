#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>
 
using namespace std;

struct BenchmarkConfig
{
    string bsds500_path;
    vector<string> test_images; // list of the testing name
    vector<int> scale_factors; // ex: 1x, 2x, 4x, ...
    vector<int> thread_counts;
    int num_runs; // to avoid single decision

    // fixed parameters
    int radius = 2;
    int sigma = 1.4f;
    float low_ratio = 0.05f;
    float high_ratio = 0.15f;

    // methods
    bool benchmark_opencv = true;
    bool benchmark_serial = true;
    bool benchmark_parallel = true;
};

struct BenchmarkResult
{
    // performance
    string method_name;
    string image_name;
    int image_width;
    int image_height;
    double execution_time_ms;  // milliseconds
    int thread_count;  // the numbers of threads

    // parameters used
    int radius;
    int sigma;
    float low_ratio;
    float high_ratio;
};

class BenchmarkRunner
{
public:
    BenchmarkRunner(const BenchmarkConfig& config);
    vector<BenchmarkResult> runAll();
    void saveResultsToCSV(const vector<BenchmarkResult>& results, const string& filename);

private:
    BenchmarkConfig config_;
    vector<cv::Mat> test_images_;
    vector<string> image_names_;

    // implementation
    void loadTestImages();
    BenchmarkResult benchmarkOpenCV(const cv::Mat& img, const string& img_name);
    BenchmarkResult benchmarkSerial(const cv::Mat& img, const string& img_name);
    BenchmarkResult benchmarkParallel(const cv::Mat& img, const string& img_name, int num_threads);

    // helper
    template<typename Func>
    double measureTime(Func func, int num_runs);
};

