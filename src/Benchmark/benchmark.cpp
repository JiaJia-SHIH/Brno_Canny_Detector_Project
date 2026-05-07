#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <omp.h>

#include "gaussian.hpp"
#include "gradient.hpp"
#include "nms.hpp"
#include "threshold.hpp"
#include "hysteresis.hpp"
#include "utils.hpp"
#include "benchmark.hpp"

using namespace std;
namespace fs = filesystem;

BenchmarkRunner::BenchmarkRunner(const BenchmarkConfig& config) : config_(config)
{
    loadTestImages();
}

void BenchmarkRunner::loadTestImages()
{
    string test_path = config_.bsds500_path + "/BSDS500/data/images/test";
    cout << "Loading the test images from: " << test_path << endl;
    // if no specitied imaged that have to be test(test_images is empty, test will run all images)
    if(config_.test_images.empty())
    {
        for(const auto& entry : fs::directory_iterator(test_path))
        {
            if(entry.path().extension() == ".jpg")
            {
                config_.test_images.push_back(entry.path().filename().string());
            }
        }

        // Limit to firat 10 images for reasonable benchmark time

        sort(config_.test_images.begin(), config_.test_images.end());
        if(config_.test_images.size() > 10)
        {
            config_.test_images.resize(10);
        }
    }

    // Load images
    for(const auto& img_name : config_.test_images)
    {
        string img_path = test_path + "/" + img_name;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        
        if(img.empty())
        {
            cerr << "Warning: Failed to load the img:" << img_name << endl;
            continue;
        }

        test_images_.push_back(img);
        image_names_.push_back(img_name);

        cout << "Loaded: " << img_name << "(" << img.cols << "x" << img.rows << ")" << endl;
    }

    cout << "Total images loaded: " << test_images_.size() << endl;
}

BenchmarkResult BenchmarkRunner::benchmarkSerial(const cv::Mat& img, const string& img_name)
{
    BenchmarkResult result;
    result.method_name = "Serial";
    result.image_name = img_name;
    result.image_width = img.cols;
    result.image_height = img.rows;
    result.thread_count = 1;

    result.radius = config_.radius;
    result.sigma = config_.sigma;
    result.low_ratio = config_.low_ratio;
    result.high_ratio = config_.high_ratio;

    // to avoid using more than 1 thread
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    auto pipeline = [&](){
        cv::Mat img32;
        img.convertTo(img32, CV_32F);

        // canny pipeline
        cv::Mat blurred = gaussianBlur(img32, config_.radius, config_.sigma);
        GradientResult gradient = computeGradient(blurred);
        cv::Mat nms = nonMaximumSuppression(gradient);
        cv::Mat thresholded = doubleThreshold(nms, config_.low_ratio, config_.high_ratio);
        cv::Mat edges = hysteresisTracking(thresholded);

        return edges;
    };

    result.execution_time_ms = measureTime(pipeline, config_.num_runs);

    return result;
}

BenchmarkResult BenchmarkRunner::benchmarkParallel(const cv::Mat& img, const string& img_name, int num_threads)
{
    BenchmarkResult result;
    result.method_name = "Parallel-" + to_string(num_threads);
    result.image_name = img_name;
    result.image_width = img.cols;
    result.image_height = img.rows;
    result.thread_count = num_threads;

    result.radius = config_.radius;
    result.sigma = config_.sigma;
    result.low_ratio = config_.low_ratio;
    result.high_ratio = config_.high_ratio;

    // check OpenCV environment
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    int actual_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    if(actual_threads != num_threads)
    {
        cerr << "Warning: Requested " << num_threads << " threads but got " << actual_threads << endl;
    }
    #else
    static bool warned = false;
    if(!warned) {
        cerr << "\n!!! WARNING: OpenMP is NOT enabled !!!" << endl;
        cerr << "Parallel benchmarks will run in SERIAL mode!" << endl;
        cerr << "Recompile with -fopenmp flag to enable parallelization." << endl;
        cerr << std::endl;
        warned = true;
    }
    #endif

    auto pipeline = [&](){
        cv::Mat img32;
        img.convertTo(img32, CV_32F);

        // canny pipeline
        cv::Mat blurred = gaussianBlur(img32, config_.radius, config_.sigma);
        GradientResult gradient = computeGradient(blurred);
        cv::Mat nms = nonMaximumSuppression(gradient);
        cv::Mat thresholded = doubleThreshold(nms, config_.low_ratio, config_.high_ratio);
        cv::Mat edges = hysteresisTracking(thresholded);

        return edges;
    };

    result.execution_time_ms = measureTime(pipeline, config_.num_runs);

    return result;
}

BenchmarkResult BenchmarkRunner::benchmarkOpenCV(const cv::Mat& img, const string& img_name)
{
    BenchmarkResult result;
    result.method_name = "OpenCV";
    result.image_name = img_name;
    result.image_width = img.cols;
    result.image_height = img.rows;
    result.thread_count = -1;

    result.radius = config_.radius;
    result.sigma = config_.sigma;
    result.low_ratio = config_.low_ratio;
    result.high_ratio = config_.high_ratio;


    auto pipeline = [&](){
        cv::Mat blurred, edges;

        // OpenCV blurred
        int ksize = 2 * config_.radius + 1;
        cv::GaussianBlur(img, blurred, cv::Size(ksize, ksize), config_.sigma);
        double maxVal = 255.0;
        double lowThresh = maxVal * config_.low_ratio;
        double highThresh = maxVal * config_.high_ratio;
        cv::Canny(blurred, edges, lowThresh, highThresh);

        return edges;
    };

    result.execution_time_ms = measureTime(pipeline, config_.num_runs);

    return result;
}

template<typename Func>
double BenchmarkRunner::measureTime(Func func, int num_runs)
{
    vector<double> times;

    for(int i = 0; i < num_runs; i++)
    {
        auto start = chrono::high_resolution_clock::now();
        func();
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> duration = end - start;

        times.push_back(duration.count());
    }

    // choose the median time instead of average value to avoid the outlier infection
    sort(times.begin(), times.end());
    return times[num_runs/2];
}

vector<BenchmarkResult> BenchmarkRunner::runAll()
{
    vector<BenchmarkResult> results;

    cout << "Start to benchmark......" << endl;

    // check the OPENMP Status
    #ifdef _OPENMP
    cout << "OpenMP ENABLE, version: " << _OPENMP << endl;
    cout << "Max threads: " << omp_get_num_threads() << endl;
    #else
    cout << "OpenMP DISABLE, version: " << endl;
    cout << "The benchmarking will run in serial mode!" << endl;
    #endif

    cout << "Images: " << test_images_.size() << endl;
    cout << "Scale factor: ";
    for(auto s : config_.scale_factors) cout << s << "x ";
    cout << endl;
    cout << "Threads counts:";
    for(auto t : config_.thread_counts) cout << t << " ";
    cout << endl;
    cout << "Runs per test: " << config_.num_runs << endl;
    cout << endl;

    // for each test image
    for(size_t img_index = 0; img_index < test_images_.size(); img_index++)
    {
        const cv::Mat& original_img = test_images_[img_index];
        const string& img_name = image_names_[img_index];

        // for each scaler
        for(int scale : config_.scale_factors)
        {
            cv::Mat test_img;
            if(scale == 1)
            {
                test_img = original_img;
            }
            else
            {
                cv::resize(original_img, test_img, cv::Size(), scale, scale, cv::INTER_CUBIC);
            }
            cout << "Testing: " << img_name << " @ " << scale << "x (" 
                << test_img.cols << "x" << test_img.rows << ")" << endl;
            
            // Benchmark Serial
            if(config_.benchmark_serial)
            {
                cout << " - Serial" << flush;
                auto result = benchmarkSerial(test_img, img_name + "_" + to_string(scale) + "x");
                results.push_back(result);
                cout << result.execution_time_ms << " ms" << endl;
            }

            // Benchmark Parallel
            if(config_.benchmark_parallel)
            {
                for(int num_threads : config_.thread_counts)
                {
                    if(num_threads == 1) continue; // skip, it is tested in serial

                    cout << "- Parallel, " << num_threads << "threads, " << flush;
                    auto result = benchmarkParallel(test_img, img_name + "_" + to_string(scale) + "x", num_threads);
                    results.push_back(result);
                    cout << result.execution_time_ms << " ms" << endl;

                }
            }

            // Benchmark OpenCV
            if(config_.benchmark_opencv)
            {
                cout << "- OpenCV, " << flush;
                auto result = benchmarkOpenCV(test_img, img_name + "_" + to_string(scale) + "x");
                results.push_back(result);
                cout << result.execution_time_ms << " ms" << endl;
            }
            cout << endl;
        }
    }
    return results;
}

void BenchmarkRunner::saveResultsToCSV(const vector<BenchmarkResult>& results, const string& filename)
{
    ofstream file(filename);

    file << "Method,image,width,height,pixels,threads,time_ms,radius,sigma,low_ratio,high_ratio\n";

    for(const auto& r : results) {
        file << r.method_name << ","
             << r.image_name << ","
             << r.image_width << ","
             << r.image_height << ","
             << (r.image_width * r.image_height) << ","
             << r.thread_count << ","
             << std::fixed << std::setprecision(3) << r.execution_time_ms << ","
             << r.radius << ","
             << r.sigma << ","
             << r.low_ratio << ","
             << r.high_ratio << "\n";
    }

    file.close();
    cout << "Results saved to: " << filename << endl;
}