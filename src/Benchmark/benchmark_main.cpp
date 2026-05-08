/**
 * @file benchmark_main.cpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief Entry point for benchmarking
 **/

#include "Benchmark/benchmark.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    BenchmarkConfig config;

    if(argc > 1)
    {
        config.bsds500_path = argv[1];
    }
    else
    {
        config.bsds500_path = "../dataset/BSR";
    }

    // test image
    config.test_images = {};

    // scale
    config.scale_factors = {1, 2};

    // threads number
    config.thread_counts = {1, 2, 4, 8};

    // runs
    config.num_runs = 5;

    // benchmark mode
    config.benchmark_serial = true;
    config.benchmark_parallel = true;
    config.benchmark_opencv = true;

    // parameter
    config.radius = 2;
    config.sigma = 1.4f;
    config.low_ratio = 0.05f;
    config.high_ratio = 0.15f;

    cout << "---------------------------------" << endl;
    cout << "Canny Edge Detector Benchmark" << endl;
    cout << " Using BSDS500 Real Images" << endl;
    cout << "---------------------------------" << endl;

    cout << "Dataset path: " << config.bsds500_path << endl;
    cout << endl;

    // run benchmark
    BenchmarkRunner runner(config);
    auto results = runner.runAll();

    // save Results
    string output_file = "../benchmark_results/benchmark_results.csv";
    runner.saveResultsToCSV(results, output_file);

    cout << std::endl;
    cout << "  Benchmark Complete!" << endl;
    cout << "---------------------------------" << endl;
    cout << "Results saved to: " << output_file << endl;
    cout << "Run plot_benchmark.py to visualize results" << endl;

    return 0;
}