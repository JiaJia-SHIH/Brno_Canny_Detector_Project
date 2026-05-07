#include "Evaluation/evaluate.hpp"
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    std::string bsds_root;
    
    if(argc > 1)
    {
        bsds_root = argv[1];
    }
    else
    {
        bsds_root = "../dataset/BSR/BSDS500";
    }
    
    cout << "------------------------------------------" << endl;
    cout << "  BSDS500 Canny Edge Detector Evaluation" << endl;
    cout << "  Manual vs Auto Threshold Comparison" << endl;
    cout << "------------------------------------------" << endl;
    cout << endl;
    cout << "Dataset: " << bsds_root << endl;
    cout << endl;
    
    // Create evaluator with standard tolerance
    BSDS500Evaluator evaluator(bsds_root, 0.0075f);
    
    // Parameters
    int radius = 2;
    float sigma = 1.4f;
    float low_ratio = 0.05f;
    float high_ratio = 0.15f;
    
    // Run comparison
    auto comparison = evaluator.compareThresholdMethods(radius, sigma, low_ratio, high_ratio);
    
    // Print summary
    cout << endl;
    cout << "-----------------------" << endl;
    cout << "  COMPARISON SUMMARY" << endl;
    cout << "-----------------------" << endl;
    cout << endl;
    cout << "Total images: " << comparison.manual_result.total_images << endl;
    cout << endl;
    cout << "MANUAL THRESHOLD (low=" << low_ratio << ", high=" << high_ratio << "):" << endl;
    cout << "  Precision: " << comparison.manual_result.mean_precision << endl;
    cout << "  Recall:    " << comparison.manual_result.mean_recall << endl;
    cout << "  F1-score:  " << comparison.manual_result.mean_f1 << endl;
    cout << endl;
    cout << "AUTO THRESHOLD:" << endl;
    cout << "  Precision: " << comparison.auto_result.mean_precision << endl;
    cout << "  Recall:    " << comparison.auto_result.mean_recall << endl;
    cout << "  F1-score:  " << comparison.auto_result.mean_f1 << endl;
    cout << endl;
    cout << "IMPROVEMENT:" << endl;
    cout << "  Precision: " << (comparison.auto_result.mean_precision - comparison.manual_result.mean_precision) << endl;
    cout << "  Recall:    " << (comparison.auto_result.mean_recall - comparison.manual_result.mean_recall) << endl;
    cout << "  F1-score:  " << (comparison.auto_result.mean_f1 - comparison.manual_result.mean_f1) << endl;
    cout << endl;
    
    // Save results
    string output_file = "../eval/threshold_comparison.csv";
    evaluator.saveComparisonResults(comparison, output_file);
    
    cout << endl;
    cout << "Evaluation complete!" << endl;
    
    return 0;
}