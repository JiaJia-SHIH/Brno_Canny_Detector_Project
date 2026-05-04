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
    cout << "------------------------------------------" << endl;
    cout << endl;
    cout << "Dataset: " << bsds_root << endl;
    cout << endl;
    
    // Create evaluator with standard tolerance
    BSDS500Evaluator evaluator(bsds_root, 0.0075f);
    
    // Evaluate with default Canny parameters
    int radius = 2;
    float sigma = 1.4f;
    float low_ratio = 0.05f;
    float high_ratio = 0.15f;
    
    auto result = evaluator.evaluateTestSet(radius, sigma, low_ratio, high_ratio);
    
    // Print summary
    cout << endl;
    cout << "--------------------" << endl;
    cout << "  Summary" << endl;
    cout << "--------------------" << endl;
    cout << "Total images:     " << result.total_images << endl;
    cout << "Mean Precision:   " << result.mean_precision << endl;
    cout << "Mean Recall:      " << result.mean_recall << endl;
    cout << "Mean F1-score:    " << result.mean_f1 << endl;
    cout << endl;
    
    // Save results
    string output_file = "../eval/bsds500_results.csv";
    evaluator.saveResults(result, output_file);
    
    cout << endl;
    cout << "Evaluation complete!" << endl;
    
    return 0;
}