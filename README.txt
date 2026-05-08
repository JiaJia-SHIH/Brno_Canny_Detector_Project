Project: Parallel Canny Edge Detection: OpenMP Optimization and Quantitative Evaluation on BSDS500
Author:  SHIH YUE JIA (xshihyu00)
Course:  Image Processing
Institution: Brno University of Technology

------------------------------------------
Dependencies
------------------------------------------
- CMake  >= 3.16.3
- GCC    >= 9.4.0  (with OpenMP support)
- OpenCV >= 4.2.0

------------------------------------------
Directory Structure
------------------------------------------
canny_project/
├── build/                  
├── dataset/                (place BSDS500 dataset here)
│   └── BSR/
│       └── BSDS500/
│           └── data/
│               ├── images/test/
│               └── groundTruth/test_png/
├── eval/                   (output images and CSVs saved here)
├── src/
│   ├── Benchmark/
│   │   ├── benchmark.hpp
│   │   ├── benchmark.cpp
│   │   └── benchmark_main.cpp
│   ├── Evaluation/
│   │   ├── evaluate.hpp
│   │   ├── evaluate.cpp
│   │   └── evaluate_main.cpp
│   ├── img/                (place input images here)
│   ├── gaussian.hpp / gaussian.cpp
│   ├── gradient.hpp / gradient.cpp
│   ├── nms.hpp      / nms.cpp
│   ├── threshold.hpp/ threshold.cpp
│   ├── hysteresis.hpp/hysteresis.cpp
│   ├── utils.hpp    / utils.cpp
│   ├── threadpool.hpp/threadpool.cpp
│   ├── server.hpp   / server.cpp
│   └── main.cpp
└── CMakeLists.txt

------------------------------------------
Build
------------------------------------------
mkdir build
cd build
cmake ..
make -j$(nproc)

This produces three executables inside build/:
  - canny_server
  - canny_benchmark
  - canny_evaluator

------------------------------------------
Usage
------------------------------------------

1. canny_server
   Runs the full Canny pipeline on a single image and starts
   an interactive HTTP server for parameter tuning.

   ./canny_server <image_path>
   Example:
   ./canny_server ../src/img/lenna.jpg

   Then open a browser at http://localhost:9090
   (or use SSH port forwarding: ssh -L 9090:localhost:9090 user@host)

   Pipeline output images are saved to ../eval/:
     01_gaussian_result.jpg
     02_magnitude_result.jpg
     02_orientation_result.jpg
     03_nms_result.jpg
     04_threshold_result.jpg
     04_threshold_auto.jpg
     04_threshold_comparison.jpg
     04_threshold_histogram.jpg
     05_hysteresis_result.jpg

2. canny_evaluator
   Evaluates the detector against the BSDS500 test set,
   comparing manual vs. auto thresholding.

   ./canny_evaluator
   Example:
   ./canny_evaluator

   Output saved to ../eval/threshold_comparison.csv

3. canny_benchmark
   Benchmarks serial vs. parallel (OpenMP) vs. OpenCV Canny
   across different thread counts and image scales.

   ./canny_benchmark
   Example:
   ./canny_benchmark

   Output saved to ../eval/benchmark_results.csv

------------------------------------------
Notes
------------------------------------------
- All visual outputs are saved as .jpg files to the eval/ directory.
- The interactive server requires port forwarding when accessed
  over SSH (see canny_server usage above).
- The eval/ directory must exist before running:
  mkdir -p eval