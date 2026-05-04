cmake_minimum_required(VERSION 3.16)
project(canny_project CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(canny_server 
        src/utils.cpp
        src/main.cpp
        src/gaussian.cpp
        src/gradient.cpp
        src/nms.cpp
        src/threshold.cpp
        src/hysteresis.cpp
        src/server.cpp
        src/threadpool.cpp
        )

add_executable(canny_benchmark 
        src/utils.cpp
        src/Benchmark/benchmark_main.cpp
        src/gaussian.cpp
        src/gradient.cpp
        src/nms.cpp
        src/threshold.cpp
        src/hysteresis.cpp
        src/threadpool.cpp
        src/Benchmark/benchmark.cpp
        )

add_executable(canny_evaluator 
        src/utils.cpp
        src/Evaluation/evaluate_main.cpp
        src/gaussian.cpp
        src/gradient.cpp
        src/nms.cpp
        src/threshold.cpp
        src/hysteresis.cpp
        src/Evaluation/evaluate.cpp
        )

target_link_libraries(canny_server
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

target_link_libraries(canny_benchmark
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

target_link_libraries(canny_evaluator
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

target_include_directories(canny_benchmark PRIVATE src)
target_include_directories(canny_evaluator PRIVATE src)