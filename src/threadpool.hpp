#pragma once
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include <functional>

using namespace std;

class ThreadPool
{
    public:
        ThreadPool(int numThreads);
        ~ThreadPool();
        void enqueue(function<void()> task);

    private:
        vector<thread> workers;
        queue<function<void()>> taskQueue;
        mutex queueMutex;
        condition_variable cv;
        bool stop;
};