/**
 * @file threadpool.cpp
 * @author SHIH YUE JIA (xshihyu00)
 * @brief Thread pool implementation for concurrent request handling
 **/

#include "threadpool.hpp"

using namespace std;

// to handle multi-request on server
ThreadPool::ThreadPool(int numThreads) : stop(false)
{
    for(int i = 0; i < numThreads; i++)
    {
        workers.emplace_back([this](){    
            while(true)
            {
                function<void()> task;

                {
                    unique_lock<mutex> lock(queueMutex);

                    // continue if stop or some tasks in queue
                    cv.wait(lock, [this](){
                        return !taskQueue.empty() || stop;
                    });

                    // if the whole working flow is to be stopped , also there is no task in queue, finished safely
                    if(stop && taskQueue.empty()) break;

                    // if stop but there still have some task need to be done, do it first
                    if(!taskQueue.empty())
                    {
                        task = move(taskQueue.front());
                        taskQueue.pop();
                    }
                }

                // do task
                if(task) task();

            }
        });
    }
}

// deconstructor
ThreadPool::~ThreadPool()
{
    // avoid race condition
    {
        unique_lock<mutex> lock(queueMutex);
        stop = true;
    }

    // call workers to check the tasks are done or not
    cv.notify_all();

    // wait all the workers finishing there job
    for(auto& worker : workers)
    {
        worker.join();
    }
}

// add tasks to queue
void ThreadPool::enqueue(function<void()> task)
{
    {
        unique_lock<mutex> lock(queueMutex);

        taskQueue.push(move(task));
    }

    // call one worker to do
    cv.notify_one();
}
