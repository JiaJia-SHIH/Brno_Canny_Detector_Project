#include "threadpool.hpp"

using namespace std;

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

                    // 如果還有任務、或者確定要stop，就可以過
                    cv.wait(lock, [this](){
                        return !taskQueue.empty() || stop;
                    });

                    // 如果沒有stop的情況下，會出現沒有任務，但stop是true，要停了！但因為裡面沒有任務，所以卡在這裡-->永遠睡著


                    if(stop && taskQueue.empty()) break;

                    if(!taskQueue.empty())
                    {
                        task = move(taskQueue.front());
                        taskQueue.pop();
                    }
                }

                if(task) task();

            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        unique_lock<mutex> lock(queueMutex);
        stop = true;
    }

    cv.notify_all();

    for(auto& worker : workers)
    {
        worker.join();
    }
}

void ThreadPool::enqueue(function<void()> task)
{
    {
        unique_lock<mutex> lock(queueMutex);

        taskQueue.push(move(task));
    }

    cv.notify_one();
}
