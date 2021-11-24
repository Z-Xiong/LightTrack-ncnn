//
// Created by xiongzhuang on 2021/1/11.
//

#ifndef BABY_FACE_NCNN_DEMO_TIMER_H
#define BABY_FACE_NCNN_DEMO_TIMER_H
#include <sys/time.h>

struct time_checker {
    timeval start_time;
    timeval stop_time;

    void start()
    {
        gettimeofday(&start_time, nullptr);
    }
    void stop()
    {
        gettimeofday(&stop_time, nullptr);
    }
    int distance()
    {
        long time_1_token = start_time.tv_sec * 1000 + start_time.tv_usec / 1000;
        long time_2_token = stop_time.tv_sec * 1000 + stop_time.tv_usec / 1000;
        return time_2_token - time_1_token;
    }
    void show_distance(const char*title = "current time")
    {
        long time_1_token = start_time.tv_sec * 1000 + start_time.tv_usec / 1000;
        long time_2_token = stop_time.tv_sec * 1000 + stop_time.tv_usec / 1000;
        printf("%s : %ld ms\n", title, time_2_token - time_1_token);
    }
    void show_ns_distance(const char*title = "current time")
    {
        long time_1_token = start_time.tv_sec * 1000000 + start_time.tv_usec;
        long time_2_token = stop_time.tv_sec * 1000000 + stop_time.tv_usec;
        printf("%s : %ld ns\n", title, time_2_token - time_1_token);
    }
    bool timeout(int second)
    {
        timeval current;
        gettimeofday(&current, nullptr);
        long time_1_token = start_time.tv_sec * 1000 + start_time.tv_usec / 1000;
        long time_2_token = current.tv_sec * 1000 + current.tv_usec / 1000;
        int value = time_2_token - time_1_token;
        return value > second;
    }
    static long get_current_timetoken()
    {
        timeval current;
        gettimeofday(&current, nullptr);
        return (current.tv_sec * 1000 + current.tv_usec / 1000);
    }
};

#endif //BABY_FACE_NCNN_DEMO_TIMER_H
