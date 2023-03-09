//
// Created by xiongzhuang on 2021/10/8.
//

#ifndef LIGHTTRACK_LIGHTTRACK_H
#define LIGHTTRACK_LIGHTTRACK_H

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "ncnn/net.h"
#include "ncnn/layer_type.h"

#define SMALL_SZ 256
#define BIG_SZ 288
#define LR 0.616
#define PENALTY_K 0.007
#define RATIO 1
#define WINDOW_INCLUENCE 0.225
#define PI 3.1415926

#define USE_GPU false

using namespace cv;

class Config
{
public:
    Config() {};
    Config(int stride, int even) : stride(stride), even(even)
    {
        this->total_stride = stride;
        if (even)
        {
            this->exemplar_size = 128;
            this->instance_size = 256;
        }
        else
        {
            this->exemplar_size = 127;
            this->instance_size = 255;
        }
        this->score_size = int(round(this->instance_size / this->total_stride));
    };

    ~Config() {};

    int stride = 16;
    int even = 0;
    float penalty_tk = 0.062;
    float window_influence = 0.38;
    float lr = 0.765;
    std::string windowing = "cosine";
    int exemplar_size;
    int instance_size;
    int total_stride;
    int score_size;
    float context_amount = 0.5;
    float ratio = 0.94;
    int small_sz = 256;
    int big_sz = 288;

    void update()
    {
        this->penalty_tk = 0.007;
        this->lr = 0.616;
        this->window_influence = 0.225;
        this->ratio = 1;
        this->renew();
    };

    void renew()
    {
        this->score_size = int(round(this->instance_size / this->total_stride));
    };
};

struct State {
    Config *p;
    int im_h;
    int im_w;
    cv::Scalar avg_chans;
    std::vector<float> window;
    cv::Point target_pos;
    cv::Point2f target_sz = {0.f, 0.f};
    cv::Mat x_crop;
    float cls_sccre_max;

};

struct CropInfo {
    std::vector<int> crop_cords;
    cv::Mat empty_mask;
    std::vector<int> pad_info;
};

class LightTrack {
public:
    LightTrack(std::string model_init, std::string model_update);
    ~LightTrack();
    void init(cv::Mat img, cv::Point target_pos, cv::Point2f target_sz, State &state);
    void update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, std::vector<float> &window, float scale_z, Config *p, float &cls_score_max);
    void track(State &state, cv::Mat im);
    void load_model(std::string model_init, std::string model_update);


    ncnn::Net net_init, net_update;
    ncnn::Mat zf, xf;

    int stride=16;
    int even=0;
    const float mean_vals[3] = { 0.485f*255.f, 0.456f*255.f, 0.406f*255.f };  // RGB
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

private:
    void grids(Config *p);
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
};




#endif //LIGHTTRACK_LIGHTTRACK_H
