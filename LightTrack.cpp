//
// Created by xiongzhuang on 2021/10/8.
//
#include "LightTrack.h"
#include "timer.h"

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;

            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }


    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2;
}


LightTrack::LightTrack(std::string model_init, std::string model_backbone, std::string model_neck_head)
{
    this->load_model(model_init, model_backbone, model_neck_head);

}

LightTrack::~LightTrack()
{

}

void LightTrack::init(cv::Mat img, cv::Point target_pos, cv::Point2f target_sz, State &state)
{
    state.p = new Config(this->stride, this->even);

    state.im_h = img.rows;
    state.im_w = img.cols;

    state.p->update();
    state.p->renew();

    if (((target_pos.x * target_pos.y) / float(state.im_w * state.im_h)) < 0.004)
    {
        state.p->instance_size = SMALL_SZ;  // 256
        state.p->renew();
    }
    else
    {
        state.p->instance_size = BIG_SZ;  // 288
        state.p->renew();
    }

    std::cout << "init target pos: " << target_pos << std::endl;
    std::cout << "init target_sz: " << target_sz << std::endl;

    this->grids(state.p);

    // 对模板图像而言：在第一帧以s_z为边长，以目标中心为中心点，截取图像补丁（如果超出第一帧的尺寸，用均值填充）。之后将其resize为127x127x3.成为模板图像
    // context = 1/2 * (w+h) = 2*pad
    float wc_z = target_sz.x + state.p->context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + state.p->context_amount * (target_sz.x + target_sz.y);
    // z_crop size = sqrt((w+2p)*(h+2p))
    float s_z = round(sqrt(wc_z * hc_z));   // orignal size

    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;
    CropInfo crop_info;

    z_crop  = get_subwindow_tracking(img, target_pos, state.p->exemplar_size, int(s_z));

    // net init
    ncnn::Extractor ex_init = net_init.create_extractor();
    ex_init.set_light_mode(true);
    ex_init.set_num_threads(16);
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(z_crop.data, ncnn::Mat::PIXEL_BGR2RGB, z_crop.cols, z_crop.rows);
    ncnn_img.substract_mean_normalize(this->mean_vals, this->norm_vals);
    ex_init.input("input1", ncnn_img);
    ex_init.extract("output.1", zf);

    std::vector<float> hanning(state.p->score_size,0);  // 18
    std::vector<float> window(state.p->score_size*state.p->score_size, 0);
    for (int i = 0; i < state.p->score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (state.p->score_size - 1));
        hanning[i] = w;
    }
    for (int i = 0; i < state.p->score_size; i++)
    {

        for (int j = 0; j < state.p->score_size; j++)
        {
            window[i*state.p->score_size+j] = hanning[i] * hanning[j];
        }
    }

    state.avg_chans = avg_chans;
    state.window = window;
    state.target_pos = target_pos;
    state.target_sz = target_sz;


}

void LightTrack::update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, std::vector<float> &window, float scale_z, Config *p, float &cls_score_max)
{
    time_checker time2, time3, time4, time5;

    time2.start();
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(x_crops.data, ncnn::Mat::PIXEL_BGR2RGB, x_crops.cols, x_crops.rows);
    ncnn_img.substract_mean_normalize(this->mean_vals, this->norm_vals);
    time2.stop();
    time2.show_distance("Update stage ---- input seting cost time");

    time3.start();
    // net backbone
    ncnn::Extractor ex_backbone = net_backbone.create_extractor();
    ex_backbone.set_light_mode(true);
    ex_backbone.set_num_threads(16);

#if NCNN_VULKAN
    sd::cout << NCNN_VULKAN << std::endl;
    ex_backbone.opt.use_vulkan_compute = True;
#endif

    
    ex_backbone.input("input1", ncnn_img);

    ex_backbone.extract("output.1", xf);
    time3.stop();
    time3.show_distance("Update stage ---- output xf extracting cost time");

    time4.start();
    // net neck head
    ncnn::Extractor ex_neck_head = net_neck_head.create_extractor();
    ex_neck_head.set_light_mode(true);
    ex_neck_head.set_num_threads(16);
    ex_neck_head.input("input1", zf);
    ex_neck_head.input("input2", xf);
    ncnn::Mat cls_score, bbox_pred;
    ex_neck_head.extract("output.1", cls_score);  // [c, w, h] = [1, 18, 18]
    ex_neck_head.extract("output.2", bbox_pred); // [c, w, h] = [4, 18, 18]
    time4.stop();
    time4.show_distance("Update stage ---- output cls_score and bbox_pred extracting cost time");

    time5.start();
    // manually call sigmoid on the output
    std::vector<float> cls_score_sigmoid;

    float* cls_score_data = (float*)cls_score.data;
    cls_score_sigmoid.clear();

    int cols = cls_score.w;
    int rows = cls_score.h;

    for (int i = 0; i < cols*rows; i++)   // 18 * 18
    {
        cls_score_sigmoid.push_back(sigmoid(cls_score_data[i]));
    }

    std::vector<float> pred_x1(cols*rows, 0), pred_y1(cols*rows, 0), pred_x2(cols*rows, 0), pred_y2(cols*rows, 0);

    float* bbox_pred_data1 = bbox_pred.channel(0);
    float* bbox_pred_data2 = bbox_pred.channel(1);
    float* bbox_pred_data3 = bbox_pred.channel(2);
    float* bbox_pred_data4 = bbox_pred.channel(3);
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            pred_x1[i*cols + j] = this->grid_to_search_x[i*cols + j] - bbox_pred_data1[i*cols + j];
            pred_y1[i*cols + j] = this->grid_to_search_y[i*cols + j] - bbox_pred_data2[i*cols + j];
            pred_x2[i*cols + j] = this->grid_to_search_x[i*cols + j] + bbox_pred_data3[i*cols + j];
            pred_y2[i*cols + j] = this->grid_to_search_y[i*cols + j] + bbox_pred_data4[i*cols + j];
        }
    }

    // size penalty (1)
    std::vector<float> w(cols*rows, 0), h(cols*rows, 0);
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            w[i*cols + j] = pred_x2[i*cols + j] - pred_x1[i*cols + j];
            h[i*rows + j] = pred_y2[i*rows + j] - pred_y1[i*cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * p->penalty_tk);
    }

    // window penalty
    std::vector<float> pscore(rows*cols,0);
    int r_max = 0, c_max = 0;
    float maxScore = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - p->window_influence) + window[i] * p->window_influence;
        if (pscore[i] > maxScore)
        {
            // get max
            maxScore = pscore[i];
            r_max = std::floor(i / rows);
            c_max = ((float)i / rows - r_max) * rows;
        }
    }

    time5.stop();
    time5.show_distance("Update stage ---- postprocess cost time");
    std::cout << "pscore_window max score is: " << pscore[r_max * cols + c_max] << std::endl;

    // to real size
    float pred_x1_real = pred_x1[r_max * cols + c_max]; // pred_x1[r_max, c_max]
    float pred_y1_real = pred_y1[r_max * cols + c_max];
    float pred_x2_real = pred_x2[r_max * cols + c_max];
    float pred_y2_real = pred_y2[r_max * cols + c_max];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;

    float diff_xs = pred_xs - p->instance_size / 2;
    float diff_ys = pred_ys - p->instance_size / 2;

    diff_xs /= scale_z;
    diff_ys /= scale_z;
    pred_w /=scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr = penalty[r_max * cols + c_max] * cls_score_sigmoid[r_max * cols + c_max] * p->lr;

    // size rate
    auto res_xs = float (target_pos.x + diff_xs);
    auto res_ys = float (target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz.y;

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz.x = target_sz.x * (1 - lr) + lr * res_w;
    target_sz.y = target_sz.y * (1 - lr) + lr * res_h;

    cls_score_max = cls_score_sigmoid[r_max * cols + c_max];
}

void LightTrack::track(State &state, cv::Mat im)
{
    time_checker time1;

    Config *p = state.p;
    cv::Scalar avg_chans = state.avg_chans;
    std::vector<float> window = state.window;
    cv::Point target_pos = state.target_pos;
    cv::Point2f target_sz = state.target_sz;

    float hc_z = target_sz.y + p->context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + p->context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  // roi size
    float scale_z = p->exemplar_size / s_z;  // 127/

    float d_search = (p->instance_size - p->exemplar_size) / 2;  // backbone_model_size - init_model_size = 288-127
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;


    time1.start();
    cv::Mat x_crop;
    CropInfo crop_info;
    x_crop  = get_subwindow_tracking(im, target_pos, p->instance_size, int(s_x));
    time1.stop();
    time1.show_distance("Update stage ---- get subwindow cost time");

    state.x_crop = x_crop;

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    float cls_score_max;
    this->update(x_crop, target_pos, target_sz, window, scale_z, p, cls_score_max);
    target_pos.x = std::max(0, min(state.im_w, target_pos.x));
    target_pos.y = std::max(0, min(state.im_h, target_pos.y));
    target_sz.x = float(std::max(10, min(state.im_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, min(state.im_h, int(target_sz.y))));

    std::cout << "track target pos: " << target_pos << std::endl;
    std::cout << "track target_sz: " << target_sz << std::endl;
    std::cout << "cls_score_max: " << cls_score_max << std::endl;

    state.target_pos = target_pos;
    state.target_sz = target_sz;
}

void LightTrack::load_model(std::string model_init, std::string model_backbone, std::string model_neck_head)
{
    this->net_init.load_param((model_init+".param").c_str());
    this->net_init.load_model((model_init+".bin").c_str());
    this->net_backbone.load_param((model_backbone+".param").c_str());
    this->net_backbone.load_model((model_backbone+".bin").c_str());
    this->net_neck_head.load_param((model_neck_head+".param").c_str());
    this->net_neck_head.load_model((model_neck_head+".bin").c_str());
}

void LightTrack::grids(Config *p)
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = p->score_size;   // 18

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            this->grid_to_search_x[i*sz+j] = j*p->total_stride;   // 0~18*16 = 0~288
            this->grid_to_search_y[i*sz+j] = i*p->total_stride;
        }
    }
}

cv::Mat LightTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
        //te_im(cv::Rect(left_pad, top_pad, im.cols, im.rows)) = im;
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, 0.f);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path;
}