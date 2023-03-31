//
// Created by xiongzhuang on 2021/10/8.
//
#include "LightTrack.h"
#include "timer.h"

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh) {
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h, float sz) {
    int rows = int(std::sqrt(h.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pad[i * cols + j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float t = std::sqrt((w[i * cols + j] + pad[i * cols + j]) * (h[i * cols + j] + pad[i * cols + j])) / sz;

            sz2.push_back(std::max(t, (float) 1.0 / t));
        }
    }


    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz) {
    int rows = int(std::sqrt(h.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float) 1.0 / t));
        }
    }

    return sz2;
}


LightTrack::LightTrack(const char *model_init, const char *model_update) {
    score_size = int(round(this->instance_size / this->total_stride));

    std::string model_init_str = model_init;
    std::string model_update_str = model_update;
    this->load_model(model_init, model_update);

}

LightTrack::~LightTrack() {

}

void LightTrack::init(const uint8_t *img, Bbox &box, int im_h , int im_w) {
    ori_img_h = im_h;
    ori_img_w = im_w;

    this->target_sz.x = box.x1-box.x0;
    this->target_sz.y = box.y1-box.y0;
    this->target_pos.x = box.x0 + (box.x1-box.x0)/2;
    this->target_pos.y = box.y0 + (box.y1-box.y0)/2;

    std::cout << "init target pos: " << target_pos << std::endl;
    std::cout << "init target_sz: " << target_sz << std::endl;

    this->grids();

    // 对模板图像而言：在第一帧以s_z为边长，以目标中心为中心点，截取图像补丁（如果超出第一帧的尺寸，用均值填充）。之后将其resize为127x127x3.成为模板图像
    // context = 1/2 * (w+h) = 2*pad
    float wc_z = target_sz.x + context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + context_amount * (target_sz.x + target_sz.y);
    // z_crop size = sqrt((w+2p)*(h+2p))
    float s_z = round(sqrt(wc_z * hc_z));   // orignal size

    cv::Mat z_crop;
    cv::Mat img_(im_h, im_w, CV_8UC3, (void*)img, im_w*3);

    z_crop = get_subwindow_tracking(img_, target_pos, exemplar_size, int(s_z));

    // net init
    ncnn::Extractor ex_init = net_init.create_extractor();
    ex_init.set_light_mode(true);
    ex_init.set_num_threads(6);
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(z_crop.data, ncnn::Mat::PIXEL_BGR2RGB, z_crop.cols, z_crop.rows);
    ncnn_img.substract_mean_normalize(this->mean_vals, this->norm_vals);
    ex_init.input("input1", ncnn_img);
    ex_init.extract("output.1", zf);

    std::vector<float> hanning(score_size, 0);  // 18
    window.resize(score_size*score_size);
    for (int i = 0; i < score_size; i++) {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
        hanning[i] = w;
    }
    for (int i = 0; i < score_size; i++) {

        for (int j = 0; j < score_size; j++) {
            window[i * score_size + j] = hanning[i] * hanning[j];
        }
    }
}

void LightTrack::update(const cv::Mat &x_crops, float scale_z) {
    time_checker time2, time3, time4, time5;

    time2.start();
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(x_crops.data, ncnn::Mat::PIXEL_BGR2RGB, x_crops.cols, x_crops.rows);
    ncnn_img.substract_mean_normalize(this->mean_vals, this->norm_vals);
    time2.stop();
    time2.show_distance("Update stage ---- input seting cost time");

    time3.start();
    // net backbone
    ncnn::Extractor ex_update = net_update.create_extractor();
    ex_update.set_light_mode(true);
    ex_update.set_num_threads(6);

#if NCNN_VULKAN and USE_GPU
    std::cout << NCNN_VULKAN << std::endl;
    ex_update.opt.use_vulkan_compute = true;
#endif


    ex_update.input("input1", zf);
    ex_update.input("input2", ncnn_img);
    ncnn::Mat cls_score, bbox_pred;
    ex_update.extract("output.1", cls_score);  // [c, w, h] = [1, 18, 18]
    ex_update.extract("output.2", bbox_pred); // [c, w, h] = [4, 18, 18]
    time3.stop();
    time3.show_distance("Update stage ---- output cls_score and bbox_pred extracting cost time");

    time4.start();
    // manually call sigmoid on the output
    std::vector<float> cls_score_sigmoid;

    float *cls_score_data = (float *) cls_score.data;
    cls_score_sigmoid.clear();

    int cols = cls_score.w;
    int rows = cls_score.h;

    for (int i = 0; i < cols * rows; i++)   // 18 * 18
    {
        cls_score_sigmoid.push_back(sigmoid(cls_score_data[i]));
    }

    std::vector<float> pred_x1(cols * rows, 0), pred_y1(cols * rows, 0), pred_x2(cols * rows, 0), pred_y2(cols * rows,
                                                                                                          0);

    float *bbox_pred_data1 = bbox_pred.channel(0);
    float *bbox_pred_data2 = bbox_pred.channel(1);
    float *bbox_pred_data3 = bbox_pred.channel(2);
    float *bbox_pred_data4 = bbox_pred.channel(3);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            pred_x1[i * cols + j] = this->grid_to_search_x[i * cols + j] - bbox_pred_data1[i * cols + j];
            pred_y1[i * cols + j] = this->grid_to_search_y[i * cols + j] - bbox_pred_data2[i * cols + j];
            pred_x2[i * cols + j] = this->grid_to_search_x[i * cols + j] + bbox_pred_data3[i * cols + j];
            pred_y2[i * cols + j] = this->grid_to_search_y[i * cols + j] + bbox_pred_data4[i * cols + j];
        }
    }

    // size penalty (1)
    std::vector<float> w(cols * rows, 0), h(cols * rows, 0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            w[i * cols + j] = pred_x2[i * cols + j] - pred_x1[i * cols + j];
            h[i * cols + j] = pred_y2[i * cols + j] - pred_y1[i * cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows * cols, 0);
    for (int i = 0; i < rows * cols; i++) {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i] - 1) * penalty_tk);
    }

    // window penalty
    std::vector<float> pscore(rows * cols, 0);
    int r_max = 0, c_max = 0;
    float maxScore = 0;
    for (int i = 0; i < rows * cols; i++) {
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - window_influence) + window[i] * window_influence;
        if (pscore[i] > maxScore) {
            // get max
            maxScore = pscore[i];
            r_max = std::floor(i / rows);
            c_max = ((float) i / rows - r_max) * rows;
        }
    }

    time4.stop();
    time4.show_distance("Update stage ---- postprocess cost time");
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

    float diff_xs = pred_xs - instance_size / 2;
    float diff_ys = pred_ys - instance_size / 2;

    diff_xs /= scale_z;
    diff_ys /= scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr_ = penalty[r_max * cols + c_max] * cls_score_sigmoid[r_max * cols + c_max] * lr;

    // size rate
    auto res_xs = float(target_pos.x + diff_xs);
    auto res_ys = float(target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr_) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr_) * target_sz.y;

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz.x = target_sz.x * (1 - lr_) + lr_ * res_w;
    target_sz.y = target_sz.y * (1 - lr_) + lr_ * res_h;
}

void LightTrack::track(const uint8_t *img) {
    time_checker time1;

    float hc_z = target_sz.y + context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  // roi size
    float scale_z = exemplar_size / s_z;  // 127/

    float d_search = (instance_size - exemplar_size) / 2;  // backbone_model_size - init_model_size = 288-127
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;


    time1.start();
    cv::Mat x_crop;
    cv::Mat img_(ori_img_h, ori_img_w, CV_8UC3, (void*)img, ori_img_w*3);
    x_crop = get_subwindow_tracking(img_, target_pos, instance_size, int(s_x));
    time1.stop();
    time1.show_distance("Update stage ---- get subwindow cost time");

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    this->update(x_crop, scale_z);
    target_pos.x = std::max(0, min(ori_img_w, target_pos.x));
    target_pos.y = std::max(0, min(ori_img_h, target_pos.y));
    target_sz.x = float(std::max(10, min(ori_img_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, min(ori_img_h, int(target_sz.y))));

    std::cout << "track target pos: " << target_pos << std::endl;
    std::cout << "track target_sz: " << target_sz << std::endl;
}

void LightTrack::load_model(std::string model_init, std::string model_update) {
    this->net_init.load_param((model_init + ".param").c_str());
    this->net_init.load_model((model_init + ".bin").c_str());
    this->net_update.load_param((model_update + ".param").c_str());
    this->net_update.load_model((model_update + ".bin").c_str());
}

void LightTrack::grids() {
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = score_size;   // 18

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            this->grid_to_search_x[i * sz + j] = j * total_stride;   // 0~18*16 = 0~288
            this->grid_to_search_y[i * sz + j] = i * total_stride;
        }
    }
}

cv::Mat LightTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz) {
    float c = (float) (original_sz + 1) / 2;
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

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0) {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
        //te_im(cv::Rect(left_pad, top_pad, im.cols, im.rows)) = im;
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, 0.f);
        im_path_original = te_im(
                cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    } else
        im_path_original = im(
                cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path;
}