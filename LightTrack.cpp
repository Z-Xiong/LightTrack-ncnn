//
// Created by xiongzhuang on 2021/10/8.
//
#include "LightTrack.h"
#include "timer.h"

void visualize(const char* title, const ncnn::Mat& m)
{
    std::vector<cv::Mat> normed_feats(m.c);

    for (int i=0; i<m.c; i++)
    {
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y=0; y<m.h; y++)
        {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x=0; x<m.w; x++)
            {
                float v = tp[x];
                if (v != v)
                {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }

                sp += 3;
            }
        }
    }

    int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
    int th = (m.c - 1) / tw + 1;

    cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
    show_map = cv::Scalar(127);

    // tile
    for (int i=0; i<m.c; i++)
    {
        int ty = i / tw;
        int tx = i % tw;

        normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
    }

    cv::resize(show_map, show_map, cv::Size(0,0), 2, 2, cv::INTER_NEAREST);
    cv::imshow(title, show_map);
}

void pretty_print(const ncnn::Mat& m)
{
    std::cout << "M channel is " << m.c << std::endl;

    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

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

std::vector<float> hanning(int M)
{
    std::vector<float> window(M);
    for (int i=0; i<(M+1)/2; i++)
    {
        window[i] = 0.5f - 0.5f * cos(2*PI*i / (M-1));
        window[M-1-i] = window[i];
    }
    return window;
}

std::vector<std::vector<float>> ones(int M, int N)
{
    std::vector<std::vector<float> > out(M);
    for (int i=0; i<M; i++)
    {
        out[i].resize(N);
    }
    for (int i=0; i<M; i++)
        for (int j = 0; j < N; j++)
        {
            out[i][j] = 1.f;
        }
    return out;
}

std::vector<std::vector<float>> vector_outer(std::vector<float> a, std::vector<float> b)
{
    std::vector<std::vector<float> > out(a.size());
    for (int i=0; i<a.size(); i++)
    {
        out[i].resize(b.size());
    }
    for (int i=0; i<a.size(); i++)
    {
        for (int j = 0; j < b.size(); j++)
        {
            out[i][j] = a[i] * b[j];
        }
    }
    return out;
}


// https://github.com/Tencent/ncnn/wiki/low-level-operation-api
enum OperationType
{
    Operation_ADD = 0,
    Operation_SUB = 1,
    Operation_MUL = 2,
    Operation_DIV = 3,
    Operation_POW = 4,
    Operation_CHANGE = 5,
    Operation_EXP = 6,
    Operation_RSUB = 7,
    Operation_RDIV = 8,
    Operation_SIGMOID = 9,
};

// https://github.com/Tencent/ncnn/blob/master/src/layer/binaryop.cpp
void binary_op(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& out, OperationType op_type, float scalar = 1.f)
{
    out.create_like(a);

    for (int c=0; c < a.c; c++)
    {
        const float *ptr_a = a.channel(c);
        const float *ptr_b = b.channel(c);
        float *ptr_out = out.channel(c);
        for (int i=0; i < a.h; i++)
        {
            for (int j=0; j < a.w; j++)
            {
                switch (op_type) {
                    case Operation_ADD:
                        ptr_out[j] = ptr_a[j] + ptr_b[j]; break;
                    case Operation_SUB:
                        ptr_out[j] = ptr_a[j] - ptr_b[j]; break;
                    case Operation_MUL:
                        ptr_out[j] = ptr_a[j] * ptr_b[j]; break;
                    case Operation_DIV:
                        ptr_out[j] = ptr_a[j] / ptr_b[j]; break;
                    case Operation_POW:
                        ptr_out[j] = pow(ptr_a[j], ptr_b[j]); break;
                    case Operation_CHANGE:
                        ptr_out[j] = std::max(ptr_a[j], 1.f / ptr_b[j]); break;
                    case Operation_EXP:
                        ptr_out[j] = std::exp(-(ptr_a[j] * ptr_b[j] - 1) * scalar); break;
                    case Operation_RSUB:
                        ptr_out[j] = ptr_b[j] - ptr_a[j]; break;
                    case Operation_RDIV:
                        ptr_out[j] = ptr_b[j] / ptr_a[j]; break;
                    default:
                        break;
                }
            }
            ptr_a += a.w; ptr_b += b.w; ptr_out += out.w;
        }
    }
}

void binary_op_scalar(const ncnn::Mat& a, const float b, ncnn::Mat& out, OperationType op_type)
{
    out.create_like(a);

    for (int c=0; c < a.c; c++)
    {
        const float *ptr_a = a.channel(c);
        float *ptr_out = out.channel(c);
        for (int i=0; i < a.h; i++)
        {
            for (int j=0; j < a.w; j++)
            {
                switch (op_type) {
                    case Operation_ADD:
                        ptr_out[j] = ptr_a[j] + b; break;
                    case Operation_SUB:
                        ptr_out[j] = ptr_a[j] - b; break;
                    case Operation_MUL:
                        ptr_out[j] = ptr_a[j] * b; break;
                    case Operation_DIV:
                        ptr_out[j] = ptr_a[j] / b; break;
                    case Operation_POW:
                        ptr_out[j] = pow(ptr_a[j], b); break;
                    case Operation_CHANGE:
                        ptr_out[j] = std::max(ptr_a[j], b); break;
                    case Operation_EXP:
                        ptr_out[j] = std::exp(-(ptr_a[j] * b)); break;
                    case Operation_RSUB:
                        ptr_out[j] = b - ptr_a[j]; break;
                    case Operation_RDIV:
                        ptr_out[j] = b / ptr_a[j]; break;
                    case Operation_SIGMOID:
                        ptr_out[j] = sigmoid(ptr_a[j]); break;
                    default:
                        break;
                }
            }
            ptr_a += a.w; ptr_out += out.w;
        }
    }
}

void binary_op_nihui(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c, OperationType op_type)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_bf16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("BinaryOp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, op_type);

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;

}

void vector2ncnnmat(const std::vector<std::vector<float> > &b, ncnn::Mat &a)
{
    int h = b.size();
    int w = b[0].size();

    a = ncnn::Mat(w, h);

    for (int c=0; c < a.c; c++)
    {
        float *ptr = a.channel(c);
        for (int i=0; i<a.h; i++)
        {
            for (int j=0; j<a.w; j++)
            {
                ptr[j] = b[i][j];
            }
            ptr += a.w;
        }
    }
}

void ncnnmat2vector(const ncnn::Mat &a, std::vector<std::vector<float> > &out)
{
    out = std::vector<std::vector<float> > (a.h);
    for (int i=0; i<a.h; i++)
    {
        out[i].resize(a.w);
    }
    for (int i=0; i<a.h; i++)
    {
        const float *ptr = a.row(i);
        for (int j = 0; j < a.w; j++)
        {
            out[i][j] = ptr[j];
        }
    }
}

LightTrack::LightTrack(std::string model_init, std::string model_backbone, std::string model_neck_head)
{
    this->load_model(model_init, model_backbone, model_neck_head);

}

LightTrack::~LightTrack()
{

}

void LightTrack::init(cv::Mat img, cv::Point target_pos, cv::Scalar target_sz, State &state)
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
    float wc_z = target_sz[0] + state.p->context_amount * (target_sz[0] + target_sz[1]);
    float hc_z = target_sz[1] + state.p->context_amount * (target_sz[0] + target_sz[1]);
    // z_crop size = sqrt((w+2p)*(h+2p))
    float s_z = round(sqrt(wc_z * hc_z));   // orignal size

    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;
    CropInfo crop_info;

    get_subwindow_tracking(img, z_crop, crop_info, target_pos, state.p->exemplar_size, s_z, state.avg_chans);  //state.p->exemplar_size=127

    // net init
    ncnn::Extractor ex_init = net_init.create_extractor();
    ex_init.set_light_mode(true);
    ex_init.set_num_threads(16);
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(z_crop.data, ncnn::Mat::PIXEL_BGR2RGB, z_crop.cols, z_crop.rows);
    ncnn_img.substract_mean_normalize(this->mean_vals, this->norm_vals);
    ex_init.input("input1", ncnn_img);
    ex_init.extract("output.1", zf);

    std::vector<std::vector<float> > window;
    if (state.p->windowing == "cosine")
        window  =vector_outer(hanning(state.p->score_size), hanning(state.p->score_size));
    else if (state.p->windowing == "uniform")
        window  = ones(state.p->score_size, state.p->score_size);
    else
        throw "Unsupported window type";

    state.avg_chans = avg_chans;
    state.window = window;
    state.target_pos = target_pos;
    state.target_sz = target_sz;


}

void LightTrack::update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Scalar &target_sz, std::vector<std::vector<float> > &window, float scale_z, Config *p, float &cls_score_max)
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
    ex_neck_head.extract("output.1", cls_score);
    ex_neck_head.extract("output.2", bbox_pred);
    time4.stop();
    time4.show_distance("Update stage ---- output cls_score and bbox_pred extracting cost time");

    time5.start();
    ncnn::Mat cls_score_sigmoid;
    // manually call sigmoid on the output
    {
        /*
        ncnn::Layer* sigmoid = ncnn::create_layer("Sigmoid");

        ncnn::ParamDict pd;
        sigmoid->load_param(ptarget_posd);

        sigmoid->forward_inplace(cls_score, net_neck_head.opt);

        delete sigmoid;
        */

        // more faster
        binary_op_scalar(cls_score, 1, cls_score_sigmoid, Operation_SIGMOID);
    }

    // bbox to real predict
    ncnn::Mat pred_x1, pred_y1, pred_x2, pred_y2;
    binary_op(this->grid_to_search_x.channel(0), bbox_pred.channel(0), pred_x1, Operation_SUB);
    binary_op(this->grid_to_search_y.channel(0), bbox_pred.channel(1), pred_y1, Operation_SUB);
    binary_op(this->grid_to_search_x.channel(0), bbox_pred.channel(2), pred_x2, Operation_ADD);
    binary_op(this->grid_to_search_y.channel(0), bbox_pred.channel(3), pred_y2, Operation_ADD);

    // size penalty (1)
    float sz_wh = this->sz_wh(target_sz);
    ncnn::Mat w_mat, h_mat, wh_mat;
    binary_op(pred_x2, pred_x1, w_mat, Operation_SUB);
    binary_op(pred_y2, pred_y1, h_mat, Operation_SUB);
    this->sz(w_mat, h_mat, wh_mat);

    ncnn::Mat change_mat1;
    binary_op_scalar(wh_mat, sz_wh, change_mat1, Operation_DIV);

    ncnn::Mat s_c;
    binary_op(change_mat1, change_mat1, s_c, Operation_CHANGE);

    // size penalty (2)
    float w_h_ratio = target_sz[0] / target_sz[1];

    ncnn::Mat pred_w_h_ratio;
    binary_op(w_mat, h_mat, pred_w_h_ratio, Operation_DIV);

    ncnn::Mat change_mat2;
    binary_op_scalar(pred_w_h_ratio, w_h_ratio, change_mat2, Operation_RDIV);

    ncnn::Mat r_c;
    binary_op(change_mat2, change_mat2, r_c, Operation_CHANGE);

    ncnn::Mat penalty, pscore;
    binary_op(r_c, s_c, penalty, Operation_EXP, p->penalty_tk);
    binary_op(penalty, cls_score_sigmoid, pscore, Operation_MUL);

    // window penalty
    ncnn::Mat window_mat, pscore_mul, pscore_window, window_mat_mul;
    vector2ncnnmat(window, window_mat);
    binary_op_scalar(window_mat, p->window_influence, window_mat_mul, Operation_MUL);
    binary_op_scalar(pscore, 1-p->window_influence, pscore_mul, Operation_MUL);
    binary_op(pscore_mul, window_mat_mul, pscore_window, Operation_ADD);

    time5.stop();
    time5.show_distance("Update stage ---- postprocess cost time");

    // get max
    std::vector<int> index(3);
    for (int c=0; c < pscore_window.c; c++)
    {
        float *ptr = pscore_window.channel(c);
        float *max_ptr = ptr;
        for (int i=0; i < pscore_window.h; i++)
        {
            for (int j=0; j < pscore_window.w; j++)
            {
                if (ptr[j] >= *max_ptr)
                {
                    max_ptr = ptr + j;
                    index[0] = c; index[1] = i; index[2] = j;
                }
            }
            ptr += pscore_window.w;
        }
    }

    int r_max = index[1]; int c_max = index[2];

    std::cout << "pscore_window max score is: " << pscore_window.channel(index[0]).row(r_max)[c_max];

    // to real size
    float pred_x1_real = pred_x1.channel(index[0]).row(r_max)[c_max]; // pred_x1[r_max, c_max]
    float pred_y1_real = pred_y1.channel(index[0]).row(r_max)[c_max];
    float pred_x2_real = pred_x2.channel(index[0]).row(r_max)[c_max];
    float pred_y2_real = pred_y2.channel(index[0]).row(r_max)[c_max];

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

    target_sz[0] = target_sz[0] / scale_z;
    target_sz[1] = target_sz[1] / scale_z;

    // size learning rate
    float lr = penalty.channel(index[0]).row(r_max)[c_max]
            * cls_score_sigmoid.channel(index[0]).row(r_max)[c_max] * p->lr;

    // size rate
    auto res_xs = float (target_pos.x + diff_xs);
    auto res_ys = float (target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz[0];
    float res_h = pred_h * lr + (1 - lr) * target_sz[1];

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz[0] = target_sz[0] * (1 - lr) + lr * res_w;
    target_sz[1] = target_sz[1] * (1 - lr) + lr * res_h;

    cls_score_max = cls_score_sigmoid.channel(index[0]).row(r_max)[c_max];
}

void LightTrack::track(State &state, cv::Mat im)
{
    time_checker time1;

    Config *p = state.p;
    cv::Scalar avg_chans = state.avg_chans;
    std::vector<std::vector<float> > window = state.window;
    cv::Point target_pos = state.target_pos;
    cv::Scalar target_sz = state.target_sz;

    float hc_z = target_sz[1] + p->context_amount * (target_sz[0] + target_sz[1]);
    float wc_z = target_sz[0] + p->context_amount * (target_sz[0] + target_sz[1]);
    float s_z = sqrt(wc_z * hc_z);  // roi size
    float scale_z = p->exemplar_size / s_z;  // 127/

    float d_search = (p->instance_size - p->exemplar_size) / 2;  // backbone_model_size - init_model_size = 288-127
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;


    time1.start();
    cv::Mat x_crop;
    CropInfo crop_info;
    get_subwindow_tracking(im, x_crop, crop_info, target_pos, p->instance_size, round(s_x), avg_chans);
    time1.stop();
    time1.show_distance("Update stage ---- get subwindow cost time");

    state.x_crop = x_crop;

    // update
    target_sz[0] = target_sz[0] * scale_z;
    target_sz[1] = target_sz[1] * scale_z;

    float cls_score_max;
    this->update(x_crop, target_pos, target_sz, window, scale_z, p, cls_score_max);
    target_pos.x = max(0, min(state.im_w, target_pos.x));
    target_pos.y = max(0, min(state.im_h, target_pos.y));
    target_sz[0]= max(10., min(double(state.im_w), target_sz[0]));
    target_sz[1] = max(10., min(double(state.im_h), target_sz[1]));

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

float LightTrack::change(float r)
{
    return std::max(r, 1.f/r);
}

void LightTrack::sz(const ncnn::Mat& w, const ncnn::Mat& h, ncnn::Mat &out)
{
    // pad = (w + h) * 0.5
    ncnn::Mat tmp, pad;
    binary_op(w, h, tmp, Operation_ADD);

    ncnn::Mat scalar;
    scalar.create_like(tmp);
    scalar.fill(0.5f);

    binary_op(tmp, scalar, pad, Operation_MUL);

    // sz2 = (w + pad) * (h + pad)
    ncnn::Mat a1, a2, sz2;
    binary_op(w, pad, a1, Operation_ADD);
    binary_op(h, pad, a2, Operation_ADD);
    binary_op(a1, a2, sz2, Operation_MUL);

    // out = sqrt(sz2)
    ncnn::Mat sqrt_mat;
    sqrt_mat.create_like(sz2);
    sqrt_mat.fill(0.5f);
    binary_op(sz2, sqrt_mat, out, Operation_POW);
}

float LightTrack::sz_wh(cv::Scalar wh)
{
    float pad = (wh[1] + wh[0]) * 0.5;
    float sz2 = (wh[1] + pad) * (wh[0] + pad);
    return pow(sz2, 0.5f);
}

void LightTrack::normalize(cv::Mat &img)
{
    img.convertTo(img, CV_32FC3, 1.0/255, 0);
    img = (img - this->INPUT_MEAN) / this->INPUT_STD;
}

void LightTrack::grids(Config *p)
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = p->score_size;   // 18

    // the real shift is -param['shifts']
    int sz_x = floor(float(sz / 2));   // 9
    int sz_y = floor(float(sz / 2));

//    this->grid_to_search_x = (float **) malloc(sz * sizeof (float *));
//    this->grid_to_search_y = (float **) malloc(sz * sizeof (float *));
//
//    for (int i=0; i < sz; i++)
//    {
//        this->grid_to_search_x[i] = (float *) malloc(sz * sizeof (float ));
//        this->grid_to_search_y[i] = (float *) malloc(sz * sizeof (float ));
//    }
//
//    for (int i=0; i < sz; i++)
//        for (int j=0; j < sz; j++)
//        {
//            this->grid_to_search_x[i][j] = (j - sz_x)*p->total_stride + p->instance_size;
//            this->grid_to_search_y[i][j] = (i - sz_y)*p->total_stride + p->instance_size;
//        }

    this->grid_to_search_x.create(sz, sz, 1);   // (18,18,1)
    this->grid_to_search_y.create(sz, sz, 1);
    for (int c=0; c < grid_to_search_x.c; c++)
    {
        float *grid_x_ptr = this->grid_to_search_x.channel(c);
        float *grid_y_ptr = this->grid_to_search_y.channel(c);

        for (int i=0; i < grid_to_search_x.h; i++) {
            for (int j = 0; j < grid_to_search_x.w; j++) {
                grid_x_ptr[j] = (j - sz_x) * p->total_stride + p->instance_size / 2;
                grid_y_ptr[j] = (i - sz_y) * p->total_stride + p->instance_size / 2;
            }
            grid_x_ptr += grid_to_search_x.w;
            grid_y_ptr += grid_to_search_x.w;
        }
    }
}

void LightTrack::get_subwindow_tracking(const cv::Mat &im, cv::Mat &out, CropInfo &crop_info, cv::Point pos, int model_sz, float original_sz, cv::Scalar avg_chans)
{
    /*
    SiamFC type cropping
    */
    // model_sz: 127 for init, 288 for backbone
    // original_sz: search roi
    cv::Size im_sz = im.size();
    float center = (original_sz + 1) / 2;

    // context rect is search range of original image
    float context_xmin = round(float(pos.x) - center);
    float context_xmax = context_xmin + original_sz - 1;
    float context_ymin = round(float(pos.y) - center);
    float context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(max(0.f, -context_xmin));
    int top_pad = int(max(0.f, -context_ymin));
    int right_pad = int(max(0.f, context_xmax - im_sz.width + 1));
    int bottom_pad = int(max(0.f, context_ymax - im_sz.height +1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    int rows = im.rows;
    int cols = im.cols;
    cv::Mat im_path_original, tete_im;

    if (top_pad || bottom_pad || left_pad || right_pad)
    {
        cv::Mat te_im(rows + top_pad + bottom_pad, cols + left_pad + right_pad, CV_8UC3, cv::Scalar::all(0));
        // for return mask
        tete_im.create(rows + top_pad + bottom_pad, cols + left_pad + right_pad, CV_8SC1);
        tete_im.setTo(cv::Scalar(0));

        //
//        te_im.colRange(left_pad, left_pad+cols).rowRange(top_pad, top_pad+rows);
        cv::Mat roi;
        roi = te_im(cv::Range(top_pad, top_pad+rows), cv::Range(left_pad, left_pad+cols));
        im.copyTo(roi);

        if (top_pad)
        {
            roi = te_im(cv::Range(0, top_pad), cv::Range(left_pad, left_pad+cols));
            roi.setTo(avg_chans);
        }
        if (bottom_pad)
        {
            roi = te_im(cv::Range(top_pad + rows, rows + top_pad + bottom_pad), cv::Range(left_pad, left_pad+cols));
            roi.setTo(avg_chans);
        }
        if (left_pad)
        {
            roi = te_im(cv::Range(0, rows+top_pad+bottom_pad), cv::Range(0, left_pad));
            roi.setTo(avg_chans);
        }
        if (right_pad)
        {
            roi = te_im(cv::Range(0, rows+top_pad+bottom_pad), cv::Range(left_pad+cols, left_pad+right_pad+cols));
            roi.setTo(avg_chans);
        }
        im_path_original = te_im(cv::Range(int(context_ymin), int(context_ymax)+1), cv::Range(int(context_xmin), int(context_xmax)+1));
    } else
    {
        tete_im.create(rows, cols, CV_8UC1);
        tete_im.setTo(cv::Scalar(0));
        im_path_original = im(cv::Range(int(context_ymin), int(context_ymax)+1), cv::Range(int(context_xmin), int(context_xmax)+1));
    }

   

    cv::Mat im_patch;
    if (model_sz != int(original_sz))
        //original_size resize to (288, 288)
        cv::resize(im_path_original, im_patch, cv::Size(model_sz, model_sz));
    else
        im_patch = im_path_original;

    out = im_patch;

    crop_info.crop_cords.resize(4);
    crop_info.crop_cords[0] = int(context_xmin);
    crop_info.crop_cords[1] = int(context_xmax);
    crop_info.crop_cords[2] = int(context_ymin);
    crop_info.crop_cords[3] = int(context_ymax);

    crop_info.empty_mask = tete_im;

    crop_info.pad_info.resize(4);
    crop_info.pad_info[0] = top_pad;
    crop_info.pad_info[1] = left_pad;
    crop_info.pad_info[3] = rows;
    crop_info.pad_info[4] = cols;
}