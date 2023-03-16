#include <iostream>
#include <cstdlib>
#include <string>
#include "opencv2/opencv.hpp"

#include "LightTrack.h"


void cxy_wh_2_rect(const cv::Point& pos, const cv::Point2f& sz, cv::Rect &rect)
{
    rect.x = max(0, pos.x - int(sz.x / 2));
    rect.y = max(0, pos.y - int(sz.y / 2));
    rect.width = int(sz.x);
    rect.height = int(sz.y);
}


double compareHist(cv::Mat src_origin_1, cv::Mat src_origin_2)
{
  // 转换到 HSV , 图片是RGB格式用CV_RGB2HSV
  cv::Mat src_1, src_2;
  cv::cvtColor( src_origin_1 , src_1 , cv::COLOR_BGR2HSV );
  cv::cvtColor( src_origin_2, src_2, cv::COLOR_BGR2HSV );

 // 对hue通道使用30个bin,对saturatoin通道使用32个bin
  int h_bins = 50; int s_bins = 60;
  int histSize[] = { h_bins, s_bins };

  // hue的取值范围从0到256, saturation取值范围从0到180
  float h_ranges[] = { 0, 256 };
  float s_ranges[] = { 0, 180 };

  const float* ranges[] = { h_ranges, s_ranges };
  // 使用第0和第1通道
  int channels[] = { 0, 1 };

 // 直方图
  cv::MatND src_1_hist,src_2_hist;
 // 计算HSV图像的直方图
  cv::calcHist( &src_1 , 1, channels, Mat(), src_1_hist, 2, histSize, ranges, true, false );
  cv::normalize( src_1_hist, src_1_hist, 0, 1, cv::NORM_MINMAX, -1, Mat() );
  cv::calcHist( &src_2 , 1, channels, Mat(), src_2_hist, 2, histSize, ranges, true, false );
  cv::normalize( src_2_hist, src_2_hist, 0, 1, cv::NORM_MINMAX, -1, Mat() );

  //对比方法
 double result = cv::compareHist( src_1_hist, src_2_hist, 0 );
 return result;
}


void track(LightTrack *siam_tracker, const char *video_path)
{
    // Read video.
    cv::VideoCapture capture;
    bool ret;
    if (strlen(video_path)==1)
        ret = capture.open(atoi(video_path));
    else
        ret = capture.open(video_path);

    // Exit if video not opened.
    if (!ret)
        std::cout << "Open cap failed!" << std::endl;

    // Read first frame.
    cv::Mat frame;

    bool ok = capture.read(frame);
    if (!ok)
    {
        std::cout<< "Cannot read video file" << std::endl;
        return;
    }

    // Select a rect.
    cv::namedWindow("demo");
    cv::Rect trackWindow = cv::selectROI("demo", frame);
    

    // Initialize tracker with first frame and rect.
    std::cout << "Start track init ..." << std::endl;
    std::cout << "==========================" << std::endl;
    cv::Point target_pos;
    target_pos.x = trackWindow.x + trackWindow.width / 2;
    target_pos.y = trackWindow.y + trackWindow.height / 2;
    siam_tracker->init(frame, target_pos, cv::Point2f {float(trackWindow.width), float(trackWindow.height)});
    std::cout << "==========================" << std::endl;
    std::cout << "Init done!" << std::endl;
    std::cout << std::endl;
    cv::Mat init_window;
    frame(trackWindow).copyTo(init_window);

    for (;;)
    {
        // Read a new frame.
        capture >> frame;
        if (frame.empty())
            break;

        // Start timer
        double t = (double)cv::getTickCount();

        // Update tracker.
        std::cout << "Start track ..." << std::endl;
        std::cout << "==========================" << std::endl;
        siam_tracker->track(frame);
        // Calculate Frames per second (FPS)
        double fps = cv::getTickFrequency() / ((double)cv::getTickCount() - t);

        // Result to rect.
        cv::Rect rect;
        cxy_wh_2_rect(siam_tracker->target_pos, siam_tracker->target_sz, rect);

        // Boundary judgment.
        cv::Mat track_window;
        if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= frame.cols && 0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= frame.rows)
        {
            frame(rect).copyTo(track_window);
            // 对比初始框和跟踪框的相似度，从而判断是否跟丢（因为LighTrack的得分输出不具有判别性，所以通过后处理引入判断跟丢机制）
            double score = compareHist(init_window, track_window);
            printf( "Similarity score= %f \n", score );

            // 显示初始框和跟踪框
            cv::imshow("init_window", init_window);
            cv::waitKey(10);
            cv::imshow("track_window", track_window);
            cv::waitKey(10);

            // 相似度大于0.5的情况才进行矩形框标注
//            if (score > 0.3)
//            {
                // Draw rect.
                cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
//            }
        }

        // Display FPS 
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "==========================" << std::endl;
        std::cout << "Track done" << std::endl;
        std::cout << std::endl;


        // Display result.
        cv::imshow("demo", frame);
        cv::waitKey(33);

        // Exit if 'q' pressed.
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }
    cv::destroyWindow("demo");
    capture.release();
}


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [videopath(file or camera)]\n", argv[0]);
        return -1;
    }

    // Get model path.
    std::string init_model = "model/lighttrack_init";
    std::string update_model = "model/lighttrack_update";

    // Get video path.
    const char* video_path = argv[1];

    // Build tracker.
    LightTrack *siam_tracker;
    siam_tracker = new LightTrack(init_model.c_str(), update_model.c_str());
    track(siam_tracker, video_path);

    return 0;
}