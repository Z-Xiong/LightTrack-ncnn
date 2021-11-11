#include <iostream>
#include <cstdlib>
#include <string>
#include "opencv2/opencv.hpp"

#include "LightTrack.h"


void cxy_wh_2_rect(const cv::Point& pos, cv::Scalar sz, cv::Rect &rect)
{
    rect.x = max(0, pos.x - int(sz[0] / 2));
    rect.y = max(0, pos.y - int(sz[1] / 2));
    rect.width = int(sz[0]);
    rect.height = int(sz[1]);
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
    State state;
    cv::Point target_pos;
    target_pos.x = trackWindow.x + trackWindow.width / 2;
    target_pos.y = trackWindow.y + trackWindow.height / 2;
    siam_tracker->init(frame, target_pos,
                                    cv::Scalar {float(trackWindow.width), float(trackWindow.height)}, state);
    std::cout << "==========================" << std::endl;
    std::cout << "Init done!" << std::endl;
    std::cout << std::endl;

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
        siam_tracker->track(state, frame);
        std::cout << "==========================" << std::endl;
        std::cout << "Track done" << std::endl;
        std::cout << std::endl;

        // Calculate Frames per second (FPS)
        double fps = cv::getTickFrequency() / ((double)cv::getTickCount() - t);

        // Result to rect.
        cv::Rect rect;
        cxy_wh_2_rect(state.target_pos, state.target_sz, rect);

        // Draw rect.
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));

        
        // String fps_text = "FPS: ";
        // String fps_value = ""+fps;
        // cv::putText(frame, fps_text + fps_value, cv::Point (100, 50), 1, 2.0, cv::Scalar(50, 170, 50), 2);

        // Display FPS 
        std::cout << "FPS: " << fps << std::endl;

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
    std::string backbone_model = "model/lighttrack_backbone";
    std::string neck_head_model = "model/lighttrack_neck_head";

    // Get video path.
    const char* video_path = argv[1];

    // Build tracker.
    LightTrack *siam_tracker;
    siam_tracker = new LightTrack(init_model, backbone_model, neck_head_model);
    track(siam_tracker, video_path);

    return 0;
}