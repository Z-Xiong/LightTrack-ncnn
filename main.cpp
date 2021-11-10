#include <iostream>
#include <cstdlib>
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
    cv::VideoCapture capture;
    bool ret;
    if (strlen(video_path)==1)
        ret = capture.open(atoi(video_path));
    else
        ret = capture.open(video_path);

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
    std::cout << "Start track init ..." << std::endl;
    std::cout << "==========================" << std::endl;
    cv::namedWindow("demo");
    cv::Rect trackWindow = cv::selectROI("demo", frame);
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
        capture >> frame;
        if (frame.empty())
            break;
        std::cout << "Start track ..." << std::endl;
        std::cout << "==========================" << std::endl;
        siam_tracker->track(state, frame);
        std::cout << "==========================" << std::endl;
        std::cout << "Track done" << std::endl;
        std::cout << std::endl;
        // result to rect
        cv::Rect rect;
        cxy_wh_2_rect(state.target_pos, state.target_sz, rect);
        // draw rect
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
        // display
        cv::imshow("demo", frame);
        cv::waitKey(33);
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

    // get model path
    std::string init_model = "model/lighttrack_init";
    std::string backbone_model = "model/lighttrack_backbone";
    std::string neck_head_model = "model/lighttrack_neck_head";

    // get video path
    const char* video_path = argv[1];

    // build tracker
    LightTrack *siam_tracker;
    siam_tracker = new LightTrack(init_model, backbone_model, neck_head_model);
    track(siam_tracker, video_path);

    return 0;
}