#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#include "rs232.h"

using namespace std;



#ifdef __linux__

#include <unistd.h>
#define SLEEP_MS(_miliseconds) (usleep(1000*_miliseconds))

#else

#include <windows.h>
#define SLEEP_MS(_miliseconds) (Sleep(_miliseconds))

#endif



struct Camera {

    CvCapture *camera_;
    cv::Mat lastFrame_;

    struct NoCamera: std::exception {
        const char* what() const noexcept { return "No camera detected.\n"; }
    };
    struct NoFrame: std::exception {
        const char* what() const noexcept { return "Frame could not be obtained.\n"; }
    };

    Camera(): camera_(nullptr) {
        camera_ = cvCaptureFromCAM(0);
        if (!camera_) throw NoCamera();
    }

    ~Camera() {
        cvReleaseCapture(&camera_);
    }

    cv::Mat &getFrame() {
        IplImage *iplImg = cvQueryFrame(camera_);
        lastFrame_ = iplImg;
        if (lastFrame_.empty()) throw NoFrame();
        return lastFrame_;
    }

};


struct Video {

    CvCapture *video_;
    cv::Mat lastFrame_;

    struct NoVideo: std::exception {
        const char* what() const noexcept { return "Video file did not find.\n"; }
    };
    struct NoFrame: std::exception {
        const char* what() const noexcept { return "Frame could not be obtained.\n"; }
    };

    Video(char const *file): video_(nullptr) {
        video_ = cvCaptureFromFile(file);
        if (!video_) throw NoVideo();
    }

    ~Video() {
        cvReleaseCapture(&video_);
    }

    cv::Mat &getFrame() {
        IplImage *iplImg = cvRetrieveFrame(video_);
        lastFrame_ = iplImg;
        if (lastFrame_.empty()) throw NoFrame();
        return lastFrame_;
    }

};


class _base_MiniMaestro {

    int port_;

    struct CannotOpenPort: std::exception {
        const char* what() const noexcept { return "Cannot open port.\n"; }
    };

public:


    _base_MiniMaestro(int portNumber, int baudrate): port_(portNumber) {
        if (RS232_OpenComport(port_,baudrate)) throw CannotOpenPort();
    }
    ~_base_MiniMaestro() { RS232_CloseComport(port_); }

    inline void setPosition(uint8_t channel, uint16_t data) {
        uint8_t buf[] = {0x84,channel,data&0x7F,(data>>7)&0x7F};
        RS232_SendBuf(port_, buf, 4);
    }

    inline void setPosition(uint8_t channel, uint8_t data) {
        uint8_t buf[] = { 0xFF, channel, data };
        RS232_SendBuf(port_, buf, 3);
    }

    inline void setSpeed(uint8_t channel, uint16_t data) {
        uint8_t buf[] = {0x87,channel,data & 0x7F,(data>>7)&0x7F};
        RS232_SendBuf(port_, buf, 4);
    }

    inline void setAcceleration(uint8_t channel, uint16_t data) {
        uint8_t buf[] = {0x89,channel,data & 0x7F,(data>>7)&0x7F};
        RS232_SendBuf(port_, buf, 4);
    }

};




cv::Point detectClosestFace(cv::Mat const &frame, cv::Point const &center, cv::CascadeClassifier &face_cascade) {
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

    if (faces.empty()) return cv::Point(-1,-1);

    cv::Point closestFaceCenter;
    double previousError = std::numeric_limits<double>::max();
    for (auto &face : faces) {
        cv::Point tmpPoint(face.x + face.width*0.5,face.y + face.height*0.5);
        double actualError = cv::norm(tmpPoint - center);
        if (actualError < previousError) {
            closestFaceCenter = tmpPoint;
            previousError = actualError;
        }
    }

    return closestFaceCenter;
}


cv::Vec2d calculateError(cv::Point const &center, cv::Point const &face) {
    return cv::Vec2d(face.x - center.x,face.y - center.y);
}


void servoRegulator(_base_MiniMaestro &maestro, cv::Vec2d const &error) {
    static uint16_t lastPWM0 = 0;
    static uint16_t lastPWM1 = 0;

    uint16_t const K = 10;

    // bistable
    if (error[0] < 0)
        lastPWM0 += K;
    else if (error[0] > 0)
        lastPWM0 -= K;

    if (error[1] < 0)
        lastPWM1 += K;
    else if (error[1] > 0)
        lastPWM1 -= K;

    maestro.setPosition(0,lastPWM0);
    SLEEP_MS(1);
    maestro.setPosition(1,lastPWM1);
}


int main() {

    struct Image { // only for single-image test
        cv::Mat lastFrame_;
        Image(char const *file): lastFrame_(cv::imread(file, CV_LOAD_IMAGE_COLOR)) {}
        cv::Mat &getFrame() { return lastFrame_; }
    };

    //Camera source;
    //Video source("path to video file");
    Image source("FreeGreatPicture.com-28660-business-people.jpg");

    cvNamedWindow("Face tracker",1);

    cv::CascadeClassifier face_cascade("haarcascade_frontalface_alt.xml");

    //_base_MiniMaestro maestro(1,115200);

    //while (true)
    {
        source.getFrame();

        cv::Point center(source.lastFrame_.cols/2,source.lastFrame_.rows/2);
        cv::Point faceCenter = detectClosestFace(source.lastFrame_,center,face_cascade);
        cv::Vec2d error = calculateError(center,faceCenter);
        std::cout << "Error: (" << error.val[0] << "," << error.val[1] << ")\n";

        //servoRegulator(maestro,error);

        cv::imshow("Face tracker", source.lastFrame_);
        //if(cv::waitKey(1000) >= 0) break;
//        SLEEP_MS(100);
    }

    cv::waitKey();

    cvDestroyWindow("Face tracker");

    return 0;
}

