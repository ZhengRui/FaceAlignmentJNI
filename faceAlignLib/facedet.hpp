/**
* @Author: Zheng Rui <zerry>
* @Date:   2016-03-31T12:23:38+08:00
* @Email:  rzhengphy@gmail.com
* @Project: Face Detection JNI Lib
* @Last modified by:   zerry
* @Last modified time: 2016-04-02T16:08:34+08:00
*/

#ifndef FACEDET_HPP
#define FACEDET_HPP

#include <android/log.h>

#define TAG "fdLib"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG ,__VA_ARGS__)

#include <string>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

class FaceDetector {
public:
    FaceDetector(std::string cascadePath, int alignedFacesCacheSize);
    ~FaceDetector();
    std::vector<cv::Rect> detect(int width, int height, unsigned char* frmCData, int front1orback0, int orientCase, bool doalign = false);
    std::vector<cv::Rect> detectMat(cv::Mat &BGRMat, bool doalign = false);
    std::vector<cv::Rect> getBbsFiltered();
    void loadShapePredictor(std::string spPath);
    std::vector<cv::Mat>* getAlignedFacesCacheAddr();
    void clearCache();
    void fromDroidCamToCV(cv::Mat &m, int front1orback0, int orientCase);

private:
    cv::CascadeClassifier facecascade;
    cv::Mat kernel;
    cv::Mat BGRMat, GRAYMat, bbHSV, skinMask, BGRMatToAlign, faceAligned;
    std::vector<cv::Rect> bbs, bbsFiltered;
    dlib::frontal_face_detector dlibDetector;
    dlib::shape_predictor dlibLandmarker;
    std::vector<cv::Mat> alignedFacesCache;
    int alignedFacesCacheSize;
};

#endif
