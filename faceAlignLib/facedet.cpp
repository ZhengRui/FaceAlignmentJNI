/**
* @Author: Zheng Rui <zerry>
* @Date:   2016-03-29T16:31:59+08:00
* @Email:  rzhengphy@gmail.com
* @Project: Face Detection JNI Lib
* @Last modified by:   zerry
* @Last modified time: 2016-04-03T07:10:30+08:00
*/


#include "facedet.hpp"

using namespace std;
using namespace cv;

FaceDetector::FaceDetector(string cascadePath, int cacheSize) {
    bool suc = facecascade.load(cascadePath);
    if (suc) {
        LOGD("Cascade File loaded.");
    } else {
        LOGD("failed to load Cascade File");
    }
    dlibDetector = dlib::get_frontal_face_detector();
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    alignedFacesCache.clear();
    alignedFacesCacheSize = cacheSize;
}


void FaceDetector::fromDroidCamToCV(Mat &m, int front1orback0, int orientCase) {
    // from android camera dataframe coordinate system to natural/viewport coordinate system.
    // this is the coordinate system in which opencv will see what we see.
    switch (orientCase) {
        case 0:
            transpose(m, m);
            flip(m, m, 1-front1orback0);
            break;
        case 1:
            flip(m, m, -1);
            break;
        case 2:
            transpose(m, m);
            flip(m, m, front1orback0);
            break;
        default:
            break;
    }

}

vector<Rect> FaceDetector::detectMat(Mat &BGRMat, bool doalign) {
    BGRMatToAlign = BGRMat;
    int width = BGRMat.cols;
    int height = BGRMat.rows;
    float scale = max(width, height) / 480.;
    resize(BGRMat, BGRMat, Size(round(BGRMat.cols / scale), round(BGRMat.rows / scale)), INTER_AREA);

    // LOGD("after transformation: %d x %d", BGRMat.cols, BGRMat.rows);

    // Way 1: cv face detection + skin filter: cv is faster than dlib and
    //        can detect small faces, bounding box is a bit too small, especially
    //        the chin part
    // face detection
    cvtColor(BGRMat, GRAYMat, CV_BGR2GRAY);
    equalizeHist(GRAYMat, GRAYMat);
    facecascade.detectMultiScale(GRAYMat, bbs, 1.1, 3, CV_HAAR_SCALE_IMAGE, cvSize(40, 40));
    // LOGD("1stStage: Detect %d faces", (int) bbs.size());

    // filters: skin + dlib
    bbsFiltered.clear();
    for(vector<Rect>::const_iterator r = bbs.begin(); r != bbs.end(); r++) {
        cvtColor(BGRMat(*r), bbHSV, CV_BGR2HSV);
        inRange(bbHSV, Scalar(0, 48, 60), Scalar(30, 255, 255), skinMask);
        dilate(skinMask, skinMask, kernel, Point(-1, -1), 2);
        skinMask = skinMask(Rect(r->width/4, r->height/5, r->width/2, r->height*4/5));
        if (sum(skinMask)[0] / (skinMask.rows * skinMask.cols * 255.0) <= 0.2)
            continue;
        // pass skin filter

        // issue: when patch is too small, cv can detect face inside but dlib can not
        //        if use dlib extra filter, requires BGRMat not scaled down too much
        // Rect patch = Rect(Point(max(r->x - 15, 0), max(r->y - 15, 0)), Point(min(r->x + r->width + 15, BGRMat.cols), min(r->y + r->height + 15, BGRMat.rows)));
        // dlib::cv_image<dlib::bgr_pixel> dlibpatch(BGRMat(patch));
        // vector<dlib::rectangle> dfaces = dlibDetector(dlibpatch);
        // LOGD("Dlib detected %d faces in this patch.", (int) dfaces.size());
        // if (!dfaces.size()) {
            // LOGD("CV false positive suspect.");
            // continue;
        // }
        // pass dlib strong filter


        bbsFiltered.push_back(Rect(r->x * scale, r->y * scale, r->width * scale, r->height * scale));
    }
    // LOGD("2ndStage: Detect %d faces", (int) bbsFiltered.size());

    /*
    // Way 2: pure dlib: dlib is slower and can not detect small faces
    //                   when it detects a face, it is truly a face
    //                   and bounding box is more accurate

    dlib::cv_image<dlib::bgr_pixel> dlibimg(BGRMat);
    vector<dlib::rectangle> dfaces = dlibDetector(dlibimg);
    LOGD("Dlib detected %d faces in this image.", (int) dfaces.size());

    bbsFiltered.clear();
    for(vector<dlib::rectangle>::const_iterator r = dfaces.begin(); r != dfaces.end(); r++) {
        bbsFiltered.push_back(Rect(r->left() * scale, r->top() * scale, r->width() * scale, r->height() * scale));
    }
    */

    if (doalign) {
        // fill aligned face cache for future feature extraction
        scale = max(width, height) / 960.;
        resize(BGRMatToAlign, BGRMatToAlign, Size(round(BGRMatToAlign.cols / scale), round(BGRMatToAlign.rows / scale)), INTER_AREA);

        int maxx = BGRMatToAlign.cols;
        int maxy = BGRMatToAlign.rows;
        // vector<Rect> bbsBigger;
        // vector<Rect> landMarks;
        // int landMarksIdx[15] = {36, 39, 42, 45, 31, 33, 35, 48, 54, 50, 51, 52, 55, 57, 59};
        Point2f anchorPts[3];
        anchorPts[0] = Point2f(29.08151817, 40.56148148);
        anchorPts[1] = Point2f(90.19503021, 39.43851852);
        anchorPts[2] = Point2f((40.68895721 + 80.59230804) / 2, (88.38999939 + 87.61001587) / 2);
        Point2f pts[3];
        dlib::cv_image<dlib::bgr_pixel> dlibimg(BGRMatToAlign);
        Mat warp_H(2, 3, CV_32FC1);
        if ((int) alignedFacesCache.size() >= alignedFacesCacheSize)
            alignedFacesCache.clear();

        for(vector<Rect>::iterator r = bbsFiltered.begin(); r != bbsFiltered.end(); ) {
            int l = r->x * 1.0 / scale;
            int t = r->y * 1.0 / scale;
            int w = r->width * 1.0 / scale;
            int h = r->height * 1.0 / scale;
            Rect patch = Rect(Point(max(l - 15, 0), max(t - 15, 0)), Point(min(l + w + 15, maxx), min(t + h + 15, maxy)));
            // LOGD("%d X %d, (%d, %d, %d, %d)", maxx, maxy, patch.x, patch.y, patch.width, patch.height);
            dlib::cv_image<dlib::bgr_pixel> dlibpatch(BGRMatToAlign(patch));
            vector<dlib::rectangle> dfaces = dlibDetector(dlibpatch);

            if (!dfaces.size()) {
                LOGD("CV false positive suspect, remove it.");
                bbsFiltered.erase(r);
            } else {
                // face alignment
                // bbsBigger.push_back(Rect(patch.x * scale, patch.y * scale, patch.width * scale, patch.height * scale));
                dlib::rectangle roi(patch.x + dfaces[0].left(), patch.y + dfaces[0].top(), patch.x + dfaces[0].right(), patch.y + dfaces[0].bottom());
                dlib::full_object_detection shape = dlibLandmarker(dlibimg, roi);
                pts[0].x = shape.part(36).x(), pts[0].y = shape.part(36).y();
                pts[1].x = shape.part(45).x(), pts[1].y = shape.part(45).y();
                pts[2].x = (shape.part(48).x() + shape.part(54).x()) / 2., pts[2].y = (shape.part(48).y() + shape.part(54).y()) / 2.;
                warp_H = getAffineTransform(pts, anchorPts);
                warpAffine(BGRMatToAlign, faceAligned, warp_H, Size(128, 128));
                // LOGD("Aligned face size: %d x %d", faceAligned.cols, faceAligned.rows);
                cvtColor(faceAligned, faceAligned, CV_BGR2GRAY);
                equalizeHist(faceAligned, faceAligned);
                faceAligned.convertTo(faceAligned, CV_32FC1);
                alignedFacesCache.push_back(faceAligned / 255.0);

                // landMarks to show
                // for (int i=0; i<26; i++) {
                    // landMarks.push_back(Rect(shape.part(landMarksIdx[i]).x() * scale, shape.part(landMarksIdx[i]).y() * scale, 1, 1));   // show eyes nose mouse
                    // landMarks.push_back(Rect((shape.part(i).x() - 1) * scale, (shape.part(i).y() - 1) * scale, 2 * scale, 2 * scale));  // show face boundary
                // }

                r++;
            }
        }

        // bbsFiltered.insert(bbsFiltered.end(), bbsBigger.begin(), bbsBigger.end());
        // bbsFiltered.insert(bbsFiltered.end(), landMarks.begin(), landMarks.end());
    }

    LOGD("aligned faces cache size: %d", (int) alignedFacesCache.size());

    return bbsFiltered;
}

vector<Rect> FaceDetector::detect(int width, int height, unsigned char *frmCData, int front1orback0, int orientCase, bool doalign) {
    Mat YUVMat(height + height / 2, width, CV_8UC1, frmCData);
    cvtColor(YUVMat, BGRMat, CV_YUV420sp2BGR);
    fromDroidCamToCV(BGRMat, front1orback0, orientCase);
    return detectMat(BGRMat, doalign);
}

void FaceDetector::loadShapePredictor(string spPath) {
    LOGD("Start loading face landmarks.");
    dlib::deserialize(spPath) >> dlibLandmarker;
    LOGD("Face landmarks loaded.");
}

std::vector<cv::Rect> FaceDetector::getBbsFiltered() {
    return bbsFiltered;
}

vector<Mat>* FaceDetector::getAlignedFacesCacheAddr() {
    return &alignedFacesCache;
}

void FaceDetector::clearCache() {
    alignedFacesCache.clear();
}
