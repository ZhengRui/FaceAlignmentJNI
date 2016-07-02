/**
* @Author: Zheng Rui <zerry>
* @Date:   2016-03-31T12:46:55+08:00
* @Email:  rzhengphy@gmail.com
* @Project: Face Detection JNI Lib
* @Last modified by:   zerry
* @Last modified time: 2016-04-02T16:15:34+08:00
*/

#include <jni.h>
#include "facedet.hpp"

#ifdef __cplusplus
extern "C" {
#endif

    JNIEXPORT jlong JNICALL Java_com_rzheng_fdlib_FaceDetector_create(JNIEnv* env, jclass, jstring cascadeFile, jint alignedFacesCacheSize) {
        // LOGD("native create() called.");
        jlong detector = 0;
        const char* cascadeFilePath = env->GetStringUTFChars(cascadeFile, NULL);
        detector = (jlong)new FaceDetector(std::string(cascadeFilePath), (int) alignedFacesCacheSize);
        env->ReleaseStringUTFChars(cascadeFile, cascadeFilePath);
        return detector;
    }

    JNIEXPORT jlong JNICALL Java_com_rzheng_fdlib_FaceDetector_getAlignedFacesAddr(JNIEnv* env, jclass, jlong thiz) {
        return (jlong) ((FaceDetector*)thiz)->getAlignedFacesCacheAddr();
    }

    JNIEXPORT jbyteArray JNICALL Java_com_rzheng_fdlib_FaceDetector_droidJPEGCalibrate(JNIEnv* env, jclass, jlong thiz, jbyteArray jpegdata, jint front1orback0, jint orientCase) {
        jbyte* picjData = env->GetByteArrayElements(jpegdata, 0);
        uchar* buf = (uchar*) picjData;
        size_t len = env->GetArrayLength(jpegdata);
        std::vector<uchar> cdata(buf, buf+len);
        cv::Mat m = cv::imdecode(cdata, CV_LOAD_IMAGE_COLOR);

        // do calibration: rotate + flip
        ((FaceDetector*)thiz)->fromDroidCamToCV(m, front1orback0, orientCase);

        LOGD("picture size after calibrated: %d X %d", m.rows, m.cols);
        std::vector<int> params;
        params.push_back(CV_IMWRITE_JPEG_QUALITY);
        params.push_back(100);
        std::vector<uchar> cdataEnc;
        cv::imencode(".jpg", m, cdataEnc, params);
        jbyteArray jpegCalibrated = env->NewByteArray(cdataEnc.size());
        env->SetByteArrayRegion(jpegCalibrated, 0, cdataEnc.size(), (jbyte*)&cdataEnc[0]);

        env->ReleaseByteArrayElements(jpegdata, picjData, JNI_ABORT);
        return jpegCalibrated;
    }

    JNIEXPORT jbyteArray JNICALL Java_com_rzheng_fdlib_FaceDetector_boxesProcess(JNIEnv* env, jclass, jlong thiz, jbyteArray jpegdata, jobjectArray bbxposArr, jobjectArray bbxtxtArr, jbooleanArray bbxprocArr, jintArray bbxproctypeArr) {
        jbyte* imgjData = env->GetByteArrayElements(jpegdata, 0);
        uchar* buf = (uchar*) imgjData;
        size_t len = env->GetArrayLength(jpegdata);
        std::vector<uchar> cdata(buf, buf+len);
        cv::Mat img = cv::imdecode(cdata, CV_LOAD_IMAGE_COLOR);

        int bbxnum = env->GetArrayLength(bbxposArr);
        // LOGD("boxes number: %d", bbxnum);
        jboolean* bbxproc = env->GetBooleanArrayElements(bbxprocArr, 0);
        jint* bbxproctype = env->GetIntArrayElements(bbxproctypeArr, 0);
        cv::Scalar color;

        for (int i=0; i < bbxnum; i++) {
            jintArray bbxposJ = (jintArray) env->GetObjectArrayElement(bbxposArr, i);
            jint *bbxpos = env->GetIntArrayElements(bbxposJ, 0);

            jstring bbxtxtJ = (jstring) env->GetObjectArrayElement(bbxtxtArr, i);
            const char* bbxtxt = env->GetStringUTFChars(bbxtxtJ, NULL);

            // LOGD("box position: (%d, %d) - (%d, %d), text: %s, do blur: %s, blur type: %d", bbxpos[0], bbxpos[1], bbxpos[2], bbxpos[3], bbxtxt, bbxproc[i]?"yes":"no", bbxproctype[i]);

            if (bbxproctype[i] == 2) {
                color = cv::Scalar(0,0,255);
            } else {
                color = cv::Scalar(0,255,0);
            }

            cv::Rect roi = cv::Rect(cv::Point(bbxpos[0], bbxpos[1]), cv::Point(bbxpos[2], bbxpos[3]));

            if (bbxproc[i]) {
                if (!bbxproctype[i]) {
                    // blur face, default
                    cv::medianBlur(img(roi), img(roi), 77);
                } else {
                    // blur body
                }
            }

            cv::rectangle(img, roi, color, 2);
            cv::putText(img, bbxtxt, cv::Point(bbxpos[0], bbxpos[1]-10), cv::FONT_HERSHEY_DUPLEX, 0.8, color, 2);

            env->ReleaseIntArrayElements(bbxposJ, bbxpos, JNI_ABORT);
            env->ReleaseStringUTFChars(bbxtxtJ, bbxtxt);
        }
        env->ReleaseBooleanArrayElements(bbxprocArr, bbxproc, JNI_ABORT);
        env->ReleaseIntArrayElements(bbxproctypeArr, bbxproctype, JNI_ABORT);

        std::vector<int> params;
        params.push_back(CV_IMWRITE_JPEG_QUALITY);
        params.push_back(100);
        std::vector<uchar> cdataEnc;
        cv::imencode(".jpg", img, cdataEnc, params);
        jbyteArray jpegBoxsProcessed = env->NewByteArray(cdataEnc.size());
        env->SetByteArrayRegion(jpegBoxsProcessed, 0, cdataEnc.size(), (jbyte*)&cdataEnc[0]);

        env->ReleaseByteArrayElements(jpegdata, imgjData, JNI_ABORT);
        return jpegBoxsProcessed;
    }

    JNIEXPORT jbyteArray JNICALL Java_com_rzheng_fdlib_FaceDetector_detectAndBlurJPEG(JNIEnv* env, jclass, jlong thiz, jbyteArray jpegdata) {
        jbyte* imgjData = env->GetByteArrayElements(jpegdata, 0);
        uchar* buf = (uchar*) imgjData;
        size_t len = env->GetArrayLength(jpegdata);
        std::vector<uchar> cdata(buf, buf+len);
        cv::Mat img = cv::imdecode(cdata, CV_LOAD_IMAGE_COLOR);
        cv::Mat imgDet = img;

        std::vector<cv::Rect> bbsFiltered = ((FaceDetector*)thiz)->detectMat(imgDet, true);
        for(std::vector<cv::Rect>::iterator r = bbsFiltered.begin(); r != bbsFiltered.end(); r++) {
            cv::medianBlur(img(*r), img(*r), 77);
        }

        std::vector<int> params;
        params.push_back(CV_IMWRITE_JPEG_QUALITY);
        params.push_back(100);
        std::vector<uchar> cdataEnc;
        cv::imencode(".jpg", img, cdataEnc, params);
        jbyteArray jpegProcessed = env->NewByteArray(cdataEnc.size());
        env->SetByteArrayRegion(jpegProcessed, 0, cdataEnc.size(), (jbyte*)&cdataEnc[0]);

        env->ReleaseByteArrayElements(jpegdata, imgjData, JNI_ABORT);
        return jpegProcessed;
    }

    JNIEXPORT jintArray JNICALL Java_com_rzheng_fdlib_FaceDetector_getBbxPositions(JNIEnv* env, jclass, jlong thiz) {
        std::vector<cv::Rect> bbsFiltered = ((FaceDetector*)thiz)->getBbsFiltered();
        jintArray faceArr = env->NewIntArray(bbsFiltered.size() * 4);
        jint faceBuf[4];
        int p = 0;
        for(std::vector<cv::Rect>::const_iterator r = bbsFiltered.begin(); r != bbsFiltered.end(); r++) {
            faceBuf[0] = r->x;
            faceBuf[1] = r->y;
            faceBuf[2] = r->x + r->width;
            faceBuf[3] = r->y + r->height;
            env->SetIntArrayRegion(faceArr, p, 4, faceBuf);
            p += 4;
        }

        return faceArr;
    }


    JNIEXPORT jintArray JNICALL Java_com_rzheng_fdlib_FaceDetector_detect(JNIEnv* env, jclass, jlong thiz, jint width, jint height, jbyteArray frmdata, jint front1orback0, jint orientCase, jboolean doalign) {
        // LOGD("native detect() called.");
        jbyte* frmjData = env->GetByteArrayElements(frmdata, 0);
        // call some image processing function
        // LOGD("frame size: %d X %d, data length: %d", width, height, env->GetArrayLength(frmdata));
        std::vector<cv::Rect> bbsFiltered = ((FaceDetector*)thiz)->detect((int) width, (int) height, (unsigned char*) frmjData, (int) front1orback0, (int) orientCase, (bool) doalign);
        jintArray faceArr = env->NewIntArray(bbsFiltered.size() * 4);
        jint faceBuf[4];
        int p = 0;
        //LOGD("Orient case: %d, Camera index: %d", orientCase, front1orback0);
        for(std::vector<cv::Rect>::const_iterator r = bbsFiltered.begin(); r != bbsFiltered.end(); r++) {
            switch (orientCase) {
                case 0:
                    faceBuf[0] = front1orback0 ? height - r->x - r->width : r->x;
                    faceBuf[1] = r->y;
                    faceBuf[2] = faceBuf[0] + r->width;
                    faceBuf[3] = faceBuf[1] + r->height;
                    break;

                case 1:
                    faceBuf[0] = r->y;
                    faceBuf[1] = front1orback0 ? r->x : width - r->x - r->width;
                    faceBuf[2] = faceBuf[0] + r->height;
                    faceBuf[3] = faceBuf[1] + r->width;
                    break;

                case 2:
                    faceBuf[0] = front1orback0 ? r->x : height - r->x - r->width;
                    faceBuf[1] = width - r->y - r->height;
                    faceBuf[2] = faceBuf[0] + r->width;
                    faceBuf[3] = faceBuf[1] + r->height;
                    break;

                case 3:
                    faceBuf[0] = height - r->y - r->height;
                    faceBuf[1] = front1orback0 ? width - r->x - r->width : r->x;
                    faceBuf[2] = faceBuf[0] + r->height;
                    faceBuf[3] = faceBuf[1] + r->width;
                    break;

                default:
                    LOGD("Wrong orientCase value, should be {0, 1, 2, 3}");
                    break;
            }

            env->SetIntArrayRegion(faceArr, p, 4, faceBuf);
            p += 4;
        }

        env->ReleaseByteArrayElements(frmdata, frmjData, JNI_ABORT);
        return faceArr;
    }


    JNIEXPORT void JNICALL Java_com_rzheng_fdlib_FaceDetector_loadShapePredictor(JNIEnv* env, jclass, jlong thiz, jstring landmarksFilePath) {
        const char* spPath = env->GetStringUTFChars(landmarksFilePath, NULL);
        ((FaceDetector*)thiz)->loadShapePredictor(spPath);
        env->ReleaseStringUTFChars(landmarksFilePath, spPath);
    }


    JNIEXPORT void JNICALL Java_com_rzheng_fdlib_FaceDetector_clearCache(JNIEnv* env, jclass, jlong thiz) {
        ((FaceDetector*)thiz)->clearCache();
    }


#ifdef __cplusplus
}
#endif
