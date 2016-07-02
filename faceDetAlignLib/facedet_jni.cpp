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

    JNIEXPORT jlong JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_create(JNIEnv* env, jclass, jstring cascadeFile, jint alignedFacesCacheSize) {
        // LOGD("native create() called.");
        jlong detector = 0;
        const char* cascadeFilePath = env->GetStringUTFChars(cascadeFile, NULL);
        detector = (jlong)new FaceDetector(std::string(cascadeFilePath), (int) alignedFacesCacheSize);
        env->ReleaseStringUTFChars(cascadeFile, cascadeFilePath);
        return detector;
    }

    JNIEXPORT void JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_loadShapePredictor(JNIEnv* env, jclass, jlong thiz, jstring landmarksFilePath) {
        const char* spPath = env->GetStringUTFChars(landmarksFilePath, NULL);
        ((FaceDetector*)thiz)->loadShapePredictor(spPath);
        env->ReleaseStringUTFChars(landmarksFilePath, spPath);
    }

    JNIEXPORT jbyteArray JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_droidJPEGCalibrate(JNIEnv* env, jclass, jlong thiz, jbyteArray jpegdata, jint front1orback0, jint orientCase) {
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

    JNIEXPORT void JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_detectFromJPEG(JNIEnv* env, jclass, jlong thiz, jbyteArray jpegdata, jboolean doalign) {
        jbyte* picjData = env->GetByteArrayElements(jpegdata, 0);
        uchar* buf = (uchar*) picjData;
        size_t len = env->GetArrayLength(jpegdata);
        std::vector<uchar> cdata(buf, buf+len);
        cv::Mat m = cv::imdecode(cdata, CV_LOAD_IMAGE_COLOR);

        ((FaceDetector*)thiz)->detectFaces(m);
        if (doalign)
            ((FaceDetector*)thiz)->alignFaces();

        env->ReleaseByteArrayElements(jpegdata, picjData, JNI_ABORT);
    }

    JNIEXPORT void JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_detectFromRaw(JNIEnv* env, jclass, jlong thiz, jint width, jint height, jbyteArray frmdata, jint front1orback0, jint orientCase, jboolean doalign) {
        // LOGD("native detect() called.");
        jbyte* frmjData = env->GetByteArrayElements(frmdata, 0);
        ((FaceDetector*)thiz)->detectRaw((int) width, (int) height, (unsigned char*) frmjData, (int) front1orback0, (int) orientCase, (bool) doalign);
        env->ReleaseByteArrayElements(frmdata, frmjData, JNI_ABORT);
    }

    JNIEXPORT jintArray JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_getPos(JNIEnv* env, jclass, jlong thiz, jint face0landmarks1, jint cv0canvas1, jint width, jint height, jint front1orback0, jint orientCase) {
        std::vector<cv::Rect> pos;
        if (!face0landmarks1)
            pos = ((FaceDetector*)thiz)->getFacesPos();
        else
            pos = ((FaceDetector*)thiz)->getLandmarksPos();

        jintArray posArr = env->NewIntArray(pos.size() * 4);
        jint posBuf[4];
        int p = 0;
        for(std::vector<cv::Rect>::const_iterator r = pos.begin(); r != pos.end(); r++) {
            if (!cv0canvas1) {
                posBuf[0] = r->x;
                posBuf[1] = r->y;
                posBuf[2] = r->x + r->width;
                posBuf[3] = r->y + r->height;
            } else {
                switch (orientCase) {
                    case 0:
                        posBuf[0] = front1orback0 ? height - r->x - r->width : r->x;
                        posBuf[1] = r->y;
                        posBuf[2] = posBuf[0] + r->width;
                        posBuf[3] = posBuf[1] + r->height;
                        break;

                    case 1:
                        posBuf[0] = r->y;
                        posBuf[1] = front1orback0 ? r->x : width - r->x - r->width;
                        posBuf[2] = posBuf[0] + r->height;
                        posBuf[3] = posBuf[1] + r->width;
                        break;

                    case 2:
                        posBuf[0] = front1orback0 ? r->x : height - r->x - r->width;
                        posBuf[1] = width - r->y - r->height;
                        posBuf[2] = posBuf[0] + r->width;
                        posBuf[3] = posBuf[1] + r->height;
                        break;

                    case 3:
                        posBuf[0] = height - r->y - r->height;
                        posBuf[1] = front1orback0 ? width - r->x - r->width : r->x;
                        posBuf[2] = posBuf[0] + r->height;
                        posBuf[3] = posBuf[1] + r->width;
                        break;

                    default:
                        LOGD("Wrong orientCase value, should be {0, 1, 2, 3}");
                        break;
                }
            }

            env->SetIntArrayRegion(posArr, p, 4, posBuf);
            p += 4;
        }

        return posArr;
    }


    JNIEXPORT jlong JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_getAlignedFacesAddr(JNIEnv* env, jclass, jlong thiz) {
        return (jlong) ((FaceDetector*)thiz)->getAlignedFacesCacheAddr();
    }

    JNIEXPORT void JNICALL Java_com_rzheng_facedetalignapp_FaceDetAlign_clearCache(JNIEnv* env, jclass, jlong thiz) {
        ((FaceDetector*)thiz)->clearCache();
    }


#ifdef __cplusplus
}
#endif
