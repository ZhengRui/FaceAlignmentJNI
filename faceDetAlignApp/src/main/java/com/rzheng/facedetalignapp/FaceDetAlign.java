package com.rzheng.facedetalignapp;

/**
 * Created by zerry on 16/7/2.
 */
public class FaceDetAlign {
    public FaceDetAlign(String cascadeFile, int alignedFacesCacheSize) {
        nativeFaceDetAlign = create(cascadeFile, alignedFacesCacheSize);
    }

    public void loadShapePredictor(String landmarksFilePath) {
        loadShapePredictor(nativeFaceDetAlign, landmarksFilePath);
    }

    // for  onPictureTaken(), calibrate followed by detection, width and height
    // in getPos() will be image size
    public byte[] droidJPEGCalibrate(byte[] jpegdata, int front1orback0, int orientCase) {
        return droidJPEGCalibrate(nativeFaceDetAlign, jpegdata, front1orback0, orientCase);
    }
    public void detectFromJPEG(byte[] jpegdata, boolean doalign) {
        detectFromJPEG(jpegdata, doalign);
    }

    // for onPreviewFrame(), width and height in getPos() will be preview size
    public void detectFromRaw(int width, int height, byte[] frmdata, int front1orback0, int orientCase, boolean doalign) {
        detectFromRaw(nativeFaceDetAlign, width, height, frmdata, front1orback0, orientCase, doalign);
    }

    public int[] getPos(int face0landmarks1, int cv0canvas1, int width, int height, int front1orback0, int orientCase) {
        return getPos(nativeFaceDetAlign, face0landmarks1, cv0canvas1, width, height, front1orback0, orientCase);
    }

    // get a pointer to "std::vector<cv::Mat> alignedFacesCache" for feature extraction, each Mat is 128x128
    public long getAlignedFacesAddr() {
        return getAlignedFacesAddr(nativeFaceDetAlign);
    }

    public void clearAlignedFacesCache() {
        clearAlignedFacesCache(nativeFaceDetAlign);
    }

    private long nativeFaceDetAlign = 0;
    private static native long create(String cascadeFile, int alignedFacesCacheSize);
    private static native void loadShapePredictor(long thiz, String landmarksFilePath);

    private static native byte[] droidJPEGCalibrate(long thiz, byte[] jpegdata, int front1orback0, int orientCase);
    private static native void detectFromJPEG(long thiz, byte[] jpegdata, boolean doalign);

    private static native void detectFromRaw(long thiz, int width, int height, byte[] frmdata,
                                             int front1orback0, int orientCase, boolean doalign);
    private static native int[] getPos(long thiz, int face0landmarks1, int cv0canvas1, int width,
                                       int height, int front1orback0, int orientCase);
    private static native long getAlignedFacesAddr(long thiz);
    private static native void clearAlignedFacesCache(long thiz);
}
