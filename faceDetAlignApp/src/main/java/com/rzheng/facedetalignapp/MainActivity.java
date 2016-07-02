package com.rzheng.facedetalignapp;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.Camera;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import com.kyleduo.switchbutton.SwitchButton;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private static FaceDetAlign fdetector;
    private SwitchButton camSwitchBtn;
    private static int front1back0 = 0;
    private int orientCase;
    private Camera mCamera;
    private SurfaceView mSurfaceView;
    private SurfaceHolder mHolder;
    private boolean mInPreview = false, mCameraConfigured = false;
    private Camera.Size size;
    private byte[] callbackBuffer;
    private OrientationEventListener mOrientationListener;

    private static final String TAG = "FaceDetAlignApp";
    private static final String DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/FaceDetAlignApp/";
    private String landmarksFilePath = DATA_PATH + "shape_predictor_68_face_landmarks.dat";

    private boolean doalignment = true;
    private int[] faceArr;
    private int[] landmarksArr;
    private DrawOnTop mDraw;

    static {
        System.loadLibrary("facedet");
    }

    private void initialize() {
        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
        File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
        if (!cascadeFile.exists()) {
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            try {
                FileOutputStream os = new FileOutputStream(cascadeFile);
                byte[] buffer = new byte[4096];
                int bytesRead;
                try {
                    while ((bytesRead = is.read(buffer)) != -1) {
                        os.write(buffer, 0, bytesRead);
                    }
                    is.close();
                    os.close();

                } catch (IOException e) {
                    e.printStackTrace();
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

        if (fdetector == null) {
            fdetector = new FaceDetAlign(cascadeFile.getAbsolutePath(), 10);
            fdetector.loadShapePredictor(landmarksFilePath);
        }

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initialize();

        mSurfaceView = (SurfaceView) findViewById(R.id.surfaceView);
        mHolder = mSurfaceView.getHolder();
        mHolder.addCallback(surfaceCallback);

        camSwitchBtn = (SwitchButton) findViewById(R.id.camswitch);
        camSwitchBtn.setChecked(front1back0 == 0);
        camSwitchBtn.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                switchCam();
            }
        });

        mOrientationListener = new OrientationEventListener(this,
                SensorManager.SENSOR_DELAY_NORMAL) {
            @Override
            public void onOrientationChanged(int orientation) {
                if ((orientation >= 0 && orientation <= 30) || (orientation >= 330 && orientation <= 360)) {
                    orientCase = 0;
                } else if (orientation >= 60 && orientation <= 120) {
                    orientCase = 1;
                } else if (orientation >= 150 && orientation <= 210) {
                    orientCase = 2;
                } else if (orientation >= 240 && orientation <= 300) {
                    orientCase = 3;
                } else {
                }
//                Log.i(TAG, "Orientation changed to " + orientation +
//                        ", case " + orientCase);
            }
        };

    }

    private SurfaceHolder.Callback surfaceCallback = new SurfaceHolder.Callback() {
        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            Log.i(TAG, "surfaceCreated() called...");
        }

        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
            Log.i(TAG, " surfaceChanged() called.");
            initPreview(width, height);
            startPreview();
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            Log.i(TAG, " surfaceDestroyed() called.");
        }
    };

    Camera.PreviewCallback frameCallback = new Camera.PreviewCallback() {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            // process data frame
            fdetector.detectFromRaw(size.width, size.height, data, front1back0, orientCase, doalignment);
            faceArr = fdetector.getPos(0, 1, size.width, size.height, front1back0, orientCase);
            landmarksArr = fdetector.getPos(1, 1, size.width, size.height, front1back0, orientCase);
            mDraw.invalidate();

            mCamera.addCallbackBuffer(callbackBuffer);
        }
    };

    private void initPreview(int width, int height) {
        Log.i(TAG, "initPreview() called");
        if (mCamera != null && mHolder.getSurface() != null) {
            if (!mCameraConfigured) {
                Camera.Parameters params = mCamera.getParameters();
                size = params.getPreviewSize();
                for (Camera.Size s : params.getSupportedPreviewSizes()) {   // get 3840x2160 for back cam
//                    Log.i(TAG, "Supported preview size: " + s.width + ", " + s.height);
                    if (s.width > size.width)
                        size = s;
                }
                params.setPreviewSize(size.width, size.height);
                Log.i(TAG, "Preview size: " + size.width + ", " + size.height);
                callbackBuffer = new byte[(size.width + size.width / 2)* size.height];
                params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);

                mCamera.setParameters(params);

                if (mDraw == null) {
                    mDraw = new DrawOnTop(this);
                    addContentView(mDraw, new ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT,
                            ViewGroup.LayoutParams.WRAP_CONTENT));
                }
                mDraw.clearCanvas();

                mCameraConfigured = true;

                if (mOrientationListener.canDetectOrientation() == true) {
                    mOrientationListener.enable();
                }

            }

            try {
                mCamera.setPreviewDisplay(mHolder);
                mCamera.addCallbackBuffer(callbackBuffer);
                mCamera.setPreviewCallbackWithBuffer(frameCallback);
            } catch (Throwable t) {
                Log.e(TAG, "Exception in initPreview()", t);
            }
        }
    }


    private void startPreview() {
        Log.i(TAG, "startPreview() called");
        if (mCameraConfigured && mCamera != null) {
            mCamera.startPreview();
            mInPreview = true;
        }
    }

    private void switchCam() {
        if (mCamera != null && mInPreview) {
            mDraw.clearCanvas();
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
            mInPreview = false;
            mCameraConfigured = false;
        }

        front1back0 = 1 - front1back0;
        mCamera = Camera.open(front1back0);   // 0 for back, 1 for frontal
        mCamera.setDisplayOrientation(90);
        initPreview(size.width, size.height);
        startPreview();
    }

    @Override
    public void onResume() {
        Log.i(TAG, " onResume() called.");
        super.onResume();
        mCamera = Camera.open(front1back0);   // 0 for back, 1 for frontal
        mCamera.setDisplayOrientation(90);
        startPreview();
    }

    @Override
    public void onPause() {
        Log.i(TAG, " onPause() called.");
        mDraw.clearCanvas();
        if (mInPreview) {
            mCamera.stopPreview();
        }
        mCamera.setPreviewCallbackWithBuffer(null);
        mCamera.release();
        mCamera = null;
        mInPreview = false;
        mCameraConfigured = false; // otherwise cannot refocus after onResume
        mOrientationListener.disable();
        super.onPause();
    }

    @Override
    protected void onStop() {
        super.onStop();

    }

    @Override
    protected void onDestroy() {
        Log.i(TAG, " onDestroy() called.");
        super.onDestroy();
    }


    class DrawOnTop extends View {
        Paint facePen;
        Paint lmkPen;

        public DrawOnTop(Context context) {
            super(context);

            facePen = new Paint();
            facePen.setStyle(Paint.Style.STROKE);
            facePen.setStrokeWidth(3);
            facePen.setColor(Color.RED);

            lmkPen = new Paint();
            lmkPen.setStyle(Paint.Style.STROKE);
            lmkPen.setStrokeWidth(6);
            lmkPen.setColor(Color.GREEN);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);

            if (faceArr.length > 0) {
                for (int i = 0; i < faceArr.length; i += 4) {
                    canvas.drawRect(faceArr[i], faceArr[i+1], faceArr[i+2], faceArr[i+3], facePen);
                }
            }

            if (landmarksArr.length > 0) {
                for (int i = 0; i < landmarksArr.length; i += 4) {
                    canvas.drawRect(landmarksArr[i]-3, landmarksArr[i+1]-3, landmarksArr[i+2]+3, landmarksArr[i+3]+3, lmkPen);
                }
            }

        }

        public void clearCanvas() {
            faceArr = new int[] {};
            landmarksArr = new int[] {};
            this.invalidate();
        }
    }
}
