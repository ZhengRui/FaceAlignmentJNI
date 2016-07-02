#!/usr/bin/env sh

mkdir -p build
cd build

ACMAKEFD="${HOME}/Work/Libs/android-cmake"
NDK_ROOT="${HOME}/Work/Libs/android-ndk-r10e"
OpenCV_DIR="${HOME}/AndroidStudioProjects/OpenCV-android-sdk/sdk/native/jni"

cmake -DCMAKE_TOOLCHAIN_FILE=${ACMAKEFD}/android.toolchain.cmake \
      -DANDROID_NDK=${NDK_ROOT} \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="armeabi-v7a with NEON" \
      -DANDROID_NATIVE_API_LEVEL=21 \
      -DOpenCV_DIR=${OpenCV_DIR} \
      ..

make VERBOSE=1

cp libfacedet.so ../../faceDetAlignApp/src/main/jniLibs/armeabi-v7a/


