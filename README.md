#### Example

![ex1](./ex1.jpg)
![ex2](./ex2.jpg)

#### Steps

1. calibrate according orientation, camera index, check [this post](http://zhengrui.github.io/android-coordinates.html).

2. face detection using opencv haar cascade detector. (scaled to max edge size = 480)

3. filtering: for each face from previous step, apply skin filter + dlib hog based face detector. dlib detector is slower but with less false positive. (scaled to max edge size = 960)

4. alignment: dlib alignment function, very fast. (original size)

#### Compile

modify relevant paths in `faceDetAlignLib/build.sh` and `faceDetAlignLib/CMakeLists.txt`, run `build.sh`
