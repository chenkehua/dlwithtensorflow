#OPENCV_DIR=/paddle-infer/opencv-3.4.7
OPENCV_DIR=~/Research/ckh/paddle-infer/test//opencv-3.4.7

BUILD_DIR=build
rm -rf ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake .. \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCMAKE_BUILD_TYPE=Debug
make -j
