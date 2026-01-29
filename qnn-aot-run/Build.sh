cmake --build build
cmake --build build-android

export DEVICE_DIR=/data/local/tmp/qnn_samp
export MODEL_PATH=/workspace/qnn-sample/qnn-aot-run/LinearHtpContext.bin

./build/QnnAOT 

adb push ${MODEL_PATH} ${DEVICE_DIR}
adb push ./build-android/QnnRun ${DEVICE_DIR}
