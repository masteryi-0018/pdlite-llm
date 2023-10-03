# 1. build paddle-lite linux x86
bash ./lite/tools/build_linux.sh --arch=x86

# 2. copy .h and .so
cp -r ./Paddle-Lite/build.lite.linux.x86.gcc/inference_lite_lib/cxx/include ./include/
cp ./Paddle-Lite/build.lite.linux.x86.gcc/inference_lite_lib/cxx/lib/libpaddle_full_api_shared.so ./lib/

# 3. make this project
mkdir build
cd build
cmake ..
make