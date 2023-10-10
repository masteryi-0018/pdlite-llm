# # 1. build paddle-lite linux x86
# bash ./lite/tools/build_linux.sh --arch=x86 --with_extra=ON
# cd ..

# # 2. copy .h and .so
# cp -r ./Paddle-Lite/build.lite.linux.x86.gcc/inference_lite_lib/cxx/ .
# cp -r ./Paddle-Lite/build.lite.linux.x86.gcc/inference_lite_lib/third_party/mklml/ .

# 3. make this project
#!/bin/bash
set -e
set -x

WITH_METAL=OFF

function print_usage() {
    echo "---------------------------------------------------------------------------------------------------------------------------------------- "
    echo -e "| usage:                                                                                                                             |"
    echo "---------------------------------------------------------------------------------------------------------------------------------------- "
    echo -e "|     ./build.sh help                                                                                                                |"
    echo "---------------------------------------------------------------------------------------------------------------------------------------- "
}


# parse command
function init() {
  for i in "$@"; do
    case $i in
      --with_metal=*)
          WITH_METAL="${i#*=}"
          shift
          ;;
      help)
          print_usage
          exit 0
          ;;
      *)
          # unknown option
          print_usage
          exit 1
          ;;
    esac
  done
}

init $@
mkdir ./build
cd ./build

if [ "${WITH_METAL}" == "ON" ]; then
  cmake .. -DMETAL=ON
else
  cmake ..
fi
make
cd ..
rm -rf ./build