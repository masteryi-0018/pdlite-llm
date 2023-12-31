# pdlite-llm

## 环境准备

- cmake
- gcc
- g++
- conda

```shell
conda create -n llm python=3.8
conda activate llm
pip install -r requirements.txt
```

## 模型准备

可以根据自己的需要，将原始模型转化至onnx格式；本项目借鉴[https://github.com/wangzhaode/llm-export](https://github.com/wangzhaode/llm-export)，直接下载了其onnx模型，使用chatglm2-6b的模型进行测试

## 模型转换

使用x2paddle将onnx转换为paddle的格式，需要对其源码进行一定修改，路径`anaconda3/envs/llm/lib/python3.8/site-packages/x2paddle/op_mapper/onnx2paddle/opset_legacy.py`，1133行添加：

```py
if axes == [-1]:
    axes = [0]
```

具体原因还不知道，先把其强行设置成为0

执行脚本

```shell
bash convert.sh
```

## 模型量化

暂时使用paddleslim，动态离线量化，并没有显著的效果

## 源码编译

```shell
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
```

将`Paddle-Lite/cmake/generic.cmake`284行修改为：
```cmake
target_link_libraries(${TARGET_NAME} ${MKLML_LIB_DIR}/libiomp5.so)
```

执行
```shell
bash linux_build.sh
```

## 使用

在源码中已经将转换好的模型路径写进去了，直接执行：

```shell
./pdlite-llm
```

即可

## 遗留问题

1. 在run()结束后，无法将模型运行结果作为返回值返回给主程序
2. log:
```
other has not been implemented transform with dtype3 X, dtype0 Out
*** Check failure stack trace: ***
``````
文件`Paddle-Lite/lite/kernels/host/cast_compute.cc`中，
- // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
- // SIZE_T = 19;UINT8 = 20;INT8 = 21;

可以发现问题原因是对于cast算子，输入类型int64无法转换为输出类型bool