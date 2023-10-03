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
cd Paddle-Lite
```

将`cmake/generic.cmake`285行修改为：
```cmake
target_link_libraries(${TARGET_NAME} ${MKLML_LIB_DIR}/libiomp5.so)
```

执行
```shell
bash linux_build.sh
```