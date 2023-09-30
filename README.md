# pdlite-llm

## 模型准备

可以根据自己的需要，将原始模型转化至onnx格式；本项目借鉴[https://github.com/wangzhaode/llm-export](https://github.com/wangzhaode/llm-export)，直接下载了其onnx模型，使用chatglm2-6b的模型进行测试

## 模型转换

使用x2paddle将onnx转换为paddle的格式，需要对其源码进行一定修改，路径`anaconda3/envs/llm/lib/python3.8/site-packages/x2paddle/op_mapper/onnx2paddle/opset_legacy.py`，1133行添加：

```py
if axes == [-1]:
    axes = [0]
```

具体原因还不知道，先把其强行设置成为0

执行脚本`convert.py`

## 模型量化

待补充

## 源码编译

