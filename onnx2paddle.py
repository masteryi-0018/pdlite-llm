from x2paddle.convert import onnx2paddle

def onnx2fluid(modelpath, savepath):
    onnx2paddle(model_path=modelpath,
                save_dir=savepath,
                input_shape_dict=None,
                convert_to_lite=False,
                lite_valid_places="x86",
                lite_model_type="naive_buffer",
                disable_feedback=False,
                enable_onnx_checker=True)

def onnx2nb(modelpath, savepath):
    onnx2paddle(model_path=modelpath,
                save_dir=savepath,
                input_shape_dict=None,
                convert_to_lite=True,
                lite_valid_places="x86",
                lite_model_type="naive_buffer",
                disable_feedback=False,
                enable_onnx_checker=True)

import sys
index = sys.argv[1]
if index != '-1':
    modelpath="../../model/chatglm2-6b/block_" + str(index) + ".onnx"
    savepath="chatglm2-6b/block_" + str(index)
    onnx2fluid(modelpath, savepath)
else:
    modelpath="../../model/chatglm2-6b/lm.onnx"
    savepath="chatglm2-6b/lm"
    onnx2fluid(modelpath, savepath)

    modelpath="../../model/chatglm2-6b/embedding.onnx"
    savepath="chatglm2-6b/embedding"
    onnx2fluid(modelpath, savepath)