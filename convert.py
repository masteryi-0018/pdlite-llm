from x2paddle.convert import onnx2paddle

def onnx2fluid(modelpath, savepath):
    onnx2paddle(model_path=modelpath,
                save_dir=savepath,
                input_shape_dict=None,
                convert_to_lite=False,
                lite_valid_places="arm",
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

modelpath="../../model/chatglm2-6b/block_0.onnx"
savepath="chatglm2-6b-fluid"

onnx2fluid(modelpath, savepath)
# onnx2nb(modelpath, savepath)