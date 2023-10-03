from paddleslim.quant import quant_post_dynamic
import paddle
paddle.enable_static()

def quantize(modelpath, savepath):
    quant_post_dynamic(
        model_dir=modelpath,
        save_model_dir=savepath,
        model_filename='model.pdmodel',
        params_filename='model.pdiparams',
        save_model_filename='model.pdmodel',
        save_params_filename='model.pdiparams'
        )

for index in range(3):
    modelpath="./chatglm2-6b/block_" + str(index) + "/inference_model"
    savepath="./chatglm2-6b/block_" + str(index)
    quantize(modelpath, savepath)

modelpath="./chatglm2-6b/lm/inference_model"
savepath="./chatglm2-6b/lm"
quantize(modelpath, savepath)

modelpath="./chatglm2-6b/embedding/inference_model"
savepath="./chatglm2-6b/embedding"
quantize(modelpath, savepath)