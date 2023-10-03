from paddlelite.lite import *

def optimize(modelpath, savepath):
    opt=Opt()
    opt.set_model_dir(modelpath)
    opt.set_valid_places("x86")
    opt.set_model_type("naive_buffer")
    opt.set_optimize_out(savepath)
    opt.run()

for index in range(3):
    modelpath="./chatglm2-6b/block_" + str(index) + "/quantized_model"
    savepath="./chatglm2-6b-opt/block_" + str(index)
    optimize(modelpath, savepath)

modelpath="./chatglm2-6b/lm/quantized_model"
savepath="./chatglm2-6b-opt/lm"
optimize(modelpath, savepath)

modelpath="./chatglm2-6b/embedding/quantized_model"
savepath="./chatglm2-6b-opt/embedding"
optimize(modelpath, savepath)