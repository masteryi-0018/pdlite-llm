#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include "paddle_api.h"


int main(int argc, char** argv) {
    // 0. Args
    std::string model_dir = "./chatglm2-6b-opt/embedding.nb";
    // std::string tokenizer_dir = "./chatglm2-6b-opt/embedding.nb";
    std::cout << "model path is " << model_dir << std::endl;
    // std::cout << "tokenizer path is " << tokenizer_dir << std::endl;

    // 1. Set CxxConfig
    CxxConfig config;
    config.set_model_dir(model_dir);
    config.set_valid_places({Place{TARGET(kX86), PRECISION(kFloat)},
                             Place{TARGET(kHost), PRECISION(kFloat)}});

    // 2. Create PaddlePredictor by CxxConfig
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<CxxConfig>(config);

    // 3. Save the optimized model
    // WARN: The `predictor->SaveOptimizedModel` method must be executed
    // before the `predictor->Run` method. Because some kernels' `PrepareForRun`
    // method maybe change some parameters' values.
    std::string FLAGS_optimized_model_dir = "opt_x86"
    predictor->SaveOptimizedModel(FLAGS_optimized_model_dir,
                                  LiteModelType::kNaiveBuffer);

    // 4. Prepare input data
    std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
    input_tensor->Resize(shape_t({1, 16}));
    auto* data = input_tensor->mutable_data<float>();
    for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
        data[i] = 1;
    }

    // 5. Run predictor
    int FLAGS_warmup = 10;
    int FLAGS_repeats = 10;
    for (int i = 0; i < FLAGS_warmup; ++i) {
        predictor->Run();
    }
    for (int j = 0; j < FLAGS_repeats; ++j) {
        predictor->Run();
    }

    // 6. Get output
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
    std::cout << "Output shape " << output_tensor->shape()[1] << std::endl;
    for (int i = 0; i < ShapeProduction(output_tensor->shape()); i++) {
        std::cout << "Output[" << i << "]: " << output_tensor->data<float>()[i]
                  << std::endl;
    }
    return 0;
}