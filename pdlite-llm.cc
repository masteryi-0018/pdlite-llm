// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <iostream>
#include <vector>
#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////
// #include "paddle_use_kernels.h"  // NOLINT
// #include "paddle_use_ops.h"      // NOLINT

using namespace paddle::lite_api;  // NOLINT

class Timer {
 private:
  std::chrono::high_resolution_clock::time_point inTime, outTime;

 public:
  void startTimer() { inTime = std::chrono::high_resolution_clock::now(); }

  // unit millisecond
  float getCostTimer() {
    outTime = std::chrono::high_resolution_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(outTime - inTime)
            .count() /
        1e+3);
  }
};

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

std::string ShapePrint(const std::vector<shape_t>& shapes) {
  std::string shapes_str{""};
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    auto shape = shapes[shape_idx];
    std::string shape_str;
    for (auto i : shape) {
      shape_str += std::to_string(i) + ",";
    }
    shapes_str += shape_str;
    shapes_str +=
        (shape_idx != 0 && shape_idx == shapes.size() - 1) ? "" : " : ";
  }
  return shapes_str;
}

std::string ShapePrint(const shape_t& shape) {
  std::string shape_str{""};
  for (auto i : shape) {
    shape_str += std::to_string(i) + " ";
  }
  return shape_str;
}

std::vector<std::string> split_string(const std::string& str_in) {
  std::vector<std::string> str_out;
  std::string tmp_str = str_in;
  while (!tmp_str.empty()) {
    size_t next_offset = tmp_str.find(":");
    str_out.push_back(tmp_str.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return str_out;
}

std::vector<int64_t> get_shape(const std::string& str_shape) {
  std::vector<int64_t> shape;
  std::string tmp_str = str_shape;
  while (!tmp_str.empty()) {
    int dim = atoi(tmp_str.data());
    shape.push_back(dim);
    size_t next_offset = tmp_str.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return shape;
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
}

// Chatglm2_6b
// std::vector<int> tokenizer_encode(std::string input_str) {
//     std::vector<int> ids;
//     std::vector<std::string> words;
//     std::string dict_path = tokenizer_dir_ + "/jieba.dict.utf8";
//     std::string model_path = tokenizer_dir_ + "/hmm_model.utf8";
//     std::string user_dict_path = tokenizer_dir_ + "/user.dict.utf8";
//     std::string idf_path = tokenizer_dir_ + "/idf.utf8";
//     std::string stopWord_path = tokenizer_dir_ + "/stop_words.utf8";
//     cppjieba::Jieba jieba(
//         dict_path,
//         model_path,
//         user_dict_path,
//         idf_path,
//         stopWord_path
//     );
//     jieba.Cut(input_str, words, true);
//     for (auto word : words) {
//         const auto& iter = word_encoder_.find(word);
//         if (iter != word_encoder_.end()) {
//             ids.push_back(iter->second);
//         }
//     }
//     return ids;
// }

// std::vector<int> Chatglm2_6b::tokenizer(const std::string& query) {
//     auto prompt = "\n问：\n" + query + "答：\n";
//     auto ids = tokenizer_encode(prompt);
//     ids.insert(ids.begin(), 64792);
//     ids.insert(ids.begin(), 64790);
//     return ids;
// }

// VARP Chatglm2_6b::gen_attention_mask(int seq_len) {
//     auto attention_mask = _Input({1, 1, seq_len, seq_len}, NCHW, halide_type_of<int>());
//     auto ptr = attention_mask->writeMap<int>();
//     if (seq_len > 1) {
//         for (int i = 0; i < seq_len; i++) {
//             for (int j = 0; j < seq_len; j++) {
//                 ptr[seq_len * i + j] = j > i;
//             }
//         }
//     } else {
//         ptr[0] = 0;
//     }
//     return attention_mask;
// }

// VARP Chatglm2_6b::gen_position_ids(int seq_len) {
//     auto position_ids = _Input({seq_len}, NCHW, halide_type_of<int>());
//     auto ptr = position_ids->writeMap<int>();
//     if (seq_len == 1) {
//         ptr[0] = gen_seq_len_;
//     } else {
//         for (int i = 0; i < seq_len; i++) {
//             ptr[i] = i;
//         }
//     }
//     return position_ids;
// }

// bool Chatglm2_6b::is_stop(int token_id) {
//     return token_id <= 2;
// }

// int forward(const std::vector<int>& input_ids) {
//     int seq_len = input_ids.size();
//     auto inputs_ids_ = _Const(input_ids.data(), {seq_len}, NCHW, halide_type_of<int>());
//     auto attention_mask = gen_attention_mask(seq_len);
//     auto position_ids = gen_position_ids(seq_len);
//     int id = -1;
//     bool is_single_ = false;
//     int layer_nums_ = 3;
//     if (is_single_) {
//         // single model
//         // auto outputs = modules_.back()->onForward({inputs_ids_, attention_mask, position_ids, past_key_values_[0]});
//         // id = outputs[0]->readMap<int>()[0];
//         // past_key_values_[0] = outputs[1];
//         std::cout << "not single" << std::endl;
//     } else {
//         // split block models
//         // auto hidden_states = modules_[layer_nums_ + 1]->onForward({inputs_ids_})[0];
//         auto hidden_states = RunModel();
//         for (int i = 0; i < layer_nums_; i++) {
//             auto outputs = modules_[i]->onForward({hidden_states, attention_mask, position_ids, past_key_values_[i]});
//             hidden_states = outputs[0];
//             past_key_values_[i] = outputs[1];
//         }
//         auto outputs = modules_[layer_nums_]->onForward({hidden_states});
//         id = outputs[0]->readMap<int>()[0];
//     }
//     all_seq_len_ += seq_len;
//     gen_seq_len_++;
//     return id;
// }

void RunModel(std::string model_dir,
              const std::vector<shape_t>& input_shapes,
              size_t repeats,
              size_t warmup,
              size_t print_output_elem
              ) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_dir);

#ifdef METAL
  std::string metal_lib_path = "../../../metal/lite.metallib";
  config.set_metal_lib_path(metal_lib_path);
  config.set_metal_use_mps(true);
#else
  // NOTE: Use gpu with opencl, you should ensure:
  //  first, [compile **cpu+opencl** paddlelite
  //    lib](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/demo_guides/opencl.md);
  //  second, [convert and use opencl nb
  //    model](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/docs/user_guides/opt/opt_bin.md).

  bool is_opencl_backend_valid = false;
  try {
    is_opencl_backend_valid =
        ::IsOpenCLBackendValid(false /*check_fp16_valid = false*/);
  } catch (...) {
    std::cerr << "Paddle-Lite Exception Happened on OpenCL Valid Check!"
              << std::endl;
    // Fall back to cpu model
  }
  std::cout << "is_opencl_backend_valid:" << is_opencl_backend_valid
            << std::endl;
#endif

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data
  // for (int j = 0; j < input_shapes.size(); ++j) {
  auto input_tensor = predictor->GetInput(0);
  // external
  std::vector<int64_t> external_data(1 * 16, 1);
  input_tensor->Resize(std::vector<int64_t>({1, 16}));
  size_t memory_size = external_data.size() * sizeof(int64_t);
  input_tensor->ShareExternalMemory(static_cast<void*>(external_data.data()),
                                    memory_size,
                                    TargetType(2));
  // std::cout << "gy ++++++" << std::endl;
  // std::cout << external_data.data() << std::endl;
  // PrecisionType pty = input_tensor->precision();
  // std::cout << int(pty) << std::endl;
  // std::cout << memory_size << std::endl;
  // shape_t sp = input_tensor->shape();
  // for (auto s : sp) {
  //   std::cout << s << std::endl;
  // }
  // }

  // 4. Run predictor
  Timer timeInstance;
  double first_duration{-1};
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      timeInstance.startTimer();
      predictor->Run();
      first_duration = timeInstance.getCostTimer();
    } else {
      predictor->Run();
    }
  }

  double sum_duration = 0.0;
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    timeInstance.startTimer();
    try {
      predictor->Run();
    } catch (...) {
      std::cerr << "Paddle-Lite Exception Happened on Run()!" << std::endl;
      // Fall back to cpu model
      std::abort();
    }

    double duration = timeInstance.getCostTimer();
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    std::cout << "run_idx:" << ridx + 1 << " / " << repeats << ": " << duration
              << " ms" << std::endl;
    if (first_duration < 0) {
      first_duration = duration;
    }
  }
  avg_duration = sum_duration / static_cast<float>(repeats);
  std::cout << "\n======= benchmark summary =======\n"
            << "input_shape(s) (NCHW):" << ShapePrint(input_shapes) << "\n"
            << "model_dir:" << model_dir << "\n"
            << "warmup:" << warmup << "\n"
            << "repeats:" << repeats << "\n"
            << "*** time info(ms) ***\n"
            << "1st_duration:" << first_duration << "\n"
            << "max_duration:" << max_duration << "\n"
            << "min_duration:" << min_duration << "\n"
            << "avg_duration:" << avg_duration << "\n";

  // 5. Get output
  std::cout << "\n====== output summary ====== " << std::endl;
  size_t output_tensor_num = predictor->GetOutputNames().size();
  std::cout << "output tensor num:" << output_tensor_num << std::endl;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor->GetOutput(tidx);
    std::cout << "\n--- output tensor " << tidx << " ---" << std::endl;
    auto out_shape = output_tensor->shape();
    auto out_data = output_tensor->data<float>();
    auto out_mean = compute_mean<float>(out_data, ShapeProduction(out_shape));
    auto out_std_dev = compute_standard_deviation<float>(
        out_data, ShapeProduction(out_shape), true, out_mean);

    std::cout << "output shape(NCHW):" << ShapePrint(out_shape) << std::endl;
    std::cout << "output tensor " << tidx
              << " elem num:" << ShapeProduction(out_shape) << std::endl;
    std::cout << "output tensor " << tidx
              << " standard deviation:" << out_std_dev << std::endl;
    std::cout << "output tensor " << tidx << " mean value:" << out_mean
              << std::endl;

    // print output
    if (print_output_elem) {
      for (int i = 0; i < ShapeProduction(out_shape); ++i) {
        std::cout << "out[" << tidx << "][" << i
                  << "]:" << output_tensor->data<float>()[i] << std::endl;
      }
    }
  }
  // std::vector<std::unique_ptr<const paddle::lite_api::Tensor>> ret;
  // for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
  //   std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
  //       predictor->GetOutput(tidx);
  //   ret.emplace_back(output_tensor);
  // }
  // return ret;
}

int main(int argc, char** argv) {
  std::vector<std::string> str_input_shapes;
  std::vector<shape_t> input_shapes{
      {1, 16}};  // shape_t ==> std::vector<int64_t>

  int repeats = 1;
  int warmup = 1;
  int print_output_elem = 0;

  if (argc > 2 && argc < 6) {
    std::cerr << "usage: ./" << argv[0] << "\n"
              << "  <naive_buffer_model_dir>\n"
              << "  <raw_input_shapes>, eg: 1,3,224,224 for 1 input; "
                 "1,3,224,224:1,5 for 2 inputs\n"
              << "  <repeats>\n"
              << "  <warmup>\n"
              << "  <print_output>" << std::endl;
    return 0;
  }

  std::string model_dir = argv[1];
  if (argc >= 6) {
    input_shapes.clear();
    std::string raw_input_shapes = argv[2];
    std::cout << "raw_input_shapes: " << raw_input_shapes << std::endl;
    str_input_shapes = split_string(raw_input_shapes);
    for (size_t i = 0; i < str_input_shapes.size(); ++i) {
      std::cout << "input shape: " << str_input_shapes[i] << std::endl;
      input_shapes.push_back(get_shape(str_input_shapes[i]));
    }

    repeats = atoi(argv[3]);
    warmup = atoi(argv[4]);
    print_output_elem = atoi(argv[5]);
  }

  RunModel(model_dir, input_shapes, repeats, warmup, print_output_elem);

  return 0;
}
