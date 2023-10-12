#include <chrono>  // NOLINT(build/c++11)
#include <cmath>
#include <iostream>
#include <vector>
#include "paddle_api.h"  // NOLINT
#include "cppjieba/Jieba.hpp"

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

// base64
typedef unsigned char BYTE;
static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(BYTE c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_decode(std::string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  BYTE char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++)
          ret.push_back(char_array_3[i]);
      i = 0;
    }
  }

  if (i) {
    for (j = i; j <4; j++)
      char_array_4[j] = 0;

    for (j = 0; j <4; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
  }

  return ret;
}

std::vector<int64_t> tokenizer_encode(std::string input_str) {
    std::vector<int64_t> ids;
    std::vector<std::string> words;
    std::string tokenizer_dir_ = "./tokenizer";
    std::string dict_path = tokenizer_dir_ + "/jieba.dict.utf8";
    std::string model_path = tokenizer_dir_ + "/hmm_model.utf8";
    std::string user_dict_path = tokenizer_dir_ + "/user.dict.utf8";
    std::string idf_path = tokenizer_dir_ + "/idf.utf8";
    std::string stopWord_path = tokenizer_dir_ + "/stop_words.utf8";
    std::vector<std::string> word_decoder_;
    std::unordered_map<std::string, int> word_encoder_;
    // load vocab
    {
        std::string model_name_ = "Chatglm2_6b";
        std::string vocab_path = tokenizer_dir_ + "/" + model_name_ + "_vocab.txt";
        printf("load %s ... ", vocab_path.c_str());
        std::ifstream vocab_file(vocab_path);
        int index = 0;
        std::string word;
        while (vocab_file >> word) {
            word = base64_decode(word);
            word_decoder_.push_back(word);
            word_encoder_.insert(std::make_pair<std::string, int>(std::move(word), index++));
        }
        printf("Done!\n");
    }
    // encode
    cppjieba::Jieba jieba(
        dict_path,
        model_path,
        user_dict_path,
        idf_path,
        stopWord_path
    );
    jieba.Cut(input_str, words, true);
    for (auto word : words) {
        const auto& iter = word_encoder_.find(word);
        if (iter != word_encoder_.end()) {
            ids.push_back(iter->second);
        }
    }
    return ids;
}

// Chatglm2_6b
std::vector<int64_t> tokenizer(const std::string& query) {
    auto prompt = "\n问：\n" + query + "答：\n";
    auto ids = tokenizer_encode(prompt);
    ids.insert(ids.begin(), 64792);
    ids.insert(ids.begin(), 64790);
    return ids;
}

std::vector<std::vector<std::vector<std::vector<int64_t>>>> gen_attention_mask(int seq_len) {
    std::vector<std::vector<std::vector<std::vector<int64_t>>>> attention_mask(1, \
      std::vector<std::vector<std::vector<int64_t>>>(1, std::vector<std::vector<int64_t>>(seq_len, \
      std::vector<int64_t>(seq_len, 0))));
    if (seq_len > 1) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                attention_mask[0][0][i][j] = j > i;
            }
        }
    } else {
        attention_mask[0][0][0][0] = 0;
    }
    return attention_mask;
}

std::vector<int64_t> gen_position_ids(int seq_len) {
    std::vector<int64_t> position_ids(seq_len, 0);
    if (seq_len == 1) {
        position_ids[0] = 0; // TODO: gen_seq_len_
    } else {
        for (int i = 0; i < seq_len; i++) {
            position_ids[i] = i;
        }
    }
    return position_ids;
}

// bool Chatglm2_6b::is_stop(int token_id) {
//     return token_id <= 2;
// }

void RunModel(std::string model_dir,
              std::vector<int64_t>& input_ids,
              size_t repeats,
              size_t warmup,
              size_t print_output_elem
              ) {
  // 1. Set MobileConfig
  MobileConfig config;
  config.set_model_from_file(model_dir);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);

  // 3. Prepare input data
  auto input_tensor = predictor->GetInput(0); // input_tensor->shape(): []
  shape_t input_shape = {int64_t(input_ids.size())};
  input_tensor->Resize(input_shape);
  size_t memory_size = input_ids.size() * sizeof(int64_t);
  input_tensor->ShareExternalMemory(static_cast<void*>(input_ids.data()),
                                    memory_size,
                                    TargetType(2));

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
            << "input_shape(s) (NCHW):" << ShapePrint(input_shape) << "\n"
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
  }

  std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
  std::cout << "output_tensor: " << output_tensor->shape() << std::endl;


  /* ------------------ block run model ------------------ */


  int seq_len = input_ids.size();
  auto inputs_ids_ = input_ids; //  to del
  auto* hidden_states = const_cast<Tensor*>(output_tensor.get());
  std::cout << "hidden_states: " << hidden_states->shape() << std::endl;
  auto attention_mask = gen_attention_mask(seq_len);
  auto position_ids = gen_position_ids(seq_len);
  int id = -1;
  int layer_nums_ = 1;
  model_dir = "./chatglm2-6b-opt/block_0.nb";

  // 1. Set MobileConfig
  MobileConfig config_1;
  config_1.set_model_from_file(model_dir);

  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor_1 =  CreatePaddlePredictor<MobileConfig>(config_1);

  // 3. Prepare input data
  auto input_tensor_1 = predictor_1->GetInput(0);
  shape_t input_shape_1 = hidden_states->shape();
  input_tensor_1->Resize(input_shape_1);
  size_t memory_size_1 = sizeof(float);
  for (auto s : input_shape_1) {
    memory_size_1 *= s;
  }
  input_tensor_1->ShareExternalMemory(static_cast<void*>(hidden_states),
                                    memory_size_1,
                                    TargetType(2));
  std::cout << "gy1 ++++++" << std::endl;

  auto input_tensor_2 = predictor_1->GetInput(1);
  shape_t input_shape_2 = {1, 1, int64_t(attention_mask[0][0].size()), int64_t(attention_mask[0][0].size())};
  input_tensor_2->Resize(input_shape_2);
  size_t memory_size_2 = sizeof(float);
  for (auto s : input_shape_2) {
    memory_size_2 *= s;
  }
  input_tensor_2->ShareExternalMemory(static_cast<void*>(attention_mask.data()),
                                    memory_size_2,
                                    TargetType(2));
  std::cout << "gy2 ++++++" << std::endl;

  auto input_tensor_3 = predictor_1->GetInput(2);
  shape_t input_shape_3 = {1, int64_t(position_ids.size())};
  input_tensor_3->Resize(input_shape_3);
  size_t memory_size_3 = sizeof(float);
  for (auto s : input_shape_3) {
    memory_size_3 *= s;
  }
  input_tensor_3->ShareExternalMemory(static_cast<void*>(position_ids.data()),
                                    memory_size_3,
                                    TargetType(2));
  std::cout << "gy3 ++++++" << std::endl;

  // 4. Run predictor
  for (size_t widx = 0; widx < warmup; ++widx) {
    if (widx == 0) {
      timeInstance.startTimer();
      predictor_1->Run();
      first_duration = timeInstance.getCostTimer();
    } else {
      predictor_1->Run();
    }
  }

  sum_duration = 0.0;
  max_duration = 1e-5;
  min_duration = 1e5;
  avg_duration = -1;
  for (size_t ridx = 0; ridx < repeats; ++ridx) {
    timeInstance.startTimer();
    try {
      predictor_1->Run();
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
            // << "input_shape(s) (NCHW):" << ShapePrint(input_tensor_1->shape()) << "\n"
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
  output_tensor_num = predictor_1->GetOutputNames().size();
  std::cout << "output tensor num:" << output_tensor_num << std::endl;

  for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
        predictor_1->GetOutput(tidx);
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
  }
}


int main(int argc, char** argv) {
  int repeats = 1;
  int warmup = 1;
  int print_output_elem = 0;
  if (argc > 1) {
    std::cerr << "usage: ./" << argv[0] << "\n"
              << std::endl;
    return 0;
  }

  std::string model_dir = "./chatglm2-6b-opt/embedding.nb";
  std::string query = "你好";
  auto input_ids = tokenizer(query); // [64790, 64792, 54761, 31211, 39701, 55437, 31211]
  RunModel(model_dir, input_ids, repeats, warmup, print_output_elem);
  return 0;
}