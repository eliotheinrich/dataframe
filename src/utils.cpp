#include "utils.hpp"
#include <glaze/glaze.hpp>

dataframe::ExperimentParams dataframe::utils::load_params(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening file.");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  std::string content = buffer.str();

  dataframe::ExperimentParams params;
  auto parse_error = glz::read_json(params, content);
  if (parse_error) {
    throw std::invalid_argument(fmt::format("Error parsing ExperimentParams: \n{}", glz::format_error(parse_error, content)));
  }
  
  return params;
}

std::string dataframe::utils::params_to_string(const dataframe::ExperimentParams& params) {
  std::string s;
  auto write_error = glz::write_json(params, s);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing params to json: \n{}", glz::format_error(write_error, s)));
  }
  return glz::prettify_json(s);
}

std::vector<dataframe::byte_t> dataframe::utils::pkl_params(const dataframe::ExperimentParams& params) {
  std::vector<dataframe::byte_t> data;
  auto write_error = glz::write_beve(params, data);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing ExperimentParams to binary: \n{}", glz::format_error(write_error, data)));
  }
  return data;
}

void dataframe::utils::load_params_from_pkl(dataframe::ExperimentParams& params, const std::vector<dataframe::byte_t>& bytes) {
  auto parse_error = glz::read_beve(params, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error parsing ExperimentParams from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
}
