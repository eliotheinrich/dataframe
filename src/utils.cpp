#include "utils.hpp"
#include <glaze/glaze.hpp>

using namespace dataframe;
using namespace dataframe::utils;

size_t dataframe::utils::shape_size(const std::vector<size_t>& shape) {
  size_t n = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    n *= shape[i];
  }
  return n;
}

std::tuple<double, double, size_t> dataframe::utils::sample_statistics(const std::vector<double>& values) {
  size_t nsamples = values.size();

  double mean = std::accumulate(values.begin(), values.end(), 0.0) / nsamples;

  auto variance_func = [&mean, &nsamples](double accumulator, const double& val) {
    return accumulator + (val - mean)*(val - mean) / (nsamples - 1);
  };

  double stddev = std::sqrt(std::accumulate(values.begin(), values.end(), 0.0, variance_func));

  return {mean, stddev, nsamples};
}

DataObject dataframe::utils::samples_to_dataobject(const std::vector<std::vector<double>>& data, const std::vector<size_t>& shape) {
  size_t data_size = dataframe::utils::shape_size(shape);

  std::vector<double> mean(data_size);
  std::vector<double> error(data_size);
  std::vector<size_t> nsamples(data_size);

  for (size_t i = 0; i < data_size; i++) {
    std::tie(mean[i], error[i], nsamples[i]) = dataframe::utils::sample_statistics(data[i]);
  }

  return {shape, mean, error, nsamples};
}

ExperimentParams dataframe::utils::load_params(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening file.");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  std::string content = buffer.str();

  ExperimentParams params;
  auto parse_error = glz::read_json(params, content);
  if (parse_error) {
    throw std::invalid_argument(fmt::format("Error parsing ExperimentParams: \n{}", glz::format_error(parse_error, content)));
  }
  
  return params;
}

std::string dataframe::utils::params_to_string(const ExperimentParams& params) {
  std::string s;
  auto write_error = glz::write_json(params, s);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing params to json: \n{}", glz::format_error(write_error, s)));
  }
  return glz::prettify_json(s);
}

std::vector<byte_t> dataframe::utils::pkl_params(const ExperimentParams& params) {
  std::vector<byte_t> data;
  auto write_error = glz::write_beve(params, data);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing ExperimentParams to binary: \n{}", glz::format_error(write_error, data)));
  }
  return data;
}

void dataframe::utils::load_params_from_pkl(ExperimentParams& params, const std::vector<byte_t>& bytes) {
  auto parse_error = glz::read_beve(params, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error parsing ExperimentParams from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
}
