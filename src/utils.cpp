#include "utils.hpp"
#include <glaze/glaze.hpp>

dataframe::Params dataframe::utils::load_params(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening file.");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  std::string content = buffer.str();

  dataframe::Params params;
  auto parse_error = glz::read_json(params, content);
  if (parse_error) {
    throw std::invalid_argument(fmt::format("Error parsing Params: \n{}", glz::format_error(parse_error, content)));
  }
  
  return params;
}

std::string dataframe::utils::params_to_string(const dataframe::Params& params) {
  std::string s;
  auto write_error = glz::write_json(params, s);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing params to json: \n{}", glz::format_error(write_error, s)));
  }
  return glz::prettify_json(s);
}
