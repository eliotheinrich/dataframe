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
  auto pe = glz::read_json(params, content);
  if (pe) {
    throw std::invalid_argument(fmt::format("Error parsing Params: \n{}", glz::format_error(pe, content)));
  }
  
  return params;
}
