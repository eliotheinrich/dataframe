#include "utils.hpp"
#include <glaze/glaze.hpp>

dataframe::Params dataframe::utils::load_params(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::invalid_argument("Error opening file.");
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  std::string content = buffer.str();

  dataframe::Params params;
  auto pe = glz::read_json(params, content);
  if (pe) {
    std::string error_message = "Error parsing DataSlide: \n" + glz::format_error(pe, content);
    throw std::invalid_argument(error_message);
  }
  
  return params;
}
