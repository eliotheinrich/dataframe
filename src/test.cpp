#include <iostream>
#include <variant>
#include <glaze/glaze.hpp>
#include "Frame.h"
#include "utils.hpp"

using namespace dataframe;
using namespace dataframe::utils;

int main(int argc, char* argv[]) {
  DataFrame frame = DataFrame::from_file("/Users/eliotheinrich/Projects/plots/data/qrpm_21_t.eve");
  std::cout << frame.to_json() << std::endl;
  std::string key = "surface";
  Params constraints;
  constraints.emplace("system_size", 128.0);

  auto query_result = frame.query(key, constraints);
}
