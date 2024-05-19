#include <iostream>
#include <variant>
#include <glaze/glaze.hpp>
#include "Frame.h"
#include "utils.hpp"

using namespace dataframe;
using namespace dataframe::utils;

int main(int argc, char* argv[]) {
  std::string filename = "Users/eliotheinrich/Projects/hypergraph.json";


  for (int i = 0; i < 10000; i++) {
    DataSlide s1;
    s1.add_data("test_data");
    s1.push_samples_to_data("test_data", 0.1);

    std::vector<byte_t> bytes = s1.to_bytes();

    DataSlide s2(bytes);
  }

}
