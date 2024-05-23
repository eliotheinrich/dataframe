#include <iostream>
#include <variant>
#include <glaze/glaze.hpp>
#include "Frame.h"
#include "utils.hpp"

using namespace dataframe;
using namespace dataframe::utils;

int main(int argc, char* argv[]) {
  DataFrame df;
  DataSlide ds;
  ds.add_data("test_data", 2);
  std::vector<std::vector<Sample>> test_data{{2.0, 3.0}, {3.0, 4.0}};
  ds.push_samples_to_data("test_data", test_data);

  ds.add_samples("test_samples", 2);
  std::vector<std::vector<double>> test_samples{{2.0, 3.0}, {3.0, 4.0}};
  ds.push_samples("test_samples", test_samples);

  df.add_slide(ds);
  std::cout << df.to_json() << "\n";
  df.average_samples_inplace();
  std::cout << df.to_json() << "\n";
}
