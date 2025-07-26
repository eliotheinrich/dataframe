#include "utils.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <Frame.h>


namespace dataframe {

class TestSampler {
  public:
    double p;
    int num_samples;

    TestSampler(ExperimentParams& params) {
      num_samples = utils::get<int>(params, "num_samples", 1);
      p = utils::get<double>(params, "p", 1.0);
    }

    void add_samples(int t, SampleMap& samples) {
      std::vector<std::vector<double>> m(num_samples);
      for (int i = 0; i < num_samples; i++) {
        m[0][i] = p*i*t;
        m[1][i] = p*std::pow(i, 2)*t;
      }

      std::vector<size_t> shape = {2, static_cast<size_t>(num_samples)};
      utils::emplace(samples, "avg", utils::samples_to_dataobject(m, shape));
      utils::emplace(samples, "t", double(t));
    }
};

}
