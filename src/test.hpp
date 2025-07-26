#include "utils.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <Frame.h>


namespace dataframe {

class TestSampler {
  public:
    double g;
    int num_samples;

    TestSampler(ExperimentParams& params) {
      num_samples = utils::get<int>(params, "sampler_num_samples", 10);
      g = utils::get<double>(params, "g", 1.0);
    }

    void add_samples(int t, SampleMap& samples) {
      std::vector<std::vector<double>> m(2, std::vector<double>(num_samples));
      for (int i = 0; i < num_samples; i++) {
        m[0][i] = g*i*t;
        m[1][i] = g*std::pow(i, 2)*t;
      }

      std::vector<size_t> shape = {2};
      utils::emplace(samples, "avg", utils::samples_to_dataobject(m, shape));
      utils::emplace(samples, "t_sampled", double(t));
    }
};

}
