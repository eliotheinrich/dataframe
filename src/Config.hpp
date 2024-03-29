#pragma once

#include "utils.hpp"
#include "DataSlide.hpp"
#include <memory>

namespace dataframe {

#define DEFAULT_NUM_RUNS 1

  class Config {
    public:
      Params params;

      Config(Params &params) : params(params) {
        num_runs = utils::get<int>(params, "num_runs", DEFAULT_NUM_RUNS);
      }

      virtual ~Config() {}

      uint32_t get_nruns() const { 
        return num_runs;
      }

      virtual DataSlide compute(uint32_t num_threads) {
        return DataSlide();
      }

      virtual std::shared_ptr<Config> clone() {
        return std::make_shared<Config>(params);
      }

    private:
      uint32_t num_runs;
  };

}
