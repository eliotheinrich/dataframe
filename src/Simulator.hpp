#pragma once

#include "DataSlide.hpp"

#include <memory>
#include <random>

namespace dataframe {

#define DEFAULT_RANDOM_SEED -1

  class Simulator {
    public:
      // All simulators are equipeed with a random number generator
      int rand() { 
        return rng(); 
      }

      double randf() { 
        return double(rng())/double(RAND_MAX); 
      }

      Simulator(Params &params) {
        seed = utils::get<int>(params, "seed", DEFAULT_RANDOM_SEED);
        if (seed == DEFAULT_RANDOM_SEED) {
          thread_local std::random_device rd;
          rng.seed(rd());
        } else {
          rng.seed(seed);
        }
      }

      virtual ~Simulator() {}

      virtual void timesteps(uint32_t num_steps)=0;

      // By default, do nothing special during equilibration timesteps
      // May want to include, i.e., annealing 
      virtual void equilibration_timesteps(uint32_t num_steps) {
        timesteps(num_steps);
      }

      virtual data_t take_samples() {
        return data_t();
      }

      virtual void cleanup() {}

    private:
      int seed;

    protected:
      std::minstd_rand rng;
  };

}
