#pragma once

#include "types.h"
#include "DataSlide.hpp"
#include <chrono>

namespace dataframe {

#define DEFAULT_EQUILIBRATION_STEPS 0u
#define DEFAULT_SAMPLING_TIMESTEPS 0u
#define DEFAULT_MEASUREMENT_FREQ 1u
#define DEFAULT_TEMPORAL_AVG true

  template <class SimulatorType>
  class TimeSamplingDriver {
    public:
      Params params;
      uint32_t sampling_timesteps;
      uint32_t equilibration_timesteps;
      uint32_t measurement_freq;
      bool temporal_avg;

      TimeSamplingDriver(Params& params) : params(params) {
        equilibration_timesteps = utils::get<int>(params, "equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
        sampling_timesteps = utils::get<int>(params, "sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
        measurement_freq = utils::get<int>(params, "measurement_freq", DEFAULT_MEASUREMENT_FREQ);
        temporal_avg = (bool) utils::get<int>(params, "temporal_avg", DEFAULT_TEMPORAL_AVG);
      }

      DataSlide generate_dataslide(uint32_t num_threads) {
        auto start_time = std::chrono::high_resolution_clock::now();

        DataSlide slide;

        std::unique_ptr<SimulatorType> simulator = std::make_unique<SimulatorType>(params, num_threads);

        int num_timesteps, num_intervals;
        if (sampling_timesteps == 0) {
          num_timesteps = 0;
          num_intervals = 1;
        } else {
          num_timesteps = measurement_freq;
          num_intervals = sampling_timesteps/measurement_freq;
        }

        simulator->equilibration_timesteps(equilibration_timesteps);

        simulator->timesteps(num_timesteps);
        data_t sample = simulator->take_samples();
        for (auto const &[key, val] : sample) {
          slide.add_data(key);
          slide.push_data(key, val);
        }


        for (int t = 1; t < num_intervals; t++) {
          simulator->timesteps(num_timesteps);
          sample = simulator->take_samples();
          if (temporal_avg) {
            for (auto const &[key, val] : sample) {
              for (uint32_t i = 0; i < val.size(); i++) {
                slide.data[key][0][i] = slide.data[key][0][i].combine(val[i]);
              }
            }
          } else {
            for (auto const &[key, val] : sample) {
              slide.data[key].push_back(val);
            }
          }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        int duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double duration_s = duration_ms/1000.0;

        slide.add_data("time");
        slide.push_data("time", duration_s);

        simulator->cleanup();

        return slide;
      }
  };

}
