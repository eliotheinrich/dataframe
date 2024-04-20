#pragma once

#include "types.h"
#include "DataSlide.hpp"
#include <chrono>

namespace dataframe {

#define DEFAULT_EQUILIBRATION_STEPS 0u
#define DEFAULT_SAMPLING_TIMESTEPS 0u
#define DEFAULT_MEASUREMENT_FREQ 1u
#define DEFAULT_TEMPORAL_AVG true
#define DEFAULT_SAVE_SAMPLES false

  template <class SimulatorType>
  class TimeSamplingDriver {
    public:
      Params params;
      uint32_t sampling_timesteps;
      uint32_t equilibration_timesteps;
      uint32_t measurement_freq;
      bool temporal_avg;

      bool save_samples;

      TimeSamplingDriver(Params& params) : params(params) {
        equilibration_timesteps = utils::get<int>(params, "equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
        sampling_timesteps = utils::get<int>(params, "sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
        measurement_freq = utils::get<int>(params, "measurement_freq", DEFAULT_MEASUREMENT_FREQ);

        temporal_avg = (bool) utils::get<int>(params, "temporal_avg", DEFAULT_TEMPORAL_AVG);

        save_samples = (bool) utils::get<int>(params, "save_samples", DEFAULT_SAVE_SAMPLES);

        if (temporal_avg && save_samples) {
          throw std::invalid_argument("Cannot both perform temporal average and save all samples.");
        }
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
        // [categories x num_samples]
        for (auto const &[key, val] : sample) {
          if (save_samples) {
            slide.add_samples(key);
            for (auto const &v : val) {
              slide.push_data(key, v);
            }
          } else {
            slide.add_data(key);
            for (auto const &v : val) {
              slide.push_data(key, Sample(v));
            }
          }
        }

        for (int t = 1; t < num_intervals; t++) {
          simulator->timesteps(num_timesteps);
          sample = simulator->take_samples();
          if (temporal_avg) { // If temporal_avg = True, then individual samples will not be saved; combine with previous Sample
            for (auto const &[key, val] : sample) {
              for (uint32_t i = 0; i < val.size(); i++) {
                Sample s1 = slide.data[key][i][0];
                Sample s2 = Sample(val[i]);
                slide.data[key][i][0] = s1.combine(s2);
              }
            }
          } else { // Regardless of save_samples, push_data will get sample to the right place in the slide
            for (auto const &[key, val] : sample) {
              for (auto const &v : val) {
                slide.push_data(key, v);
              }
            }
          }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0;

        slide.add_data("time");
        slide.push_data("time", duration);

        simulator->cleanup();

        return slide;
      }
  };

}
