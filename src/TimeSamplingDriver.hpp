#pragma once

#include "types.h"
#include "DataSlide.hpp"
#include "Simulator.hpp"
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

      ~TimeSamplingDriver()=default;

      void init_simulator(size_t num_threads, const std::optional<std::vector<byte_t>>& data = std::nullopt) {
        simulator = std::make_unique<SimulatorType>(params, num_threads);
        if (data.has_value()) {
          simulator->deserialize(data.value());
        }
      }

      DataSlide generate_dataslide(bool serialize) {
        auto start_time = std::chrono::high_resolution_clock::now();

        DataSlide slide;

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

        if (save_samples) {
          slide.add_samples(sample);
          slide.push_samples(sample);
        } else {
          slide.add_data(sample);
          slide.push_samples_to_data(sample);
        }

        for (int t = 1; t < num_intervals; t++) {
          simulator->timesteps(num_timesteps);
          sample = simulator->take_samples();

          if (save_samples) {
            slide.push_samples(sample);
          } else {
            slide.push_samples_to_data(sample, temporal_avg);
          }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()/1000.0;

        slide.add_data("time");
        slide.push_samples_to_data("time", duration);

        if (serialize) {
          slide.buffer = simulator->serialize();
        }

        simulator->cleanup();

        return slide;
      }

    private:
      std::unique_ptr<SimulatorType> simulator;
  };

}
