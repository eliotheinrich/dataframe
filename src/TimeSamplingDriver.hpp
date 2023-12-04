#pragma once

#include "Simulator.hpp"
#include "Config.hpp"

#define EXPORT_SIMULATOR_DRIVER(A)                                  \
nanobind::class_<TimeSamplingDriver<A>>(m, #A)                      \
    .def(nanobind::init<Params>())                                 \
    .def("generate_slide", &TimeSamplingDriver<A>::generate_slide);

#define EXPORT_CONFIG(A)                                                \
nanobind::class_<A>(m, #A)                                              \
    .def(nanobind::init<Params&>())                                     \
    .def("compute", &A::compute)                                        \
    .def("clone", &A::clone)                                            \
    .def("__getstate__", [](const A& config) { return config.params; }) \
    .def("__setstate__", [](A& config, Params& params){ new (&config) A(params); } )

namespace dataframe {

#define DEFAULT_EQUILIBRATION_STEPS 0u
#define DEFAULT_SAMPLING_TIMESTEPS 0u
#define DEFAULT_MEASUREMENT_FREQ 1u
#define DEFAULT_TEMPORAL_AVG true

template <class SimulatorType>
class TimeSamplingDriver {
    public:
        Params& params;
        uint32_t sampling_timesteps;
        uint32_t equilibration_timesteps;
        uint32_t measurement_freq;
        bool temporal_avg;

        TimeSamplingDriver<SimulatorType>(Params& params) : params(params) {
            equilibration_timesteps = utils::get<int>(params, "equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
            sampling_timesteps = utils::get<int>(params, "sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
            measurement_freq = utils::get<int>(params, "measurement_freq", DEFAULT_MEASUREMENT_FREQ);
            temporal_avg = (bool) utils::get<int>(params, "temporal_avg", DEFAULT_TEMPORAL_AVG);
        }

        DataSlide generate_slide(uint32_t num_threads) {
            auto start_time = std::chrono::high_resolution_clock::now();

            DataSlide slide;

            std::unique_ptr<SimulatorType> simulator = std::make_unique<SimulatorType>(params, num_threads);

            simulator->equilibration_timesteps(equilibration_timesteps);

            int num_timesteps, num_intervals;
            if (sampling_timesteps == 0) {
                num_timesteps = 0;
                num_intervals = 1;
            } else {
                num_timesteps = measurement_freq;
                num_intervals = sampling_timesteps/measurement_freq;
            }

            std::map<std::string, std::vector<std::vector<Sample>>> samples;

            for (int t = 0; t < num_intervals; t++) {
                simulator->timesteps(num_timesteps);
                data_t sample = simulator->take_samples();
                for (auto const &[key, val] : sample) {
                    samples[key].push_back(val);
                }
            }

            for (auto const &[key, ksamples] : samples) {
                slide.add_data(key);
                if (temporal_avg) {
                    slide.push_data(key, Sample::collapse_samples(ksamples));
                } else {
                    for (auto s : ksamples) {
                        slide.push_data(key, s);
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