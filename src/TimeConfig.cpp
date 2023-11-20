#include "TimeConfig.h"
#include "utils.h"

#define DEFAULT_EQUILIBRATION_STEPS 0u
#define DEFAULT_SAMPLING_TIMESTEPS 0u
#define DEFAULT_MEASUREMENT_FREQ 1u
#define DEFAULT_TEMPORAL_AVG true

using namespace dataframe_utils;

TimeConfig::TimeConfig(Params &params) : Config(params) {
    equilibration_timesteps = get<int>(params, "equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
    sampling_timesteps = get<int>(params, "sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
    measurement_freq = get<int>(params, "measurement_freq", DEFAULT_MEASUREMENT_FREQ);
    temporal_avg = (bool) get<int>(params, "temporal_avg", DEFAULT_TEMPORAL_AVG);
}

DataSlide TimeConfig::compute(uint32_t num_threads) {
    auto start_time = std::chrono::high_resolution_clock::now();

    DataSlide slide;

    simulator->init_state(num_threads);

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
            slide.push_data(key, collapse_samples(ksamples));
        } else {
            for (auto s : ksamples) 
                slide.push_data(key, s);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    slide.add_data("time");
    slide.push_data("time", duration);

    simulator->cleanup();

    return slide;
}

std::shared_ptr<Config> TimeConfig::clone() {
    std::shared_ptr<TimeConfig> config(new TimeConfig(params));
    std::shared_ptr<Simulator> sim = simulator->clone(params);
    config->init_simulator(sim);
    return config;
}

std::shared_ptr<Config> TimeConfig::deserialize(Params &params, const std::string &data) {
    std::shared_ptr<TimeConfig> config(new TimeConfig(params));
    std::shared_ptr<Simulator> sim = simulator->deserialize(params, data);
    config->init_simulator(sim);
    return config;
}
