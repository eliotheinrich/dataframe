#pragma once

#include "Simulator.hpp"
#include "Config.hpp"

namespace dataframe {

#define DEFAULT_EQUILIBRATION_STEPS 0u
#define DEFAULT_SAMPLING_TIMESTEPS 0u
#define DEFAULT_MEASUREMENT_FREQ 1u
#define DEFAULT_TEMPORAL_AVG true


class TimeConfig : public Config {
    public:
        uint32_t sampling_timesteps;
        uint32_t equilibration_timesteps;
        uint32_t measurement_freq;
        bool temporal_avg;

        void init_simulator(std::shared_ptr<Simulator> sim) {
            simulator = std::move(sim);
        }

        TimeConfig(Params &params) : Config(params) {
            equilibration_timesteps = utils::get<int>(params, "equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
            sampling_timesteps = utils::get<int>(params, "sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
            measurement_freq = utils::get<int>(params, "measurement_freq", DEFAULT_MEASUREMENT_FREQ);
            temporal_avg = (bool) utils::get<int>(params, "temporal_avg", DEFAULT_TEMPORAL_AVG);
        }

        virtual DataSlide compute(uint32_t num_threads) override {
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

        virtual std::shared_ptr<Config> clone() override {
            std::shared_ptr<TimeConfig> config(new TimeConfig(params));
            std::shared_ptr<Simulator> sim = simulator->clone(params);
            config->init_simulator(sim);
            return config;
        }

        virtual std::string write_serialize() const override {
            return simulator->serialize();
        }

        virtual std::shared_ptr<Config> deserialize(Params &params, const std::string &data) override {
            std::shared_ptr<TimeConfig> config(new TimeConfig(params));
            std::shared_ptr<Simulator> sim = simulator->deserialize(params, data);
            config->init_simulator(sim);
            return config;
        }

    private:
        std::shared_ptr<Simulator> simulator;

};

// Prepares a TimeConfig with a templated Simulator type.
template <class SimulatorType>
std::shared_ptr<Config> prepare_timeconfig(Params &params) {
    std::shared_ptr<TimeConfig> config(new TimeConfig(params));
    std::shared_ptr<Simulator> sim(new SimulatorType(params));

    config->init_simulator(std::move(sim));
    return config;
}

}