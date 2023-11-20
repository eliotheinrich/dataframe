#pragma once

#include "Simulator.hpp"
#include "Config.hpp"

class TimeConfig : public Config {
    private:
        std::shared_ptr<Simulator> simulator;

    public:
        uint32_t sampling_timesteps;
        uint32_t equilibration_timesteps;
        uint32_t measurement_freq;
        bool temporal_avg;

        void init_simulator(std::shared_ptr<Simulator> sim) {
            simulator = std::move(sim);
        }

        TimeConfig(Params &params);

        virtual DataSlide compute(uint32_t num_threads) override;

        virtual std::shared_ptr<Config> clone() override;

        virtual std::string write_serialize() const override {
            return simulator->serialize();
        }

        virtual std::shared_ptr<Config> deserialize(Params &params, const std::string &data) override;
};

// Prepares a TimeConfig with a templated Simulator type.
template <class SimulatorType>
std::shared_ptr<Config> prepare_timeconfig(Params &params) {
    std::shared_ptr<TimeConfig> config(new TimeConfig(params));
    std::shared_ptr<Simulator> sim(new SimulatorType(params));

    config->init_simulator(std::move(sim));
    return config;
}