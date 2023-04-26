#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <iostream>
#include <DataFrame.hpp>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <memory>
#include <random>

#define DEFAULT_NUM_RUNS 1u
#define DEFAULT_EQUILIBRATION_STEPS 0u
#define DEFAULT_SAMPLING_TIMESTEPS 0u
#define DEFAULT_MEASUREMENT_FREQ 1u
#define DEFAULT_SPACING 1u
#define DEFAULT_TEMPORAL_AVG true
#define DEFAULT_RANDOM_SEED -1

#define CLONE(A, B) virtual std::unique_ptr<A> clone(Params &params) override { return std::unique_ptr<A>(new B(params)); }

class Simulator {
    private:
        std::minstd_rand rng;
        int seed;

    public:
        int rand() { return rng(); }
        float randf() { return float(rng())/float(RAND_MAX); }

        Simulator(Params &params) {
            seed = params.get<int>("random_seed", DEFAULT_RANDOM_SEED);
            if (seed == -1) rng = std::minstd_rand(std::rand());
            else rng = std::minstd_rand(seed);
        }

        virtual ~Simulator() {}

        virtual std::unique_ptr<Simulator> clone(Params &params)=0;
        virtual void timesteps(uint num_steps)=0;

        // By default, do nothing special during equilibration timesteps
        // May want to include, i.e., annealing 
        virtual void equilibration_timesteps(uint num_steps) {
            timesteps(num_steps);
        }
        
        virtual std::map<std::string, Sample> take_samples() {
            return std::map<std::string, Sample>();
        }
        virtual std::map<std::string, std::vector<Sample>> take_vector_samples() {
            return std::map<std::string, std::vector<Sample>>();
        }

        virtual void init_state()=0;
};

class TimeConfig : public Config {
    private:
        std::unique_ptr<Simulator> simulator;
        uint nruns;

    public:
        uint sampling_timesteps;
        uint equilibration_timesteps;
        uint measurement_freq;
        bool temporal_avg;

        void init_simulator(std::unique_ptr<Simulator> sim) {
            simulator = std::move(sim);
        }

        TimeConfig(Params &params) : Config(params) {
            nruns = params.get<int>("num_runs", DEFAULT_NUM_RUNS);
            equilibration_timesteps = params.get<int>("equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
            sampling_timesteps = params.get<int>("sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
            measurement_freq = params.get<int>("measurement_freq", DEFAULT_MEASUREMENT_FREQ);
            temporal_avg = (bool) params.get<int>("temporal_avg", DEFAULT_TEMPORAL_AVG);
        }

        virtual uint get_nruns() const override { return nruns; }

        virtual DataSlide compute() override {
            DataSlide slide;

            simulator->init_state();

            simulator->equilibration_timesteps(equilibration_timesteps);

            int num_timesteps, num_intervals;
            if (sampling_timesteps == 0) {
                num_timesteps = 0;
                num_intervals = 1;
            } else {
                num_timesteps = measurement_freq;
                num_intervals = sampling_timesteps/measurement_freq;
            }

            std::map<std::string, std::vector<Sample>> samples;

            for (int t = 0; t < num_intervals; t++) {
                simulator->timesteps(num_timesteps);
                std::map<std::string, Sample> sample = simulator->take_samples();
                for (auto const &[key, val] : sample) {
                    samples[key].push_back(val);
                }

                std::map<std::string, std::vector<Sample>> vector_sample = simulator->take_vector_samples();
                for (auto const &[key, vec] : vector_sample) {
                    slide.add_data(key);
                    for (auto const v : vec) slide.push_data(key, v);
                }
            }

            for (auto const &[key, ksamples] : samples) {
                slide.add_data(key);
                if (temporal_avg) {
                    slide.push_data(key, Sample::collapse(ksamples));
                } else {
                    for (auto s : ksamples) slide.push_data(key, s);
                }
            }

            slide.add("num_samples", (int) num_intervals);

            return slide;
        }

        virtual std::unique_ptr<Config> clone() override {
            std::unique_ptr<TimeConfig> config(new TimeConfig(params));
            std::unique_ptr<Simulator> sim = simulator.get()->clone(params);
            config->init_simulator(std::move(sim));
            return config;
        }
};

#endif