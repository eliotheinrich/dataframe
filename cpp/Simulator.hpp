#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <iostream>
#include <DataFrame.hpp>
#include <algorithm>
#include <numeric>
#include <math.h>

#define DEFAULT_NUM_RUNS 1u
#define DEFAULT_EQUILIBRATION_STEPS 0u
#define DEFAULT_SAMPLING_TIMESTEPS 0u
#define DEFAULT_MEASUREMENT_FREQ 1u
#define DEFAULT_SPACING 1u
#define DEFAULT_TEMPORAL_AVG true

class Simulator {
    public:
        virtual Simulator* clone(Params &params)=0;
        virtual void timesteps(uint num_steps)=0;
        virtual void equilibration_timesteps(uint num_steps) {
            timesteps(num_steps);
        }
        virtual std::map<std::string, Sample> take_samples()=0;
        virtual void init_state()=0;
};

class TimeConfig : public Config {
    private:
        Simulator *simulator;
        uint nruns;

    public:
        uint sampling_timesteps;
        uint equilibration_timesteps;
        uint measurement_freq;
        bool temporal_avg;

        void init_simulator(Simulator *sim) {
            simulator = sim;
        }

        TimeConfig(Params &params) : Config(params) {
            nruns = params.geti("num_runs", DEFAULT_NUM_RUNS);
            equilibration_timesteps = params.geti("equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
            sampling_timesteps = params.geti("sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
            measurement_freq = params.geti("measurement_freq", DEFAULT_MEASUREMENT_FREQ);
            temporal_avg = (bool) params.geti("temporal_avg", DEFAULT_TEMPORAL_AVG);
        }

        virtual uint get_nruns() const { return nruns; }

        void compute(DataSlide *slide) {
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
            }

            for (auto const &[key, ksamples] : samples) {
                slide->add_data(key);
                if (temporal_avg) {
                    slide->push_data(key, Sample::collapse(ksamples));
                } else {
                    for (auto s : ksamples) {
                        slide->push_data(key, s);
                    }
                }
            }
        }

        virtual Config* clone() {
            TimeConfig* config = new TimeConfig(params);
            config->simulator = simulator->clone(params);
            return config;
        }
};

#endif