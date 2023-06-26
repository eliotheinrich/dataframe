#pragma once

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


#define DEFAULT_AUTOCONVERGE false
#define DEFAULT_CONVERGENCE_THRESHOLD 0.75

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
        
        virtual data_t take_samples() {
            return data_t();
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

        static float correlation_coefficient(const std::vector<double> &y) {
            uint n = y.size();

            double varx = std::sqrt(n*(n*n - 1.)/((n-1.)*12.));
            double my = 0.;
            for (uint i = 0; i < n; i++)
                my += y[i];
            my /= n;

            double vary = 0.;
            for (uint i = 0; i < n; i++)
                vary += std::pow(y[i] - my, 2);
            vary = std::sqrt(vary/(n - 1.));

            // For numerical stability
            if (vary < 1e-5) return 0;

            double sumxy = 0.;
            for (uint i = 0; i < n; i++)
                sumxy += i*y[i];
            
            float r = (sumxy - n*my*(n-1.)/2.)/((n-1.)*varx*vary);

            assert(std::abs(r) < 1.);
            return r;
        }

        bool samples_converged(const std::map<std::string, std::vector<Sample>> &samples) const {  
            for (auto const &[key, ksamples]: samples) {
                float r = correlation_coefficient(Sample::get_means(ksamples));
                if (r*r > convergence_threshold)
                    return false;
            }

                /*
                uint n = ksamples.size();
                double sx = n*(n-1.)/2.;
                double sxx = n*(n-1.)*(2.*n - 1.)/6.;
                double sy = 0.;
                double sxy;

                auto means = Sample::get_means(ksamples);
                for (uint i = 0; i < n; i++) {
                    sy += means[i];
                    sxy += i*means[i];
                }

                double slope = (n*sxy - sx*sy)/(n*sxx - sx*sx);
                double intercept = (sy*sxx - sx*sxy)/(n*sxx - sx*sx);

                std::vector<double> noise;
                for (uint i = 0; i < n; i++)
                    noise.push_back(means[i] - (slope*i + intercept));
                
                double m = 0.;
                double m2 = 0.;
                for (uint i = 0; i < n; i++) {
                    m += noise[i];
                    m2 += noise[i]*noise[i];
                }

                double s = std::sqrt(std::abs(m2 - m*m));

                float bounded = 0.;
                for (uint i = 0; i < n; i++) {
                    if (noise[i] > m + 2*s || noise[i] < m - 2*s)
                        bounded++;
                }

                if (bounded/n > convergence_threshold)
                    return false;
            }
            */

            return true;
        }


    public:
        uint sampling_timesteps;
        uint equilibration_timesteps;
        uint measurement_freq;
        bool temporal_avg;


        bool autoconverge;
        float convergence_threshold;

        void init_simulator(std::unique_ptr<Simulator> sim) {
            simulator = std::move(sim);
        }

        TimeConfig(Params &params) : Config(params) {
            nruns = params.get<int>("num_runs", DEFAULT_NUM_RUNS);
            equilibration_timesteps = params.get<int>("equilibration_timesteps", DEFAULT_EQUILIBRATION_STEPS);
            sampling_timesteps = params.get<int>("sampling_timesteps", DEFAULT_SAMPLING_TIMESTEPS);
            measurement_freq = params.get<int>("measurement_freq", DEFAULT_MEASUREMENT_FREQ);
            temporal_avg = (bool) params.get<int>("temporal_avg", DEFAULT_TEMPORAL_AVG);

            autoconverge = (bool) params.get<int>("autoconverge", DEFAULT_AUTOCONVERGE);
            if (autoconverge) {
                convergence_threshold = params.get<float>("convergence_threshold", DEFAULT_CONVERGENCE_THRESHOLD);
            }
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
                data_t sample = simulator->take_samples();
                for (auto const &[key, val] : sample) {
                    samples[key].push_back(val);
                }

                std::map<std::string, std::vector<Sample>> vector_sample = simulator->take_vector_samples();
                for (auto const &[key, vec] : vector_sample) {
                    slide.add_data(key);
                    for (auto const &v : vec) slide.push_data(key, v);
                }
            }

            int convergence_timesteps = sampling_timesteps;

            if (autoconverge) {
                while (!samples_converged(samples)) {
                    simulator->timesteps(num_timesteps);
                    data_t sample = simulator->take_samples();
                    for (auto const &[key, val] : sample) {
                        samples[key].push_back(val);
                        samples[key].erase(samples[key].begin());
                    }

                    convergence_timesteps += num_timesteps;
                }
                
                slide.add_data("convergence_timesteps");
                slide.push_data("convergence_timesteps", convergence_timesteps);
            }

            for (auto const &[key, ksamples] : samples) {
                slide.add_data(key);
                if (temporal_avg) {
                    slide.push_data(key, Sample::collapse(ksamples));
                } else {
                    for (auto s : ksamples) slide.push_data(key, s);
                }
            }

            slide.add_param("num_samples", (int) num_intervals);

            return slide;
        }

        virtual std::unique_ptr<Config> clone() override {
            std::unique_ptr<TimeConfig> config(new TimeConfig(params));
            std::unique_ptr<Simulator> sim = simulator.get()->clone(params);
            config->init_simulator(std::move(sim));
            return config;
        }
};