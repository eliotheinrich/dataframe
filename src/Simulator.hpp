#pragma once

#include "DataSlide.h"
#include "types.h"
#include "Sample.h"

#include <memory>
#include <random>

#define CLONE(A, B) virtual std::shared_ptr<A> clone(Params &params) override { return std::shared_ptr<A>(new B(params)); }

#define DEFAULT_RANDOM_SEED -1

class Simulator {
    private:
        int seed;
        
    protected:
        std::minstd_rand rng;

    public:
        int rand() { return rng(); }
        float randf() { return float(rng())/float(RAND_MAX); }

        Simulator(Params &params) {
            seed = dataframe_utils::get<int>(params, "random_seed", DEFAULT_RANDOM_SEED);
            if (seed == -1) {
                thread_local std::random_device rd;
                rng.seed(rd());
            } else {
                rng.seed(seed);
            }
        }

        virtual ~Simulator() {}

        virtual std::string serialize() const {
            return "serialize is not implemented for this simulator.\n";
        }

        virtual void timesteps(uint32_t num_steps)=0;

        // By default, do nothing special during equilibration timesteps
        // May want to include, i.e., annealing 
        virtual void equilibration_timesteps(uint32_t num_steps) {
            timesteps(num_steps);
        }
        
        virtual data_t take_samples() {
            return data_t();
        }
        virtual std::map<std::string, std::vector<Sample>> take_vector_samples() {
            return std::map<std::string, std::vector<Sample>>();
        }

        virtual void init_state(uint32_t num_threads)=0;
        virtual void cleanup() {}
        
        virtual std::shared_ptr<Simulator> clone(Params &params)=0;
        virtual std::shared_ptr<Simulator> deserialize(Params &params, const std::string&) { return clone(params); }
};
