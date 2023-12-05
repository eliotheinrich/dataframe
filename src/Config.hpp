#pragma once

#include "utils.hpp"
#include "DataSlide.hpp"

namespace dataframe {

#define DEFAULT_NUM_RUNS 1
#define DEFAULT_SERIALIZE false


class Config {
	public:
		Params params;

		Config(Params &params) : params(params) {
			num_runs = utils::get<int>(params, "num_runs", DEFAULT_NUM_RUNS);
		}

		virtual ~Config() {}

		std::string to_string() const {
			return "{" + utils::params_to_string(params) + "}";
		}

		uint32_t get_nruns() const { 
			return num_runs;
		}

		virtual DataSlide compute(uint32_t num_threads) {
			return DataSlide();
		}
		
		virtual std::shared_ptr<Config> clone() {
			return std::make_shared<Config>(params);
		}

	private:
		uint32_t num_runs;
};

}