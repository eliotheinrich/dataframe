#pragma once

namespace dataframe {

#define DEFAULT_NUM_RUNS 1
#define DEFAULT_SERIALIZE false


class Config {
	public:
		bool serialize;
		Params params;

		Config(Params &params) : params(params) {
			num_runs = utils::get<int>(params, "num_runs", DEFAULT_NUM_RUNS);
			serialize = utils::get<int>(params, "serialize", DEFAULT_SERIALIZE);
		}

		Config(Config &c) : Config(c.params) {}

		virtual ~Config() {}

		std::string to_string() const {
			return "{" + utils::params_to_string(params) + "}";
		}

		uint32_t get_nruns() const { 
			return num_runs;
		}

		// To implement
		virtual std::string write_serialize() const {
			return "No implementation for config.serialize() provided.\n";
		}

		virtual DataSlide compute(uint32_t num_threads)=0;
		virtual std::shared_ptr<Config> clone()=0;
		virtual std::shared_ptr<Config> deserialize(Params&, const std::string&) { 
			return clone(); // By default, just clone config
		}

	private:
		uint32_t num_runs;
};

}