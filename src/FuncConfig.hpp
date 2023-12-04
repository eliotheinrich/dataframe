#pragma once

#include "types.h"
#include "Config.hpp"

//#include <nanobind/stl/function.h>

namespace dataframe {

class FuncConfig : public Config {
	public:
		std::function<DataSlide(Params&, uint32_t)> callable;

		FuncConfig(Params& params, std::function<DataSlide(Params&, uint32_t)> callable) : Config(params), callable(callable) {}

		virtual DataSlide compute(uint32_t num_threads) override {
			return callable(params, num_threads);
		}

		virtual std::shared_ptr<Config> clone() const override {
			Params params_copy = params;
			return std::make_shared<FuncConfig>(params_copy, callable);
		}
};

}