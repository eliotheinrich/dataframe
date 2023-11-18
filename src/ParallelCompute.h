#pragma once

#include "types.h"
#include "DataFrame.h"
#include "Config.hpp"

#include <optional>
#include <chrono>

class ParallelCompute {
	private:
		uint32_t percent_finished;
		uint32_t prev_percent_finished;

		std::vector<std::shared_ptr<Config>> configs;
		DataFrame serialize_df;

		void print_progress(
            uint32_t i, 
            uint32_t N, 
            std::optional<std::chrono::high_resolution_clock::time_point> run_start = std::nullopt
        );

		// Static so that can be passed to threadpool without memory sharing issues
		static compute_result_t thread_compute(std::shared_ptr<Config> config, uint32_t num_threads);

		std::vector<compute_result_t> compute_serial(
			std::vector<std::shared_ptr<Config>> total_configs,
			bool verbose
		);

		std::vector<compute_result_t> compute_bspl(
			std::vector<std::shared_ptr<Config>> total_configs, 
			bool verbose
		);

		std::vector<compute_result_t> compute_omp(
			std::vector<std::shared_ptr<Config>> total_configs, 
			bool verbose
		);

	public:
		DataFrame df;
		uint32_t num_threads;
		uint32_t num_threads_per_task;

		double atol;
		double rtol;

		bool average_congruent_runs;
		bool serialize;

		parallelization_t parallelization_type;

		bool record_error;


		ParallelCompute(Params& metaparams, std::vector<std::shared_ptr<Config>> configs);

		void compute(bool verbose=false);

		void write_json(std::string filename) const {
			df.write_json(filename, record_error);
		}

		void write_serialize_json(std::string filename) const {
			if (serialize)
				serialize_df.write_json(filename, record_error);
		}
};