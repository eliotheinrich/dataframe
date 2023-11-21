#pragma once

#include "DataFrame.hpp"
#include "Config.hpp"

#include <BS_thread_pool.hpp>
#include <omp.h>

#include <optional>
#include <chrono>

namespace dataframe {

class ParallelCompute {
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


		ParallelCompute(Params& metaparams, std::vector<std::shared_ptr<Config>> configs) : configs(configs) {
			num_threads = utils::get<int>(metaparams, "num_threads", 1);
			num_threads_per_task = utils::get<int>(metaparams, "num_threads_per_task", 1);

			serialize = utils::get<int>(metaparams, "serialize", false);

			atol = utils::get<double>(metaparams, "atol", ATOL);
			rtol = utils::get<double>(metaparams, "rtol", RTOL);

			average_congruent_runs = utils::get<int>(metaparams, "average_congruent_runs", true);

			parallelization_type = (parallelization_t) utils::get<int>(metaparams, "parallelization_type", parallelization_t::threadpool);

			record_error = utils::get<int>(metaparams, "record_error", false);

			df.add_metadata(metaparams);
			df.atol = atol;
			df.rtol = rtol;

			serialize_df.add_metadata(metaparams);
			serialize_df.atol = atol;
			serialize_df.rtol = rtol;
		}

		void compute(bool verbose=false){
			auto start = std::chrono::high_resolution_clock::now();

			uint32_t num_configs = configs.size();

			std::vector<std::shared_ptr<Config>> total_configs;
			for (uint32_t i = 0; i < num_configs; i++) {
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				for (uint32_t j = 0; j < nruns; j++)
					total_configs.push_back(configs[i]->clone());
			}

			uint32_t num_jobs = total_configs.size();



			std::vector<compute_result_t> results;

			switch (parallelization_type) {
			case parallelization_t::threadpool:
				results = compute_bspl(total_configs, verbose);
				break;
			case parallelization_t::openmp:
				results = compute_omp(total_configs, verbose);
				break;
			case parallelization_t::serial:
				results = compute_serial(total_configs, verbose);
				break;
			}

			if (verbose)
				std::cout << "\n";
			
			utils::var_t_eq equality_comparator(atol, rtol);
			uint32_t idx = 0;
			for (uint32_t i = 0; i < num_configs; i++) {
				auto [slide, serialization] = results[idx];
				uint32_t nruns = configs[i]->get_nruns();

				std::vector<std::optional<std::string>> slide_serializations;
				slide_serializations.push_back(serialization);

				for (uint32_t j = 1; j < nruns; j++) {
					idx++;
					auto [slide_tmp, serialization] = results[idx];

					if (average_congruent_runs) {
						slide = slide.combine(slide_tmp, equality_comparator);
					} else {
						df.add_slide(slide_tmp);
					}

					slide_serializations.push_back(serialization);
				}
				idx++;

				df.add_slide(slide);	

				// Add serializations
				DataSlide serialize_ds = DataSlide::copy_params(slide);
				for (uint32_t j = 0; j < nruns; j++) {
					if (slide_serializations[j].has_value())
						serialize_ds.add_param("serialization_" + std::to_string(j), slide_serializations[j].value());
				}
				serialize_df.add_slide(serialize_ds);
			}

			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

			df.add_metadata("num_threads", (int) num_threads);
			df.add_metadata("num_jobs", (int) num_jobs);
			df.add_metadata("total_time", (int) duration.count());

			serialize_df.add_metadata("num_threads", (int) num_threads);
			serialize_df.add_metadata("num_jobs", (int) num_jobs);
			serialize_df.add_metadata("total_time", (int) duration.count());
			// A little hacky; need to set num_runs = 1 so that configs are not duplicated when a run is
			// started from serialized data
			for (auto &slide : serialize_df.slides)
				slide.add_param("num_runs", 1);

			df.promote_params();
			if (average_congruent_runs)
				df.reduce();

			if (verbose)
				std::cout << "Total runtime: " << (int) duration.count() << std::endl;
		}

		void write_json(const std::string& filename) const {
			df.write_json(filename, record_error);
		}

		void write_serialize_json(const std::string& filename) const {
			if (serialize)
				serialize_df.write_json(filename, record_error);
		}

	private:
		uint32_t percent_finished;
		uint32_t prev_percent_finished;

		std::vector<std::shared_ptr<Config>> configs;
		DataFrame serialize_df;

		void print_progress(
            uint32_t i, 
            uint32_t N, 
            std::optional<std::chrono::high_resolution_clock::time_point> run_start = std::nullopt
        ) {
			percent_finished = std::round(float(i)/N * 100);
			if (percent_finished != prev_percent_finished) {
				prev_percent_finished = percent_finished;
				int duration = -1;
				if (run_start.has_value()) {
					auto now = std::chrono::high_resolution_clock::now();
					duration = std::chrono::duration_cast<std::chrono::seconds>(now - run_start.value()).count();
				}
				float seconds_per_job = duration/float(i);
				int remaining_time = seconds_per_job * (N - i);

				float progress = percent_finished/100.;

				int bar_width = 70;
				std::cout << "[";
				int pos = bar_width * progress;
				for (int i = 0; i < bar_width; ++i) {
					if (i < pos) std::cout << "=";
					else if (i == pos) std::cout << ">";
					else std::cout << " ";
				}
				std::stringstream time;
				if (duration == -1) time << "";
				else {
					time << " [ ETA: ";
					uint32_t num_seconds = remaining_time % 60;
					uint32_t num_minutes = remaining_time/60;
					uint32_t num_hours = num_minutes/60;
					num_minutes -= num_hours*60;
					time << std::setfill('0') << std::setw(2) << num_hours << ":" 
							<< std::setfill('0') << std::setw(2) << num_minutes << ":" 
							<< std::setfill('0') << std::setw(2) << num_seconds << " ] ";
				}
				std::cout << "] " << int(progress * 100.0) << " %" << time.str()  << "\r";
				std::cout.flush();
			}
		}

		// Static so that can be passed to threadpool without memory sharing issues
		static compute_result_t thread_compute(std::shared_ptr<Config> config, uint32_t num_threads) {
			DataSlide slide = config->compute(num_threads);

			std::optional<std::string> serialize_result = std::nullopt;
			if (config->serialize)
				serialize_result = config->write_serialize();

			slide.add_param(config->params);

			return std::make_pair(slide, serialize_result);
		}

		std::vector<compute_result_t> compute_serial(
			std::vector<std::shared_ptr<Config>> total_configs,
			bool verbose
		) {
			uint32_t total_runs = total_configs.size();
			uint32_t num_configs = configs.size();

			if (verbose) {
				std::cout << "Computing in serial.\n";
				std::cout << "num_configs: " << num_configs << std::endl;
				std::cout << "total_runs: " << total_runs << std::endl;
				print_progress(0, total_runs);	
			}

			std::vector<compute_result_t> results(total_runs);

			auto run_start = std::chrono::high_resolution_clock::now();
			for (uint32_t i = 0; i < total_runs; i++) {
				results[i] = ParallelCompute::thread_compute(total_configs[i], num_threads_per_task);

				if (verbose)
					print_progress(i, total_runs, run_start);
			}

			return results;
		}

		std::vector<compute_result_t> compute_bspl(
			std::vector<std::shared_ptr<Config>> total_configs, 
			bool verbose
		) {
			uint32_t total_runs = total_configs.size();
			uint32_t num_configs = configs.size();

			if (verbose) {
				std::cout << "Computing with BSPL. " << num_threads << " threads available.\n";
				std::cout << "num_configs: " << num_configs << std::endl;
				std::cout << "total_runs: " << total_runs << std::endl;
				print_progress(0, total_runs);
			}

			std::vector<DataSlide> slides(total_runs);
			BS::thread_pool threads(num_threads/num_threads_per_task);
			std::vector<std::future<compute_result_t>> futures(total_runs);
			std::vector<compute_result_t> results(total_runs);


			auto run_start = std::chrono::high_resolution_clock::now();
			for (uint32_t i = 0; i < total_runs; i++)
				futures[i] = threads.submit(ParallelCompute::thread_compute, total_configs[i], num_threads_per_task);

			for (uint32_t i = 0; i < total_runs; i++) {
				results[i] = futures[i].get();
				
				if (verbose)
					print_progress(i, total_runs, run_start);
			}

			return results;
		}

		std::vector<compute_result_t> compute_omp(
			std::vector<std::shared_ptr<Config>> total_configs, 
			bool verbose
		) {
			uint32_t total_runs = total_configs.size();
			uint32_t num_configs = configs.size();

			if (verbose) {
				std::cout << "Computing with OpenMP. " << num_threads << " threads available.\n";
				std::cout << "num_configs: " << num_configs << std::endl;
				std::cout << "total_runs: " << total_runs << std::endl;
				print_progress(0, total_runs);
			}


			std::vector<DataSlide> slides(total_runs);
			std::vector<compute_result_t> results(total_runs);

			auto run_start = std::chrono::high_resolution_clock::now();
			uint32_t completed = 0;

			#pragma omp parallel for num_threads(num_threads)
			for (uint32_t i = 0; i < total_runs; i++) {
				results[i] = ParallelCompute::thread_compute(total_configs[i], num_threads_per_task);
				completed++;

				if (verbose)
					print_progress(completed, total_runs, run_start);
			}

			return results;
		}
};

}