#pragma once

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <variant>
#include <nlohmann/json.hpp>

#ifdef DEBUG
#define LOG(x) std::cout << x
#else
#define LOG(x)
#endif

#ifdef OMPI // OMPI definitions and requirements

#include <mpi.h>

#define MASTER 0
#define TERMINATE 0
#define CONTINUE 1

#define DO_IF_MASTER(x) {								\
	int __rank;											\
	MPI_Comm_rank(MPI_COMM_WORLD, &__rank);				\
	if (__rank == MASTER) {								\
		x												\
	}													\
}

#else

#define DO_IF_MASTER(x) x

#ifndef SERIAL
#include <BS_thread_pool.hpp>
#endif

#endif


class Sample;
class DataSlide;
class DataFrame;
class Config;
class Params;
class ParallelCompute;

static std::string join(const std::vector<std::string> &v, const std::string &delim);

typedef std::variant<int, float, std::string> var_t;
typedef std::map<std::string, Sample> data_t;

struct var_to_string {
	std::string operator()(const int& i) const { return std::to_string(i); }
	std::string operator()(const float& f) const { return std::to_string(f); }
	std::string operator()(const std::string& s) const { return "\"" + s + "\""; }
};

struct get_var {
	int operator()(const int& i) const { return i; }
	float operator()(const float& f) const { return f; }
	std::string operator()(const std::string& s) const { return s; }
};

#define DF_EPS 0.00001

static bool operator==(const var_t& v, const var_t& t) {
	if (v.index() != t.index()) return false;

	if (v.index() == 0) return std::get<int>(v) == std::get<int>(t);
	else if (v.index() == 1) return std::abs(std::get<float>(v) - std::get<float>(t)) < DF_EPS;
	else return std::get<std::string>(v) == std::get<std::string>(t);
}

static bool operator!=(const var_t& v, const var_t& t) {
	return !(v == t);
}

class Params {		
	public:
		std::map<std::string, var_t> fields;

		Params() {}
		~Params() {}

		Params(Params *p) {
			for (auto &[key, val] : p->fields) fields.emplace(key, val); // TODO CHECK COPY SEMANTICS
		}

		std::string to_string(uint indentation=0) const {
			std::string s = "";
			for (uint i = 0; i < indentation; i++) s += "\t";
			std::vector<std::string> buffer;
			

			for (auto const &[key, field] : fields) {
				buffer.push_back("\"" + key + "\": " + std::visit(var_to_string(), field));
			}

			std::string delim = ",\n";
			for (uint i = 0; i < indentation; i++) delim += "\t";
			s += join(buffer, delim);

			return s;
		}

		template <typename T>
		T get(std::string s) const {
			if (!contains(s)) {
				std::cout << "Key \"" + s + "\" not found.\n"; 
                assert(false);
			}

			return std::get<T>(fields.at(s));
		}

		template <typename T>
		T get(std::string s, T defaultv) {
			if (!contains(s)) {
				add(s, defaultv);
				return defaultv;
			}

			return std::get<T>(fields.at(s));
		}

		template <typename T>
		void add(std::string s, T const& val) { fields[s] = var_t{val}; }

		bool contains(std::string s) const { return fields.count(s); }
		bool remove(std::string s) {
			if (fields.count(s)) {
				fields.erase(s);
				return true;
			}
			return false;
		}

		bool operator==(const Params &p) const {
			for (auto const &[key, field] : fields) {
				if (!p.contains(key)) return false;
				if (p.fields.at(key) != field) return false;
			}
			for (auto const &[key, field] : p.fields) {
				if (!contains(key)) return false;
			}

			return true;
		}

		bool operator!=(const Params &p) const {
			return !((*this) == p);
		}

		template <typename json_object>
		static var_t parse_json_type(json_object p) {
			if ((p.type() == nlohmann::json::value_t::number_integer) || 
				(p.type() == nlohmann::json::value_t::number_unsigned) ||
				(p.type() == nlohmann::json::value_t::boolean)) {
				return var_t{(int) p};
			}  else if (p.type() == nlohmann::json::value_t::number_float) {
				return var_t{(float) p};
			} else if (p.type() == nlohmann::json::value_t::string) {
				return var_t{std::string(p)};
			} else {
				std::cout << "Invalid json item type on " << p << "; aborting.\n";
				assert(false);

				return var_t{0};
			}
		}

		static std::vector<Params> load_json(nlohmann::json data, Params p, bool debug) {
			if (debug) {
				DO_IF_MASTER(std::cout << "Loaded: \n" << data.dump() << "\n";)
			}

			std::vector<Params> params;

			// Dealing with model parameters
			std::vector<std::map<std::string, var_t>> zparams;
			if (data.contains("zparams")) {
				for (uint i = 0; i < data["zparams"].size(); i++) {
					zparams.push_back(std::map<std::string, var_t>());
					for (auto const &[key, val] : data["zparams"][i].items()) {
						if (data.contains(key)) {
							std::cout << "Key " << key << " passed as a zipped parameter and an unzipped parameter; aborting.\n";
							assert(false);
						}
						zparams[i][key] = parse_json_type(val);
					}
				}

				data.erase("zparams");
			}

			if (zparams.size() > 0) {
				for (uint i = 0; i < zparams.size(); i++) {
					for (auto const &[k, v] : zparams[i]) p.add(k, v);
					std::vector<Params> new_params = load_json(data, Params(&p), false);
					params.insert(params.end(), new_params.begin(), new_params.end());
				}

				return params;
			}

			// Dealing with config parameters
			std::vector<std::string> scalars;
			std::string vector_key; // Only need one for next recursive call
			bool contains_vector = false;
			for (auto const &[key, val] : data.items()) {
				if (val.type() == nlohmann::json::value_t::array) {
					vector_key = key;
					contains_vector = true;
				} else {
					p.add(key, parse_json_type(val));
					scalars.push_back(key);
				}
			}

			for (auto key : scalars) data.erase(key);

			if (!contains_vector) {
				params.push_back(p);
			} else {
				auto vals = data[vector_key];
				data.erase(vector_key);
				for (auto v : vals) {
					p.add(vector_key, parse_json_type(v));

					std::vector<Params> new_params = load_json(data, &p, false);
					params.insert(params.end(), new_params.begin(), new_params.end());
				}
			}

			return params;
		}

		static std::vector<Params> load_json(nlohmann::json data, bool debug=false) {
			return load_json(data, Params(), debug);
		}

};

class Sample {
    private:
        double mean;
        double std;
        uint num_samples;

	public:
		Sample() : mean(0.), std(0.), num_samples(0) {}
        Sample(double mean) : mean(mean), std(0.), num_samples(1) {}
		Sample(double mean, double std, uint num_samples) : mean(mean), std(std), num_samples(num_samples) {}

		template<class T>
		Sample(const std::vector<T> &v) {
			num_samples = v.size();
			mean = std::accumulate(v.begin(), v.end(), 0.0);
			float sum = 0.0;
			for (auto const t : v) {
				sum += std::pow(t - mean, 2.0);
			}

			std = std::sqrt(sum/(num_samples - 1.));
		}

		Sample(const std::string &s) {
			if (s.front() == '[' && s.back() == ']') {
				std::string trimmed = s.substr(1, s.length() - 2);
				std::vector<uint> pos;
				for (uint i = 0; i < trimmed.length(); i++) {
					if (trimmed[i] == ',')
						pos.push_back(i);
				}

				assert(pos.size() == 2);

				mean = std::stof(trimmed.substr(0, pos[0]));
				std = std::stof(trimmed.substr(pos[0]+1, pos[1]));
				num_samples = std::stoi(trimmed.substr(pos[1]+1, trimmed.length()-1));
			} else {
				mean = std::stof(s);
				std = 0.;
				num_samples = 1;
			}
		}

        double get_mean() const {
			return this->mean;
		}
		void set_mean(double mean) {
			this->mean = mean;
		}

        double get_std() const {
			return this->std;
		}
		void set_std(double std) {
			this->std = std;
		}

        uint get_num_samples() const {
			return this->num_samples;
		}
		void set_num_samples(uint num_samples) {
			this->num_samples = num_samples;
		}

        Sample combine(const Sample &other) const {
			uint combined_samples = this->num_samples + other.get_num_samples();
			if (combined_samples == 0) return Sample();
			
			double samples1f = get_num_samples(); double samples2f = other.get_num_samples();
			double combined_samplesf = combined_samples;

			double combined_mean = (samples1f*this->get_mean() + samples2f*other.get_mean())/combined_samplesf;
			double combined_std = std::pow((samples1f*(std::pow(this->get_std(), 2) + std::pow(this->get_mean() - combined_mean, 2))
								          + samples2f*(std::pow(other.get_std(), 2) + std::pow(other.get_mean() - combined_mean, 2))
								           )/combined_samplesf, 0.5);

			return Sample(combined_mean, combined_std, combined_samples);
		}

		static Sample collapse(const std::vector<Sample> &samples) {
			Sample s = samples[0];
			for (uint i = 1; i < samples.size(); i++) {
				s = s.combine(samples[i]);
			}

			return s;
		}

		static std::vector<double> get_means(const std::vector<Sample> &samples) {
			std::vector<double> v;
			for (auto const &s : samples)
				v.push_back(s.get_mean());
			return v;
		}

		std::string to_string(bool full_sample = false) const {
			if (full_sample) {
				std::string s = "[";
				s += std::to_string(this->mean) + ", " + std::to_string(this->std) + ", " + std::to_string(this->num_samples) + "]";
				return s;
			} else {
				return std::to_string(this->mean);
			}
		}
};

class DataSlide {
	public:
		Params params;
		std::map<std::string, std::vector<Sample>> data;

		DataSlide() {}
		DataSlide(Params &params) : params(params) {}

		bool contains(std::string s) const {
			return params.contains(s) || data.count(s);
		}

		var_t get(std::string s) const {
			if (contains(s)) return params.fields.at(s);

			std::cout << "Key not found: " << s << std::endl;
			assert(false);
		}

		template <typename T>
		void add_param(std::string s, T const& t) { params.add(s, t); }

		void add_param(Params &params) {
			for (auto const &[key, field] : params.fields) {
				add_param(key, field);
			}
		}

		void add_data(std::string s) { data.emplace(s, std::vector<Sample>()); }
		void push_data(std::string s, Sample sample) {
			data[s].push_back(sample);
		}

		bool remove(std::string s) {
			if (params.contains(s)) { 
				return params.remove(s);
			} else if (data.count(s)) {
				data.erase(s);
				return true;
			}
			return false;
		}

		std::string to_string(uint indentation=0, bool pretty=true, bool save_full_sample=false) const {
			std::string tab = pretty ? "\t" : "";
			std::string nline = pretty ? "\n" : "";
			std::string tabs = "";
			for (uint i = 0; i < indentation; i++) tabs += tab;
			
			std::string s = params.to_string(indentation);

			if ((params.fields.size() != 0) && (data.size() != 0)) s += "," + nline + tabs;

			std::string delim = "," + nline + tabs;
			std::vector<std::string> buffer;

			for (auto const &[key, samples] : data) {
				std::vector<std::string> sample_buffer;
				for (auto sample : samples) {
					sample_buffer.push_back(sample.to_string(save_full_sample));
				}

				buffer.push_back("\"" + key + "\": [" + join(sample_buffer, ", ") + "]");
			}

			s += join(buffer, delim);
			return s;
		}

		static DataSlide from_string(std::string ds_str) {
			nlohmann::json ds_json = nlohmann::json::parse("{" + ds_str + "}");

			DataSlide ds;

			for (auto const &[k, val] : ds_json.items()) {
				if (val.type() == nlohmann::json::value_t::array) {
					ds.add_data(k);
					for (auto const &v : val)
						ds.push_data(k, Sample(v.dump()));
				} else
					ds.add_param(k, Params::parse_json_type(val));
			}

			return ds;
		}

		bool congruent(DataSlide &ds) {
			if (params != ds.params) return false;

			for (auto const &[key, samples] : data) {
				if (!ds.data.count(key)) return false;
				if (ds.data[key].size() != data[key].size()) return false;
			}
			for (auto const &[key, val] : ds.data) {
				if (!data.count(key)) return false;
			}

			return true;
		}

		DataSlide combine(DataSlide &ds) {
			if (!congruent(ds)) {
				std::cout << "DataSlides not congruent.\n"; 
				std::cout << to_string() << "\n\n\n" << ds.to_string() << std::endl;
				assert(false);
			}

			DataSlide dn(params); 

			for (auto const &[key, samples] : data) {
				dn.add_data(key);
				for (uint i = 0; i < samples.size(); i++) {
					dn.push_data(key, samples[i].combine(ds.data[key][i]));
				}
			}

			return dn;
		}
};



class DataFrame {
	private:
		Params params;
		std::vector<DataSlide> slides;


	public:
		DataFrame() {}

		DataFrame(std::vector<DataSlide> slides) {
			for (uint i = 0; i < slides.size(); i++) add_slide(slides[i]); 
		}

		void add_slide(DataSlide ds) {
			slides.push_back(ds);
		}

		template <typename T>
		void add_param(std::string s, T const& t) { params.add(s, t); }
		void add_param(Params &params) {
			for (auto const &[key, field] : params.fields) {
				add_param(key, field);
			}
		}

		bool remove(std::string s) {
			return params.remove(s);
		}

		// TODO use nlohmann?
		void write_json(std::string filename) const {
			std::string s = "";

			s += "{\n\t\"params\": {\n";

			s += params.to_string(2);

			s += "\n\t},\n\t\"slides\": [\n";

			int num_slides = slides.size();
			std::vector<std::string> buffer;
			for (int i = 0; i < num_slides; i++) {
				buffer.push_back("\t\t{\n" + slides[i].to_string(3) + "\n\t\t}");
			}

			s += join(buffer, ",\n");

			s += "\n\t]\n}\n";

			// Save to file
			if (std::remove(filename.c_str())) std::cout << "Deleting old data\n";

			std::ofstream output_file(filename);
			output_file << s;
			output_file.close();
		}

		bool field_congruent(std::string s) {
			if (slides.size() == 0) return true;

			DataSlide first_slide = slides[0];

			if (!first_slide.contains(s)) return false;

			var_t first_slide_val = first_slide.get(s);

			for (auto slide : slides) {
				if (!slide.contains(s)) return false;
				if (slide.get(s) != first_slide_val) return false;
			}

			return true;
		}

		void promote_field(std::string s) {
			add_param(s, slides.begin()->get(s));
			for (auto &slide : slides) {
				slide.remove(s);
			}
		}

		void promote_params() {
			if (slides.size() == 0) return;

			DataSlide first_slide = slides[0];

			std::vector<std::string> keys;
			for (auto const &[key, _] : first_slide.params.fields) keys.push_back(key);
			for (auto key : keys) {
				if (field_congruent(key)) promote_field(key);
			}
		}
};

class Config {
	protected:
		Params params;

	public:
		friend class ParallelCompute;

		Config(Params &params) : params(params) {}
		Config(Config &c) : params(c.params) {}
		virtual ~Config() {}

		std::string to_string() const {
			return "{" + params.to_string() + "}";
		}

		// To implement
		virtual uint get_nruns() const { return 1; }
		virtual DataSlide compute()=0;
		virtual std::unique_ptr<Config> clone()=0;
};

static void print_progress(float progress, int expected_time = -1) {
	DO_IF_MASTER(
		int bar_width = 70;
		std::cout << "[";
		int pos = bar_width * progress;
		for (int i = 0; i < bar_width; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::stringstream time;
		if (expected_time == -1) time << "";
		else {
			time << " [ ETA: ";
			uint num_seconds = expected_time % 60;
			uint num_minutes = expected_time/60;
			uint num_hours = num_minutes/60;
			num_minutes -= num_hours*60;
			time << std::setfill('0') << std::setw(2) << num_hours << ":" 
				<< std::setfill('0') << std::setw(2) << num_minutes << ":" 
				<< std::setfill('0') << std::setw(2) << num_seconds << " ] ";
		}
		std::cout << "] " << int(progress * 100.0) << " %" << time.str()  << "\r";
		std::cout.flush();
	)
}

class ParallelCompute {
	private:
		std::vector<std::unique_ptr<Config>> configs;
		static DataSlide thread_compute(std::shared_ptr<Config> config) {
			DataSlide slide = config->compute();
			slide.add_param(config->params);
			return slide;
		}

		void compute_ompi(bool display_progress) {
#ifdef OMPI
			DO_IF_MASTER(
				std::cout << "Computing with OMPI.\n";
			)
			auto start = std::chrono::high_resolution_clock::now();

			uint num_configs = configs.size();

			std::vector<std::unique_ptr<Config>> total_configs;
			uint total_runs = 0;
			for (uint i = 0; i < num_configs; i++) {
				configs[i]->clone();
				uint nruns = configs[i]->get_nruns();
				total_runs += nruns;
				for (uint j = 0; j < nruns; j++)
					total_configs.push_back(std::move(configs[i]->clone()));
			}

			DO_IF_MASTER(
				std::cout << "num_configs: " << num_configs << std::endl;
				std::cout << "total_runs: " << total_runs << std::endl;
			)
			if (display_progress) 
				print_progress(0.);	

			std::vector<DataSlide> slides(total_runs);

			int world_size, rank;
			int index_buffer;
			int control_buffer;

			MPI_Comm_size(MPI_COMM_WORLD, &world_size);
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);

			if (rank == MASTER) {
				uint num_workers = world_size - 1;
				std::vector<bool> free_processes(num_workers, true);
				bool terminate = false;

				auto run_start = std::chrono::high_resolution_clock::now();
				uint percent_finished = 0;
				uint prev_percent_finished = percent_finished;

				uint completed = 0;

				uint head = 0;
				while (completed < total_runs) {
					// Assign work to all free processes
					for (uint j = 0; j < num_workers; j++) {
						if (head >= total_runs)
							terminate = true;

						if (free_processes[j]) {
							free_processes[j] = false;

							if (terminate) {
								control_buffer = TERMINATE;
							} else {
								control_buffer = CONTINUE;
								index_buffer = head;
							}

							MPI_Send(&index_buffer, 1, MPI_INT, j+1, control_buffer, MPI_COMM_WORLD);
							
							head++;
						}
					}

					// MASTER can also do some work here
					if (head < total_runs) {
						slides[head] = total_configs[head]->compute();
						head++;
						completed++;
					}

					if (world_size != 1) {
						// Collect results and free workers
						MPI_Status status;
						MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						int message_length;
						MPI_Get_count(&status, MPI_CHAR, &message_length);
						int message_source = status.MPI_SOURCE;
						index_buffer = status.MPI_TAG;

						char* message_buffer = (char*) std::malloc(message_length);
						MPI_Recv(message_buffer, message_length, MPI_CHAR, message_source, index_buffer, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						slides[index_buffer] = DataSlide::from_string(std::string(message_buffer));

						// Mark worker as free
						free_processes[message_source-1] = true;
						completed++;
					}

					// Display progress
					if (display_progress) {
						percent_finished = std::round(float(completed)/total_runs * 100);
						if (percent_finished != prev_percent_finished) {
							prev_percent_finished = percent_finished;
							auto elapsed = std::chrono::high_resolution_clock::now();
							int duration = std::chrono::duration_cast<std::chrono::seconds>(elapsed - run_start).count();
							float seconds_per_job = duration/float(completed);
							int remaining_time = seconds_per_job * (total_runs - completed);

							print_progress(percent_finished/100., remaining_time);
						}
					}
				}

				// Cleanup remaining workers
				for (uint i = 0; i < num_workers; i++) {
					if (free_processes[i])
						MPI_Send(&index_buffer, 1, MPI_INT, i+1, TERMINATE, MPI_COMM_WORLD);
				}

				// Construct final DataFrame and return
				uint idx = 0;
				for (uint i = 0; i < num_configs; i++) {
					DataSlide ds = slides[idx];
					uint nruns = configs[i]->get_nruns();
					for (uint j = 1; j < nruns; j++) {
						idx++;
						ds = ds.combine(slides[idx]);
					}
					idx++;

					df.add_slide(ds);
				}


				if (display_progress) {
					print_progress(1., 0);	
					std::cout << std::endl;
				}

				auto stop = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

				df.add_param("num_threads", (int) num_threads);
				df.add_param("num_jobs", (int) total_runs);
				df.add_param("time", (int) duration.count());
				df.promote_params();
			} else {
				uint idx;
				while (true) {
					// Receive control code and index
					MPI_Status status;
					MPI_Probe(MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					control_buffer = status.MPI_TAG;
					MPI_Recv(&index_buffer, 1, MPI_INT, MASTER, control_buffer, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (control_buffer == TERMINATE)
						break;

					// Do work
					DataSlide slide = total_configs[index_buffer]->compute();
					std::string message = slide.to_string(0, false, true);
					MPI_Send(message.c_str(), message.size(), MPI_CHAR, MASTER, index_buffer, MPI_COMM_WORLD);
				}
			}
#endif
		}

		void compute_serial(bool display_progress) {
			std::cout << "Computing in serial.\n";
			auto start = std::chrono::high_resolution_clock::now();

			uint num_configs = configs.size();

			std::vector<std::unique_ptr<Config>> total_configs;
			uint total_runs = 0;
			for (uint i = 0; i < num_configs; i++) {
				configs[i]->clone();
				uint nruns = configs[i]->get_nruns();
				total_runs += nruns;
				for (uint j = 0; j < nruns; j++)
					total_configs.push_back(std::move(configs[i]->clone()));
			}

			std::cout << "num_configs: " << num_configs << std::endl;
			std::cout << "total_runs: " << total_runs << std::endl;
			if (display_progress) print_progress(0.);	

			std::vector<DataSlide> slides(total_runs);

			uint idx = 0;
			auto run_start = std::chrono::high_resolution_clock::now();
			uint percent_finished = 0;
			uint prev_percent_finished = percent_finished;
			for (uint i = 0; i < num_configs; i++) {
				// Cloning and discarding calls constructors which emplace default values into params of configs[i]
				// This is a gross hack
				// TODO fix
				configs[i]->clone();
				uint nruns = configs[i]->get_nruns();
				for (uint j = 0; j < nruns; j++) {
					std::shared_ptr<Config> cfg = configs[i]->clone();
					df.add_slide(cfg->compute());
					idx++;

					if (display_progress) {
						percent_finished = std::round(float(i)/total_runs * 100);
						if (percent_finished != prev_percent_finished) {
							prev_percent_finished = percent_finished;
							auto elapsed = std::chrono::high_resolution_clock::now();
							int duration = std::chrono::duration_cast<std::chrono::seconds>(elapsed - run_start).count();
							float seconds_per_job = duration/float(i);
							int remaining_time = seconds_per_job * (total_runs - i);

							print_progress(percent_finished/100., remaining_time);
						}
					}
				}
			}

			if (display_progress) {
				print_progress(1., 0);	
				std::cout << std::endl;
			}


			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

			df.add_param("num_threads", (int) num_threads);
			df.add_param("num_jobs", (int) total_runs);
			df.add_param("time", (int) duration.count());
			df.promote_params();
		}

		void compute_bspl(bool display_progress) {
#ifndef OMPI
#ifndef SERIAL
			std::cout << "Computing with BSPL.\n";
			auto start = std::chrono::high_resolution_clock::now();

			uint num_configs = configs.size();

			std::vector<std::unique_ptr<Config>> total_configs;
			uint total_runs = 0;
			for (uint i = 0; i < num_configs; i++) {
				configs[i]->clone();
				uint nruns = configs[i]->get_nruns();
				total_runs += nruns;
				for (uint j = 0; j < nruns; j++)
					total_configs.push_back(std::move(configs[i]->clone()));
			}

			std::cout << "num_configs: " << num_configs << std::endl;
			std::cout << "total_runs: " << total_runs << std::endl;
			if (display_progress) print_progress(0.);	

			std::vector<DataSlide> slides(total_runs);
			BS::thread_pool threads(num_threads);
			std::vector<std::future<DataSlide>> results(total_runs);

			uint idx = 0;
			for (uint i = 0; i < num_configs; i++) {
				// Cloning and discarding calls constructors which emplace default values into params of configs[i]
				// This is a gross hack
				// TODO fix
				configs[i]->clone();
				uint nruns = configs[i]->get_nruns();
				for (uint j = 0; j < nruns; j++) {
					std::shared_ptr<Config> cfg = configs[i]->clone();
					results[idx] = threads.submit(ParallelCompute::thread_compute, cfg);
					idx++;
				}
			}

			auto run_start = std::chrono::high_resolution_clock::now();
			uint percent_finished = 0;
			uint prev_percent_finished = percent_finished;
			for (uint i = 0; i < total_runs; i++) {
				slides[i] = results[i].get();
				
				if (display_progress) {
					percent_finished = std::round(float(i)/total_runs * 100);
					if (percent_finished != prev_percent_finished) {
						prev_percent_finished = percent_finished;
						auto elapsed = std::chrono::high_resolution_clock::now();
						int duration = std::chrono::duration_cast<std::chrono::seconds>(elapsed - run_start).count();
						float seconds_per_job = duration/float(i);
						int remaining_time = seconds_per_job * (total_runs - i);

						print_progress(percent_finished/100., remaining_time);
					}
				}
			}

			idx = 0;
			for (uint i = 0; i < num_configs; i++) {
				DataSlide ds = slides[idx];
				uint nruns = configs[i]->get_nruns();
				for (uint j = 1; j < nruns; j++) {
					idx++;
					ds = ds.combine(slides[idx]);
				}
				idx++;

				df.add_slide(ds);
			}

			if (display_progress) {
				print_progress(1., 0);	
				std::cout << std::endl;
			}


			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

			df.add_param("num_threads", (int) num_threads);
			df.add_param("num_jobs", (int) total_runs);
			df.add_param("time", (int) duration.count());
			df.promote_params();
#endif
#endif
		}

	public:
		DataFrame df;
		uint num_threads;

		ParallelCompute(std::vector<std::unique_ptr<Config>> configs, uint num_threads) : configs(std::move(configs)),
																						  num_threads(num_threads) {}

		void compute(bool display_progress=false) {
#ifdef OMPI
			compute_ompi(display_progress);
#elif defined SERIAL
			compute_serial(display_progress);
#else
			compute_bspl(display_progress);
#endif
		}

		bool write_json(std::string filename) const {
			df.write_json(filename);
			return true;
		}
};

static std::string join(const std::vector<std::string> &v, const std::string &delim) {
    std::string s = "";
    for (const auto& i : v) {
        if (&i != &v[0]) {
            s += delim;
        }
        s += i;
    }
    return s;
}