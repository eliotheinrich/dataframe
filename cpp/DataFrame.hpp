#ifndef DATAFRAME_H
#define DATAFRAME_H

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
#include <BS_thread_pool.hpp>
#include <nlohmann/json.hpp>

#ifdef DEBUG
#define LOG(x) std::cout << x
#else
#define LOG(x)
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

#define EPS 0.00001

static bool operator==(const var_t& v, const var_t& t) {
	if (v.index() != t.index()) return false;

	if (v.index() == 0) return std::get<int>(v) == std::get<int>(t);
	else if (v.index() == 1) return std::abs(std::get<float>(v) - std::get<float>(t)) < EPS;
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
			}
		}

		static std::vector<Params> load_json(nlohmann::json data, Params p, bool debug) {
			if (debug) {
				std::cout << "Loaded: \n";
				std::cout << data.dump() << "\n";
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

		static Sample collapse(std::vector<Sample> samples) {
			Sample s = samples[0];
			for (uint i = 1; i < samples.size(); i++) {
				s = s.combine(samples[i]);
			}

			return s;
		}

		std::string to_string(bool err = false) const {
			if (err) {
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
		void add(std::string s, T const& t) { params.add(s, t); }

		void add(Params &params) {
			for (auto const &[key, field] : params.fields) {
				add(key, field);
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

		std::string to_string(uint indentation=0) const {
			std::string s = "";

			std::string tabs = "";
			for (uint i = 0; i < indentation; i++) tabs += "\t";
			
			s += params.to_string(indentation);

			if ((params.fields.size() != 0) && (data.size() != 0)) s += ",\n" + tabs;

			std::string delim = ",\n" + tabs;
			std::vector<std::string> buffer;

			for (auto const &[key, samples] : data) {
				std::vector<std::string> sample_buffer;
				for (auto sample : samples) {
					sample_buffer.push_back(sample.to_string());
				}

				buffer.push_back("\"" + key + "\": [" + join(sample_buffer, ", ") + "]");
			}

			s += join(buffer, delim);
			return s;
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
		void add(std::string s, T const& t) { params.add(s, t); }
		void add(Params &params) {
			for (auto const &[key, field] : params.fields) {
				add(key, field);
			}
		}

		bool remove(std::string s) {
			return params.remove(s);
		}

		// TODO use nlohmann?
		void write_json(std::string filename) {
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
			add(s, slides.begin()->get(s));
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

}

class ParallelCompute {
	private:
		std::vector<std::unique_ptr<Config>> configs;
		static DataSlide thread_compute(std::shared_ptr<Config> config) {
			DataSlide slide = config->compute();
			slide.add(config->params);
			return slide;
		}



	public:
		ParallelCompute(std::vector<std::unique_ptr<Config>> configs) : configs(std::move(configs)) {}

		DataFrame compute(uint num_threads, bool display_progress=false) {
			auto start = std::chrono::high_resolution_clock::now();

			uint num_configs = configs.size();
			uint total_runs = 0;
			for (auto const &config : configs) total_runs += config->get_nruns();

			std::cout << "num_configs: " << num_configs << std::endl;
			std::cout << "total_runs: " << total_runs << std::endl;
			if (display_progress) print_progress(0.);	

			BS::thread_pool threads(num_threads);

			std::vector<std::future<DataSlide>> results(total_runs);
			std::vector<DataSlide> slides(total_runs);
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

			if (display_progress) {
				print_progress(1., 0);	
				std::cout << std::endl;
			}

			DataFrame df;

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

			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

			df.add("num_threads", (int) num_threads);
			df.add("num_jobs", (int) total_runs);
			df.add("time", (int) duration.count());
			df.promote_params();

			return df;
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

#endif
