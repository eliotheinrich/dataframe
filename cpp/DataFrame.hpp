#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <chrono>
#include <ctpl.h>
#include <assert.h>
#include <iostream>
#include <nlohmann/json.hpp>

class Sample;
class DataSlide;
class DataFrame;
class Config;
class Params;
class ParallelCompute;

static std::string join(const std::vector<std::string> &v, const std::string &delim);

enum datafield_t { df_float, df_int, df_string };
struct datafield {
	private:
		datafield_t _type;
		float fdata;
		int idata;
		std::string sdata;

		constexpr static const float EPS = 0.0001;

	public:
		datafield_t type() const { return _type; }

		datafield() {};
		datafield(int i) {
			_type = datafield_t::df_int;
			idata = i;
		}

		datafield(float f) {
			_type = datafield_t::df_float;
			fdata = f;
		}

		datafield(std::string s) {
			_type = datafield_t::df_string;
			sdata = s;
		}

		int unwrapi() const { return idata; }
		float unwrapf() const { return fdata; }
		std::string unwraps() const { return sdata; }

		std::string to_string() const {
			switch(type()) {
				case datafield_t::df_int : return std::to_string(idata);
				case datafield_t::df_float : return std::to_string(fdata);
				case datafield_t::df_string : return "\"" + sdata + "\"";
			}
			std::cout << "Printed datafield is invalid.\n";
			assert(false);
			return "";
		}

		datafield clone() {
			switch(type()) {
				case datafield_t::df_int : return datafield(idata);
				case datafield_t::df_float : return datafield(fdata);
				case datafield_t::df_string : return datafield(sdata);
			}
			std::cout << "Cloned datafield is invalid.\n";
			assert(false);
			return datafield(idata);
		}

		bool operator==(const datafield &df) {
			if (type() != df.type()) return false;
			
			if (type() == datafield_t::df_int) return idata == df.idata;
			if (type() == datafield_t::df_float) return std::abs(fdata - df.fdata) < EPS;
			if (type() == datafield_t::df_string) return sdata == df.sdata;

			return false;
		}

		bool operator!=(const datafield &df) {
			return !((*this) == df);
		}
};

class Params {		
	public:
		std::map<std::string, datafield> fields;

		Params() {}
		~Params() {}

		Params(Params *p) {
			for (auto &[key, val] : p->fields) fields.emplace(key, val.clone());
		}

		std::string to_string(uint indentation=0) const {
			std::string s = "";
			for (uint i = 0; i < indentation; i++) s += "\t";
			std::vector<std::string> buffer;
			

			for (auto const &[key, field] : fields) {
				buffer.push_back("\"" + key + "\": " + field.to_string());
			}

			std::string delim = ",\n";
			for (uint i = 0; i < indentation; i++) delim += "\t";
			s += join(buffer, delim);

			return s;
		}

		datafield get(std::string s) const { 
			if (fields.count(s)) {
				return fields.at(s);
			} else {
				std::cout << "Key \"" + s + "\" not found.\n"; assert(false);
			}
		}
		int geti(std::string s) const { return get(s).unwrapi(); }
		int geti(std::string s, int defaulti) { 
			if (fields.count(s) && fields.at(s).type() == datafield_t::df_int) return get(s).unwrapi();
			else {
				add(s, defaulti);
				return defaulti;
			}
		}
		float getf(std::string s) const { return fields.at(s).unwrapf(); }
		float getf(std::string s, float defaultf) { 
			if (fields.count(s) && fields.at(s).type() == datafield_t::df_float) return get(s).unwrapf();
			else {
				add(s, defaultf);
				return defaultf;
			}
		}
		std::string gets(std::string s) const { return fields.at(s).unwraps(); }
		std::string gets(std::string s, std::string defaults) { 
			if (fields.count(s) && fields.at(s).type() == datafield_t::df_string) return get(s).unwraps();
			else {
				add(s, defaults);
				return defaults;
			};
		}

		void add(std::string s, datafield df) { fields[s] = df; }
		void add(std::string s, int i) { fields[s] = datafield(i); }
		void add(std::string s, float f) { fields[s] = datafield(f); }
		void add(std::string s, std::string r) { fields[s] = datafield(r); }

		bool contains(std::string s) const { return fields.count(s); }
		bool remove(std::string s) {
			if (fields.count(s)) {
				fields.erase(s);
				return true;
			}
			return false;
		}

		bool operator==(const Params &p) {
			for (auto const &[key, field] : fields) {
				if (!p.contains(key)) return false;
				if (p.get(key) != field) return false;
			}
			for (auto const &[key, field] : p.fields) {
				if (!contains(key)) return false;
			}

			return true;
		}

		bool operator!=(const Params &p) {
			return !((*this) == p);
		}

		template <typename json_object>
		static datafield parse_json_type(json_object p) {
			if ((p.type() == nlohmann::json::value_t::number_integer) || 
				(p.type() == nlohmann::json::value_t::number_unsigned) ||
				(p.type() == nlohmann::json::value_t::boolean)) {
				return datafield((int) p);
			}  else if (p.type() == nlohmann::json::value_t::number_float) {
				return datafield((float) p);
			} else if (p.type() == nlohmann::json::value_t::string) {
				return datafield(std::string(p));
			} else {
				std::cout << "Invalid json item type on " << p << "; aborting.\n";
				assert(false);
				return datafield();
			}
		}

		static std::vector<Params> load_json(nlohmann::json data, Params p, bool debug) {
			if (debug) {
				std::cout << "Loaded: \n";
				std::cout << data.dump() << "\n";
			}
			std::vector<Params> params;

			// Dealing with model parameters
			std::vector<std::map<std::string, datafield>> zparams;
			if (data.contains("zparams")) {
				for (uint i = 0; i < data["zparams"].size(); i++) {
					zparams.push_back(std::map<std::string, datafield>());
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

		std::string to_string() const {
			std::string s = "[";
			s += std::to_string(this->mean) + ", " + std::to_string(this->std) + ", " + std::to_string(this->num_samples) + "]";
			return s;
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

		datafield get(std::string s) { return params.get(s); }
		int geti(std::string s) { return params.geti(s); }
		double getf(std::string s) { return params.getf(s); }
		std::string gets(std::string s) { return params.gets(s); }

		void add(std::string s, datafield df) { params.add(s, df); }
		void add(std::string s, int i) { params.add(s, datafield(i)); }
		void add(std::string s, float f) { params.add(s, datafield(f)); };
		void add(std::string s, std::string r) { params.add(s, datafield(r)); }
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
		int geti(std::string s) { return params.geti(s); }
		double getf(std::string s) { return params.getf(s); }
		std::string gets(std::string s) { return params.gets(s); }

		void add(std::string s, datafield df) { params.add(s, df); }
		void add(std::string s, int i) { params.add(s, datafield(i)); }
		void add(std::string s, float f) { params.add(s, datafield(f)); };
		void add(std::string s, std::string r) { params.add(s, datafield(r)); }
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
			std::ofstream output_file(filename);
			output_file << s;
			output_file.close();
		}

		bool field_congruent(std::string s) {
			if (slides.size() == 0) return true;

			DataSlide first_slide = slides[0];

			if (!first_slide.contains(s)) return false;

			datafield first_slide_val = first_slide.get(s);

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

static void print_progress(float progress) {
	int bar_width = 70;

	std::cout << "[";
	int pos = bar_width * progress;
	for (int i = 0; i < bar_width; ++i) {
		if (i < pos) std::cout << "=";
		else if (i == pos) std::cout << ">";
		else std::cout << " ";
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	std::cout.flush();

}


class ParallelCompute {
	private:
		std::vector<std::unique_ptr<Config>> configs;

		static DataSlide thread_compute(int id, std::unique_ptr<Config> &config) {
			DataSlide slide = config->compute();
			slide.add(config->params);
			config.release();
			return slide;
		}

	public:
		ParallelCompute(std::vector<std::unique_ptr<Config>> configs) : configs(std::move(configs)) {}

		DataFrame compute(uint num_threads, bool display_progress=false) {
			auto start = std::chrono::high_resolution_clock::now();

			uint num_configs = configs.size();
			uint total_runs = 0;
			for (auto &config : configs) total_runs += config->get_nruns();

			std::cout << "num_configs: " << num_configs << std::endl;
			std::cout << "total_runs: " << total_runs << std::endl;
			if (display_progress) print_progress(0.);	

			ctpl::thread_pool threads(num_threads);

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
					results[idx] = threads.push(thread_compute, configs[i]->clone());
					idx++;
				}
			}

			uint percent_finished = 0;
			uint prev_percent_finished = percent_finished;
			for (uint i = 0; i < total_runs; i++) {
				slides[i] = results[i].get();
				
				if (display_progress) {
					percent_finished = std::round(float(i)/total_runs * 100);
					if (percent_finished != prev_percent_finished) {
						prev_percent_finished = percent_finished;
						print_progress(percent_finished/100.);
					}
				}
			}

			if (display_progress) {
				print_progress(1.);	
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