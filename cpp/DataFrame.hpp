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

// TODO unifying datatype for int, float, and string so that I can avoid code duplication

#define EPS 0.0001

class Sample;
class DataSlide;
class DataFrame;
class Config;

template <typename ConfigType>
class ParallelCompute;

static std::string join(const std::vector<std::string> &v, const std::string &delim);
static std::map<std::string, Sample> to_sample(std::map<std::string, double> *s);

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
			double samples1f = get_num_samples(); double samples2f = other.get_num_samples();
			double combined_samplesf = combined_samples;

			double combined_mean = (samples1f*this->get_mean() + samples2f*other.get_mean())/combined_samplesf;
			double combined_std = std::pow(((samples1f*std::pow(this->get_std(), 2) + std::pow(this->get_mean() - combined_mean, 2))
								+ (samples2f*std::pow(other.get_std(), 2) + std::pow(other.get_mean() - combined_mean, 2))
								)/combined_samplesf, 0.5);

			return Sample(combined_mean, combined_std, combined_samples);
		}

		static Sample collapse(std::vector<Sample> samples) {
			Sample s = samples[0];
			for (int i = 1; i < samples.size(); i++) {
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
	private:
		bool congruentf(DataSlide &ds) {
			for (auto const &[key, val] : data_double) {
				if (!ds.contains_double(key)) return false;
				if (!(std::abs(val - ds.get_double(key)) < EPS)) return false;
			}

			for (auto const &[key, val] : ds.data_double) {
				if (!contains_double(key)) return false;
			}

			return true;
		}

		bool congruenti(DataSlide &ds) {
			for (auto const &[key, val] : data_int) {
				if (!ds.contains_int(key)) return false;
				if (!(val == ds.get_int(key))) return false;
			}

			for (auto const &[key, val] : ds.data_int) {
				if (!contains_int(key)) return false;
			}

			return true;
		}

		bool congruents(DataSlide &ds) {
			for (auto const &[key, val] : data_string) {
				if (!ds.contains_string(key)) return false;
				if (!(val == ds.get_string(key))) return false;
			}

			for (auto const &[key, val] : ds.data_int) {
				if (!contains_string(key)) return false;
			}

			return true;
		}

		bool congruentd(DataSlide &ds) {
			for (auto const &[key, samples] : data) {
				if (!ds.contains_data(key)) return false;
				if (!(samples.size() == ds.get_data(key)->size())) return false;
			}

			for (auto const &[key, samples] : ds.data) {
				if (!contains_data(key)) return false;
			}

			return true;
		}

	public:
		std::map<std::string, int> data_int;
		std::map<std::string, double> data_double;
		std::map<std::string, std::string> data_string;
		std::map<std::string, std::vector<Sample>> data;

		DataSlide() : data_int(std::map<std::string, int>()), 
					  data_double(std::map<std::string, double>()), 
					  data_string(std::map<std::string, std::string>()),
					  data(std::map<std::string, std::vector<Sample>>()) {}

		bool contains_int(std::string s) {
			return data_int.count(s);
		}
		bool contains_double(std::string s) {
			return data_double.count(s);
		}
		bool contains_string(std::string s) {
			return data_string.count(s);
		}
		bool contains_data(std::string s) {
			return data.count(s);
		}

		int get_int(std::string s) {
			if (contains_int(s)) {
				return data_int[s];
			} else { // TODO better error handling
				return -1;
			}
		}
		double get_double(std::string s) {
			if (contains_double(s)) {
				return data_double[s];
			} else {
				return -1.;
			}
		}
		std::string get_string(std::string s) {
			if (contains_string(s)) {
				return data_string[s];
			} else {
				return "";
			}
		}
		std::vector<Sample>* get_data(std::string s) {
			if (contains_data(s)) {
				return &data[s];
			} else {
				return nullptr;
			}
		} 
		
		void add_int(std::string s, int i) {
			data_int.emplace(s, i);
		}
		void add_double(std::string s, double f) {
			data_double.emplace(s, f);
		}
		void add_string(std::string s, std::string r) {
			data_string.emplace(s, r);
		}
		void add_data(std::string s) {
			data.emplace(s, std::vector<Sample>());
		}
		void push_data(std::string s, Sample sample) {
			data[s].push_back(sample);
		}
		bool remove_int(std::string s) {
			if (contains_int(s)) {
				data_int.erase(s);
				return true;
			}
			return false;
		}
		bool remove_double(std::string s) {
			if (contains_double(s)) {
				data_double.erase(s);
				return true;
			}
			return false;
		}
		bool remove_string(std::string s) {
			if (contains_double(s)) {
				data_string.erase(s);
				return true;
			}
			return false;
		}

		std::string to_string() const {
			std::string s = "\t\t\t";
			

			std::string delim = ",\n\t\t\t";
			std::vector<std::string> buffer;

			for (const auto &[key, val] : data_string) {
				buffer.push_back("\"" + key + "\": {\"String\": " + val + "}");
			}

			s += join(buffer, delim);

			buffer.clear();
			if (!data_int.empty()) {
				s += ",\n\t\t\t";
			}

			for (const auto &[key, val] : data_int) {
				buffer.push_back("\"" + key + "\": {\"Int\": " + std::to_string(val) + "}");
			}

			s += join(buffer, delim);

			buffer.clear();
			if (!data_double.empty()) {
				s += ",\n\t\t\t";
			}

			for (const auto &[key, val] : data_double) {
				buffer.push_back("\"" + key + "\": {\"Float\": " + std::to_string(val) + "}");
			}

			s += join(buffer, delim);

			buffer.clear();
			if (!data.empty() && ((!data_int.empty()) || !(data_double.empty()))) {
				s += ",\n\t\t\t";
			}

			std::vector<std::string> sample_buffer;
			std::string ss;

			for (const auto &[key, samples] : data) {
				ss = "\"" + key + "\": {\"Data\": [";
				sample_buffer.clear();
				for (const Sample sample : samples) {
					sample_buffer.push_back(sample.to_string());
				}
				ss += join(sample_buffer, ", ") + "]}";

				buffer.push_back(ss);
			}

			s += join(buffer, delim);

			return s;
		}

		bool congruent(DataSlide &ds) {
			return (congruenti(ds) && congruentf(ds) && congruentd(ds) && congruents(ds));
		}

		DataSlide combine(DataSlide &ds) {
			if (!congruent(ds)) {
				std::cout << "DataSlides not congruent.\n"; 
				std::cout << to_string() << "\n\n\n" << ds.to_string() << std::endl;
				assert(false);
			}

			DataSlide dn; 

			for (auto const &[key, val] : data_int) {
				dn.add_int(key, val);
			}

			for (auto const &[key, val] : data_double) {
				dn.add_double(key, val);
			}

			for (auto const &[key, val] : data_string) {
				dn.add_string(key, val);
			}

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
		std::vector<DataSlide> slides;
		std::map<std::string, int> iparams;
		std::map<std::string, double> fparams;
		std::map<std::string, std::string> sparams;


	public:
		DataFrame() : iparams(std::map<std::string, int>()), fparams(std::map<std::string, double>()), 
					  sparams(std::map<std::string, std::string>()), slides(std::vector<DataSlide>()) {}

		DataFrame(std::vector<DataSlide> slides) : iparams(std::map<std::string, int>()), 
												   fparams(std::map<std::string, double>()), 
												   sparams(std::map<std::string, std::string>()) {

			for (int i = 0; i < slides.size(); i++) {
				add_slide(slides[i]);
			}
		}

		void add_slide(DataSlide ds) {
			slides.push_back(ds);
		}
		void add_iparam(std::string s, int i) {
			iparams.emplace(s, i);
		}
		void add_fparam(std::string s, double f) {
			fparams.emplace(s, f);
		}
		void add_sparam(std::string s, std::string r) {
			sparams.emplace(s, r);
		}

		// TODO use nlohmann?
		void write_json(std::string filename) {
			std::string s = "";

			s += "{\n\t\"params\": {\n";

			std::vector<std::string> buffer;
			for (auto const &[key, val] : sparams) {
				buffer.push_back("\t\t\"" + key + "\": {\"Int\": " + val + "}");
			}
			for (auto const &[key, val] : iparams) {
				buffer.push_back("\t\t\"" + key + "\": {\"Int\": " + std::to_string(val) + "}");
			}
			for (auto const &[key, val] : fparams) {
				buffer.push_back("\t\t\"" + key + "\": {\"Int\": " + std::to_string(val) + "}");
			}

			s += join(buffer, ",\n");

			s += "\n\t},\n\t\"slides\": [\n";

			buffer.clear();
			int num_slides = slides.size();
			for (int i = 0; i < num_slides; i++) {
				buffer.push_back("\t\t{\n" + slides[i].to_string() + "\n\t\t}");
			}

			s += join(buffer, ",\n");

			s += "\n\t]\n}\n";

			// Save to file
			std::ofstream output_file(filename);
			output_file << s;
			output_file.close();
		}

		bool iparam_congruent(std::string s) {
			if (slides.size() == 0) return true;

			DataSlide *first_slide = &*slides.begin();
			if (!first_slide->contains_int(s)) return false;

			int first_slide_val = first_slide->get_int(s);

			for (auto slide : slides) {
				if (!slide.contains_int(s)) return false;
				if (!(slide.get_int(s) == first_slide_val)) return false;
			}

			return true;
		}

		bool fparam_congruent(std::string s) {
			if (slides.size() == 0) return true;

			DataSlide *first_slide = &*slides.begin();
			if (!first_slide->contains_double(s)) return false;

			double first_slide_val = first_slide->get_double(s);

			for (auto slide : slides) {
				if (!slide.contains_double(s)) return false;
				if (!(std::abs(slide.get_double(s) - first_slide_val) < EPS)) return false;
			}

			return true;
		}

		bool sparam_congruent(std::string s) {
			if (slides.size() == 0) return true;

			DataSlide *first_slide = &*slides.begin();
			if (!first_slide->contains_string(s)) return false;

			std::string first_slide_val = first_slide->get_string(s);

			for (auto slide : slides) {
				if (!slide.contains_string(s)) return false;
				if (!(slide.get_string(s) == first_slide_val)) return false;
			}

			return true;
		}

		void iparam_promote(std::string s) {
			add_iparam(s, slides.begin()->get_int(s));
			for (auto &slide : slides) {
				slide.remove_int(s);
			}
		}

		void fparam_promote(std::string s) {
			add_fparam(s, slides.begin()->get_double(s));
			for (auto &slide : slides) {
				slide.remove_double(s);
			}
		}

		void sparam_promote(std::string s) {
			add_sparam(s, slides.begin()->get_string(s));
			for (auto &slide : slides) {
				slide.remove_string(s);
			}
		}

		void promote_params() {
			if (slides.size() == 0) return;

			DataSlide *first_slide = &*slides.begin();

			std::vector<std::string> keys;
			for (auto const &[key, _] : first_slide->data_int) keys.push_back(key);
			for (auto key : keys) {
				if (iparam_congruent(key)) iparam_promote(key);
			}

			keys.clear();

			for (auto const &[key, _] : first_slide->data_double) keys.push_back(key);
			for (auto key : keys) {
				if (fparam_congruent(key)) fparam_promote(key);
			}

			keys.clear();

			for (auto const &[key, _] : first_slide->data_string) keys.push_back(key);
			for (auto key : keys) {
				if (sparam_congruent(key)) sparam_promote(key);
			}
		}
};

class Params {
	private:
		std::map<std::string, int> iparams;
		std::map<std::string, float> fparams;
		std::map<std::string, std::string> sparams;
		
	public:
		friend class Config; 

		Params() {};

		Params(std::map<std::string, int> iparams, std::map<std::string, float> fparams, std::map<std::string, std::string> sparams) {
			for (auto const &[key, val] : iparams) this->iparams[key] = val;
			for (auto const &[key, val] : fparams) this->fparams[key] = val;
			for (auto const &[key, val] : sparams) this->sparams[key] = val;
		}

		Params(Params *p) : Params(p->iparams, p->fparams, p->sparams) {}

		static std::vector<Params> load_json(json data, Params p) {
			std::vector<Params> params;

			// Dealing with model parameters
			std::vector<std::map<std::string, int>> iparams;
			std::vector<std::map<std::string, float>> fparams;
			if (data.contains("iparams")) {
				for (uint i = 0; i < data["iparams"].size(); i++) {
					iparams.push_back(std::map<std::string, int>());
					for (auto const &[key, val] : data["iparams"[i]].items()) iparams[i][key] = val;
				}
				data.erase("iparams");
			}

			if (data.contains("fparams")) {
				for (uint i = 0; i < data["fparams"].size(); i++) {
					fparams.push_back(std::map<std::string, float>());
					for (auto const &[key, val] : data["fparams"][i].items()) fparams[i][key] = val;
				}
				data.erase("fparams");
			}

			uint num_iparams = iparams.size();
			uint num_fparams = fparams.size();

			bool contains_model_iparams = num_iparams != 0;
			bool contains_model_fparams = num_fparams != 0;

			if (contains_model_iparams && contains_model_fparams) {
				assert(num_iparams == num_fparams);
				for (uint i = 0; i < num_iparams; i++) {
					for (auto const &[k, v] : iparams[i]) p.set(k, v);
					for (auto const &[k, v] : fparams[i]) p.set(k, v);
					std::vector<Params> new_params = load_json(data, Params(&p));
					params.insert(params.end(), new_params.begin(), new_params.end());
				}
				return params;
			} else if (contains_model_fparams) {
				for (uint i = 0; i < num_fparams; i++) {
					for (auto const &[k, v] : fparams[i]) p.set(k, v);
					std::vector<Params> new_params = load_json(data, Params(&p));
					params.insert(params.end(), new_params.begin(), new_params.end());
				}
				return params;
			} else if (contains_model_iparams) {
				for (uint i = 0; i < num_iparams; i++) {
					for (auto const &[k, v] : iparams[i]) p.set(k, v);
					std::vector<Params> new_params = load_json(data, Params(&p));
					params.insert(params.end(), new_params.begin(), new_params.end());
				}
				return params;
			}

			// Dealing with config parameters
			std::vector<std::string> scalars;
			std::string vector_key; // Only need one for next recursive call
			bool contains_vector = false;
			for (auto const &[key, val] : data.items()) {
				auto type = val.type();
				if ((type == json::value_t::number_integer) 
				|| (type == json::value_t::number_unsigned)
				|| (type == json::value_t::boolean)) {
					p.set(key, (int) val);
					scalars.push_back(key);
				} else if (type == json::value_t::number_float) {
					p.set(key, (float) val);
					scalars.push_back(key);
				} else if (type == json::value_t::string) {
					p.set(key, val.dump());
					scalars.push_back(key);
				} else if (type == json::value_t::array) {
					vector_key = key;
					contains_vector = true;
				}
			}

			for (auto key : scalars) data.erase(key);

			if (!contains_vector) {
				params.push_back(p);
			} else {
				std::vector<int> vals = data[vector_key];
				data.erase(vector_key);
				for (auto v : vals) {
					p.set(vector_key, v);
					std::vector<Params> new_params = load_json(data, &p);
					params.insert(params.end(), new_params.begin(), new_params.end());
				}
			}

			return params;
		}

		static std::vector<Params> load_json(json data) {
			return load_json(data, Params());
		}

		std::string to_string() const {
			std::string s = "";
			std::vector<std::string> keyvals;
			
			for (auto const &[key, val] : sparams) keyvals.push_back(key + ": " + val);
			s += join(keyvals, ", ");
			
			if (!sparams.empty() && !iparams.empty()) s += ", ";
			keyvals.clear();

			for (auto const &[key, val] : iparams) keyvals.push_back(key + ": " + std::to_string(val));
			s += join(keyvals, ", ");

			if (!iparams.empty() && !fparams.empty()) s += ", ";
			keyvals.clear();

			for (auto const &[key, val] : fparams) keyvals.push_back(key + ": " + std::to_string(val));
			s += join(keyvals, ", ");

			return s;
		}

		int geti(std::string s) const { return iparams.at(s); }
		int geti(std::string s, int defaulti) const { 
			if (iparams.count(s)) return iparams.at(s);
			else return defaulti;
		}
		int getf(std::string s) const { return fparams.at(s); }
		float getf(std::string s, float default) const { 
			if (fparams.count(s)) return fparams.at(s);
			else return defaultf
		}
		std::string gets(std::string s) const { return sparams.at(s); }
		std::string gets(std::string s, std::string defaults) const { 
			if (sparams.count(s)) return sparams.at(s);
			else return defaults
		}
		void set(std::string s, int i) { iparams[s] = i; }
		void set(std::string s, float f) { fparams[s] = f; }
		void set(std::string s, std::string r) { sparams[s] = s; }
};

class Config {
	protected:
		Params params;

	public:
		template <typename T>
		friend class ParallelCompute;

		Config(Params params) : params(params) {}
		Config(Config &c) : params(c.params) {}

		std::map<std::string, int> get_iparams() const { return params.iparams; }
		std::map<std::string, float> get_fparams() const { return params.fparams; }
		std::map<std::string, std::string> get_sparams() const { return params.sparams; }

		std::string to_string() const {
			return "{" + params.to_string() + "}";
		}

		// To implement
		virtual uint get_nruns() const { return 1; }
		virtual void compute(DataSlide *slide)=0;
};

template <typename ConfigType>
class ParallelCompute {
	private:
		std::vector<ConfigType*> configs;

	public:
		ParallelCompute(std::vector<ConfigType*> configs) : configs(configs) {}

		DataFrame compute(uint num_threads) {
			auto start = std::chrono::high_resolution_clock::now();

			uint num_configs = configs.size();
			uint total_runs = 0;
			for (auto config : configs) total_runs += config->get_nruns();

			ctpl::thread_pool threads(num_threads);
			std::vector<std::future<void>> results(total_runs);

			// Workhorse lambda functions runs compute on configs
			auto compute = [](int id, ConfigType *config, DataSlide *slide) {
				config->compute(slide);

				for (auto const &[key, val] : config->get_iparams()) {
					slide->add_int(key, val);
				}
				for (auto const &[key, val] : config->get_fparams()) {
					slide->add_double(key, val);
				}
				for (auto const &[key, val] : config->get_sparams()) {
					slide->add_string(key, val);
				}
			};

			std::vector<DataSlide> slides(total_runs);
			uint idx = 0;
			for (uint i = 0; i < num_configs; i++) {
				std::map<std::string, int> iparams = configs[i]->get_iparams();
				std::map<std::string, float> fparams = configs[i]->get_fparams();
				uint nruns = configs[i]->get_nruns();
				for (uint j = 0; j < nruns; j++) {
					results[idx] = threads.push(compute, new ConfigType(configs[i]->params), &slides[idx]);
					idx++;
				}
			}

			for (uint i = 0; i < total_runs; i++) {
				results[i].get();
			}


			DataFrame df;

			idx = 0;
			for (uint i = 0; i < num_configs; i++) {
				DataSlide ds = slides[idx];
				uint p = idx;
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

			df.add_iparam("num_threads", num_threads);
			df.add_iparam("num_jobs", total_runs);
			df.add_iparam("time", duration.count());
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

static std::map<std::string, Sample> to_sample(std::map<std::string, double> *s) {
    std::map<std::string, Sample> new_s;
    for (std::map<std::string, double>::iterator it = s->begin(); it != s->end(); ++it) {
        new_s.emplace(it->first, Sample(it->second));
    }
    return new_s;
}

#endif