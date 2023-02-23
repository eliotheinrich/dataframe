#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>

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
        double num_samples;

	public:
		Sample() : mean(0.), std(0.), num_samples(0) {}
        Sample(double mean) : mean(mean), std(0.), num_samples(1) {}
		Sample(double mean, double std, int num_samples) : mean(mean), std(std), num_samples(num_samples) {}

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

        double get_num_samples() const {
			return this->num_samples;
		}
		void set_num_samples(int num_samples) {
			this->num_samples = num_samples;
		}

        Sample combine(Sample* other) const {
			int combined_samples = this->num_samples + other->get_num_samples();
			double combined_mean = (this->get_num_samples()*this->get_mean() + other->get_num_samples()*other->get_mean())/combined_samples;
			double combined_std = std::pow(((this->get_num_samples()*std::pow(this->get_std(), 2) + std::pow(this->get_mean() - combined_mean, 2))
								+ (other->get_num_samples()*std::pow(other->get_std(), 2) + std::pow(other->get_mean() - combined_mean, 2))
								)/combined_samples, 0.5);

			return Sample(combined_mean, combined_std, combined_samples);
		}

		static Sample collapse(std::vector<Sample> samples) {
			Sample s = samples[0];
			for (int i = 1; i < samples.size(); i++) {
				s.combine(&samples[i]);
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
		std::map<std::string, int> data_int;
		std::map<std::string, double> data_double;
		std::map<std::string, std::vector<Sample>> data;

		DataSlide() : data_int(std::map<std::string, int>()), 
					  data_double(std::map<std::string, double>()), 
					  data(std::map<std::string, std::vector<Sample>>()) {}

		bool contains_int(std::string s) {
			return this->data_int.count(s);
		}
		bool contains_double(std::string s) {
			return this->data_double.count(s);
		}
		bool contains_data(std::string s) {
			return this->data.count(s);
		}

		int get_int(std::string s) {
			if (this->contains_int(s)) {
				return this->data_int[s];
			} else { // TODO better error handling
				return -1;
			}
		}
		double get_double(std::string s) {
			if (this->contains_double(s)) {
				return this->data_double[s];
			} else {
				return -1.;
			}
		}
		std::vector<Sample>* get_data(std::string s) {
			if (this->contains_data(s)) {
				return &this->data[s];
			} else {
				return nullptr;
			}
		} 
		
		void add_int(std::string s, int i) {
			this->data_int.emplace(s, i);
		}
		void add_double(std::string s, double f) {
			this->data_double.emplace(s, f);
		}
		void add_data(std::string s) {
			this->data.emplace(s, std::vector<Sample>());
		}
		void push_data(std::string s, Sample sample) {
			this->data[s].push_back(sample);
		}

		std::string to_string() const {
			std::string s = "\t\t\t";
			

			std::string delim = ",\n\t\t\t";
			std::vector<std::string> buffer;

			for (const auto &[key, val] : this->data_int) {
				buffer.push_back("\"" + key + "\": {\"Int\": " + std::to_string(val) + "}");
			}

			s += join(buffer, delim);

			buffer.clear();
			if (!this->data_double.empty()) {
				s += ",\n\t\t\t";
			}

			for (const auto &[key, val] : this->data_double) {
				buffer.push_back("\"" + key + "\": {\"Float\": " + std::to_string(val) + "}");
			}

			s += join(buffer, delim);

			buffer.clear();
			if (!this->data.empty()) {
				s += ",\n\t\t\t";
			}

			std::vector<std::string> sample_buffer;
			std::string ss;

			for (const auto &[key, samples] : this->data) {
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
};




class DataFrame {
	private:
		std::map<std::string, int> params;
		std::vector<DataSlide> slides;


	public:
		DataFrame() : params(std::map<std::string, int>()), slides(std::vector<DataSlide>()) {}

		DataFrame(std::vector<DataSlide> slides) : params(std::map<std::string, int>()) {
			for (int i = 0; i < slides.size(); i++) {
				add_slide(slides[i]);
			}
		}

		void add_slide(DataSlide ds) {
			this->slides.push_back(ds);
		}
		void add_param(std::string s, int i) {
			this->params.emplace(s, i);
		}

		void write_json(std::string filename) {
			std::string s = "";

			s += "{\n\t\"params\": {\n";

			std::vector<std::string> buffer;
			for (auto const &[key, val] : this->params) {
				buffer.push_back("\t\t\"" + key + "\": {\"Int\": " + std::to_string(val) + "}");
			}

			s += join(buffer, ",\n");

			s += "\n\t},\n\t\"slides\": [\n";

			buffer.clear();
			int num_slides = this->slides.size();
			for (int i = 0; i < num_slides; i++) {
				buffer.push_back("\t\t{\n" + this->slides[i].to_string() + "\n\t\t}");
			}

			s += join(buffer, ",\n");

			s += "\n\t]\n}\n";

			// Save to file
			std::ofstream output_file(filename);
			output_file << s;
			output_file.close();
		}
};

class Config {
	private:
		template<typename T>
		std::string params_to_string(std::map<std::string, T> &params) const {
			std::vector<std::string> keyvals(0);
			for (auto const &[key, val] : params) {
				keyvals.push_back(key + ": " + std::to_string(val));
			}
			return join(keyvals, ", ");
		}

	public:
		virtual std::map<std::string, int> get_iparams() const {
			return std::map<std::string, int>();
		}
		virtual std::map<std::string, float> get_fparams() const {
			return std::map<std::string, float>();
		}

		std::string to_string() const {
			std::string s = "{";
			std::map<std::string, int>   iparams = get_iparams();
			std::map<std::string, float> fparams = get_fparams();

			s += params_to_string(iparams);
			if (!iparams.empty() && !fparams.empty()) {
				s += ", ";
			}
			s += params_to_string(fparams);

			s += "}";
			return s;
		}

		virtual void compute(DataSlide *slide)=0;
};

template <typename ConfigType>
class ParallelCompute {
	private:
		std::vector<ConfigType*> configs;

	public:
		ParallelCompute(std::vector<ConfigType*> configs) : configs(configs) {}

		DataFrame compute(uint num_threads) {
			uint num_configs = configs.size();
			std::vector<DataSlide> slides(num_configs);
			for (uint i = 0; i < num_configs; i++) {
				// TODO multithread
				// TODO pool congruent runs
				configs[i].compute(&slides[i]); 

				// Adding params
				for (auto const& [key, val] : configs[i].get_iparams()) {
					slides[i].add_int(key, val);
				}
				for (auto const& [key, val] : configs[i].get_fparams()) {
					slides[i].add_double(key, val);
				}
			}

			DataFrame df(slides);

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