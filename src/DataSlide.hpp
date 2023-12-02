#pragma once

#include "utils.hpp"
#include "Sample.hpp"

namespace dataframe {

class DataSlide {
	public:
		Params params;
		std::map<std::string, std::vector<std::vector<Sample>>> data;

		DataSlide() {}

		DataSlide(Params &params) : params(params) {}

		DataSlide(const std::string &s) {
			std::string trimmed = s;
			uint32_t start_pos = trimmed.find_first_not_of(" \t\n\r");
			uint32_t end_pos = trimmed.find_last_not_of(" \t\n\r");
			trimmed = trimmed.substr(start_pos, end_pos - start_pos + 1);

			nlohmann::json ds_json;
			if (trimmed.empty() || trimmed.front() != '{' || trimmed.back() != '}') {
				ds_json = nlohmann::json::parse("{" + trimmed + "}");
			} else {
				ds_json = nlohmann::json::parse(trimmed);
			}

			for (auto const &[k, val] : ds_json.items()) {
				if (val.type() == nlohmann::json::value_t::array) {
					add_data(k);

					for (auto const &v : val) {
						std::vector<Sample> samples = Sample::read_samples(v); // TODO avoid dumping back to string
						push_data(k, samples);
					}
				} else {
					add_param(k, utils::parse_json_type(val));
				}
			}
		}

		DataSlide(const DataSlide& other) {
			for (auto const& [key, val]: other.params) {
				add_param(key, val);
			}

			for (auto const& [key, vals] : other.data) {
				add_data(key);
				for (auto const& val : vals) {
					push_data(key, val);
				}
			}
		}

		static DataSlide copy_params(const DataSlide& other) {
			DataSlide slide;
			for (auto const& [key, val]: other.params) {
				slide.add_param(key, val);
			}

			return slide;
		}

		bool contains(const std::string& s) const {
			return params.count(s) || data.count(s);
		}

		var_t get_param(const std::string& s) const {
			return params.at(s);
		}

		template <typename T>
		void add_param(const std::string& s, T const& t) { 
			params[s] = t; 
		}

		void add_param(const Params &params) {
			for (auto const &[key, field] : params) {
				add_param(key, field);
			}
		}

		void add_data(const std::string& s) { 
            data.emplace(s, std::vector<std::vector<Sample>>()); 
        }

		void push_data(const std::string& s, const std::vector<Sample>& samples) {
            data[s].push_back(samples);
		}

        void push_data(const std::string& s, const Sample& sample) {
			std::vector<Sample> sample_vec{sample};
			push_data(s, sample_vec);
        }

		std::vector<std::vector<double>> get_data(const std::string& s) const {
			if (!data.count(s)) {
				return std::vector<std::vector<double>>();
			}

			size_t N = data.at(s).size();
			if (N == 0) {
				return std::vector<std::vector<double>>();
			}
			
			size_t M = data.at(s)[0].size();
			std::vector<std::vector<double>> d(N, std::vector<double>(M));

			for (uint32_t i = 0; i < N; i++) {
				std::vector<Sample> di = data.at(s)[i];
				if (di.size() != M) {
					throw std::invalid_argument("Stored data is not square.");
				}

				for (uint32_t j = 0; j < M; j++) {
					d[i][j] = di[j].get_mean();
				}
			}

			return d;
		}
	
        std::vector<std::vector<double>> get_std(const std::string& s) const {
			if (!data.count(s)) {
				return std::vector<std::vector<double>>();
			}

			size_t N = data.at(s).size();
			if (N == 0) {
				return std::vector<std::vector<double>>();
			}
			
			size_t M = data.at(s)[0].size();
			std::vector<std::vector<double>> d(N, std::vector<double>(M));

			for (uint32_t i = 0; i < N; i++) {
				std::vector<Sample> di = data.at(s)[i];
				if (di.size() != M) {
					throw std::invalid_argument("Stored data is not square.");
				}

				for (uint32_t j = 0; j < M; j++) {
					d[i][j] = di[j].get_std();
				}
			}

			return d;
		}

		bool remove(const std::string& s) {
			if (params.count(s)) { 
				return params.erase(s);
			} else if (data.count(s)) {
				data.erase(s);
				return true;
			}
			return false;
		}

		std::string to_string(uint32_t indentation=0, bool pretty=true, bool record_error=false) const {
			std::string tab = pretty ? "\t" : "";
			std::string nline = pretty ? "\n" : "";
			std::string tabs = "";
			for (uint32_t i = 0; i < indentation; i++) {
				tabs += tab;
			}
			
			std::string s = utils::params_to_string(params, indentation);

			if ((!params.empty()) && (!data.empty())) {
				s += "," + nline + tabs;
			}

			std::string delim = "," + nline + tabs;
			std::vector<std::string> buffer;

			for (auto const &[key, samples] : data) {
				size_t N = samples.size();
				std::vector<std::string> sample_buffer1(N);
				for (uint32_t i = 0; i < N; i++) {
					size_t M = samples[i].size();
					std::vector<std::string> sample_buffer2(M);
					for (uint32_t j = 0; j < M; j++) {
						sample_buffer2[j] = samples[i][j].to_string(record_error);
					}

					sample_buffer1[i] = "[" + utils::join(sample_buffer2, ", ") + "]";
				}

				buffer.push_back("\"" + key + "\": [" + utils::join(sample_buffer1, ", ") + "]");
			}

			s += utils::join(buffer, delim);
			return s;
		}

		bool congruent(const DataSlide &ds, const utils::var_t_eq& equality_comparator) {
			if (!utils::params_eq(params, ds.params, equality_comparator)) {
				return false;
			}

			for (auto const &[key, samples] : data) {
				if (!ds.data.count(key)) {
					std::cout << key << " not congruent.\n";
					return false;
				}
				if (ds.data.at(key).size() != data.at(key).size()) {
					std::cout << key << " not congruent.\n";
					return false;
				}
			}
			for (auto const &[key, val] : ds.data) {
				if (!data.count(key)) {
					std::cout << key << " not congruent.\n";
					return false;
				}
			}

			return true;
		}

		DataSlide combine(const DataSlide &ds, const utils::var_t_eq& equality_comparator) {
			if (!congruent(ds, equality_comparator)) {
				std::stringstream ss;
				ss << "DataSlides not congruent.\n"; 
				ss << to_string() << "\n\n\n" << ds.to_string() << std::endl;
				std::string error_message = ss.str();
				throw std::invalid_argument(error_message);
			}

			DataSlide dn(params); 

			for (auto const &[key, samples] : data) {
				dn.add_data(key);
				for (uint32_t i = 0; i < samples.size(); i++) {
					if (samples[i].size() != ds.data.at(key)[i].size()) {
						std::string error_message = "Samples with key '" + key + "' have incongruent length and cannot be combined.";
						throw std::invalid_argument(error_message);
					}

					dn.push_data(key, Sample::combine_samples(samples[i], ds.data.at(key)[i]));
				}
			}

			return dn;
		}
};

}