#pragma once

#include "types.h"
#include "utils.h"
#include "Sample.h"

class DataSlide {
	public:
		Params params;
		std::map<std::string, std::vector<Sample>> data;

		DataSlide() {}

		DataSlide(Params &params) : params(params) {}

		DataSlide(const std::string &s);

		DataSlide(const DataSlide& other);

		static DataSlide copy_params(const DataSlide& other);

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

		void add_data(const std::string& s) { data.emplace(s, std::vector<Sample>()); }

		void push_data(const std::string& s, Sample sample) {
			data[s].push_back(sample);
		}

		void push_data(const std::string& s, double d) {
			data[s].push_back(Sample(d));
		}

		void push_data(const std::string& s, double d, double std, uint32_t num_samples) {
			data[s].push_back(Sample(d, std, num_samples));
		}

		std::vector<double> get_data(const std::string& s) const;

		std::vector<double> get_std(const std::string& s) const;

		bool remove(const std::string& s);

		std::string to_string(uint32_t indentation=0, bool pretty=true, bool record_error=false) const;

		bool congruent(const DataSlide &ds, const dataframe_utils::var_t_eq& equality_comparator);

		DataSlide combine(const DataSlide &ds, const dataframe_utils::var_t_eq& equality_comparator);
};

