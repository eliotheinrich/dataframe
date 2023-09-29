#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <chrono>
#include <assert.h>
#include <exception>
#include <stdio.h>
#include <iostream>
#include <variant>
#include <optional>
#include <set>
#include <nlohmann/json.hpp>

#ifndef SERIAL
#include <BS_thread_pool.hpp>
#endif

class Sample;
class DataSlide;
class DataFrame;
class Config;
class ParallelCompute;

static std::string join(const std::vector<std::string> &, const std::string &);
static std::vector<std::string> split(const std::string &, const std::string &);
static void escape_sequences(std::string &);

#define ATOL 1e-6
#define RTOL 1e-5

// --- DEFINING VALID PARAMETER VALUES ---
typedef std::variant<int, double, std::string> var_t;

static std::string to_string_with_precision(const double d, const int n) {
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << d;
	return std::move(out).str();
}

struct var_t_to_string {
	std::string operator()(const int& i) const { return std::to_string(i); }
	std::string operator()(const double& f) const {
		for (int exp = 6; exp <= 30; exp += 6) {
			double threshold = std::pow(10.0, -static_cast<double>(exp));
			if (f > threshold)
				return to_string_with_precision(f, exp);
		}

		return "0.000000";
	}
	std::string operator()(const std::string& s) const {
		std::string tmp = s;
		escape_sequences(tmp);
		return "\"" + tmp + "\""; 
	}
};

struct var_t_eq {
	double atol;
	double rtol;

	var_t_eq(double atol=ATOL, double rtol=RTOL) : atol(atol), rtol(rtol) {}

	bool operator()(const var_t& v, const var_t& t) const {
		if (v.index() != t.index()) return false;

		if (v.index() == 0) {
			return std::get<int>(v) == std::get<int>(t);
		} else if (v.index() == 1) { // comparing doubles is the tricky part
			double vd = std::get<double>(v);
			double vt = std::get<double>(t);

			// check absolute tolerance first
			if (std::abs(vd - vt) < atol)
				return true;

			double max_val = std::max(std::abs(vd), std::abs(vt));

			// both numbers are very small; use absolute comparison
			if (max_val < std::numeric_limits<double>::epsilon())
				return std::abs(vd - vt) < atol;

			// resort to relative tolerance
			return std::abs(vd - vt)/max_val < rtol;
		} else {
			return std::get<std::string>(v) == std::get<std::string>(t);
		}
	}
};

static bool operator<(const var_t& lhs, const var_t& rhs) {
	if (lhs.index() == 2 && rhs.index() == 2) return std::get<std::string>(lhs) < std::get<std::string>(rhs);
	else if (lhs.index() == 2 && rhs.index() != 2) return true;
	else if (lhs.index() != 2 && rhs.index() == 2) return false;

	double d1 = 0., d2 = 0.;
	if (lhs.index() == 0) d1 = double(std::get<int>(lhs));
	else if (lhs.index() == 1) d1 = std::get<double>(lhs);
	else d1 = 0.;

	if (rhs.index() == 0) d2 = double(std::get<int>(rhs));
	else if (rhs.index() == 1) d2 = std::get<double>(rhs);

	return d1 < d2;
}

typedef std::map<std::string, Sample> data_t;
typedef std::map<std::string, var_t> Params;

static bool params_eq(const Params& lhs, const Params& rhs, const var_t_eq& equality_comparator) {
	if (lhs.size() != rhs.size()) {
		std::cout << "params unequal size\n";
		return false;
	}
	
	for (auto const &[key, val] : lhs) {
		if (rhs.count(key)) {
			if (!equality_comparator(rhs.at(key), val)) {
				std::cout << key << " not congruent\n";
				return false;
			}
		} else {
			std::cout << key << " not congruent\n";
			return false;
		}
	}

	return true;
}

// Debug traps; these should never be called.
static bool operator==(const var_t&, const var_t&) {
	throw std::invalid_argument("Called operator==(var_t, var_t)!");
}

static bool operator!=(const var_t&, const var_t&) {
	throw std::invalid_argument("Called operator!=(var_t, var_t)!");
}

static bool operator==(const Params&, const Params&) {
	throw std::invalid_argument("Called operator==(Params, Params)!");
}

static bool operator!=(const Params&, const Params&) {
	throw std::invalid_argument("Called operator!=(Params, Params)!");
}

// --- DEFINING VALID QUERY RESULTS ---
typedef std::variant<var_t, std::vector<var_t>, std::vector<std::vector<double>>> query_t;

struct query_t_to_string {
	std::string operator()(const var_t& v) { return std::visit(var_t_to_string(), v); }

	std::string operator()(const std::vector<var_t>& vec) {
		std::vector<std::string> buffer;
		for (auto const val : vec)
			buffer.push_back(std::visit(var_t_to_string(), val));
		return "[" + join(buffer, ", ") + "]";
	}

	std::string operator()(const std::vector<std::vector<double>>& v) {
		std::vector<std::string> buffer1;
		for (auto const& row : v) {
			std::vector<std::string> buffer2;
			for (auto const d : row)
				buffer2.push_back(std::visit(var_t_to_string(), var_t{d}));
			buffer1.push_back("[" + join(buffer2, ", ") + "]");
		}
		return "[" + join(buffer1, "\n") + "]";
	}
};

typedef std::variant<query_t, std::vector<query_t>> query_result;

struct query_to_string {
	std::string operator()(const query_t& q) { 
		return std::visit(query_t_to_string(), q); 
	}

	std::string operator()(const std::vector<query_t>& results) { 
		std::vector<std::string> buffer;
		for (auto const& q : results)
			buffer.push_back(std::visit(query_t_to_string(), q));
		
		return "[" + join(buffer, ", ") + "]";
	}
};

struct make_query_t_unique {
	var_t_eq var_visitor;

	make_query_t_unique(double atol=ATOL, double rtol=RTOL) {
		var_visitor = var_t_eq{atol, rtol};
	}

	query_t operator()(const var_t& v) const { return std::vector<var_t>{v}; }
	query_t operator()(const std::vector<std::vector<double>>& data) const { return data; }

	query_t operator()(const std::vector<var_t>& vec) const { 
		std::vector<var_t> return_vals;

		for (auto const &tar_val : vec) {
			auto result = std::find_if(return_vals.begin(), return_vals.end(), [tar_val, &var_visitor=var_visitor](const var_t& val) {
				return var_visitor(tar_val, val);
			});

			if (result == return_vals.end())
				return_vals.push_back(tar_val);
		}

		std::sort(return_vals.begin(), return_vals.end());

		return return_vals;
	}
};


struct make_query_unique {
	make_query_t_unique query_t_visitor;

	make_query_unique(double atol=ATOL, double rtol=RTOL) {
		query_t_visitor = make_query_t_unique{atol, rtol};
	}

	query_result operator()(const query_t& q) {
		return std::visit(query_t_visitor, q);
	}

	query_result operator()(const std::vector<query_t>& results) {
		std::vector<query_t> new_results(results.size());
		std::transform(results.begin(), results.end(), std::back_inserter(new_results),
			[&query_t_visitor=query_t_visitor](const query_t& q) { return std::visit(query_t_visitor, q); }
		);

		return new_results;
	}
};

template <class json_object>
static var_t parse_json_type(json_object p) {
	if ((p.type() == nlohmann::json::value_t::number_integer) || 
		(p.type() == nlohmann::json::value_t::number_unsigned) ||
		(p.type() == nlohmann::json::value_t::boolean)) {
		return var_t{(int) p};
	}  else if (p.type() == nlohmann::json::value_t::number_float) {
		return var_t{(double) p};
	} else if (p.type() == nlohmann::json::value_t::string) {
		return var_t{std::string(p)};
	} else {
		std::stringstream ss;
		ss << "Invalid json item type on " << p << "; aborting.\n";
		throw std::invalid_argument(ss.str());
	}
}

static std::string params_to_string(const Params& params, uint32_t indentation=0) {
	std::string s = "";
	for (uint32_t i = 0; i < indentation; i++) s += "\t";
	std::vector<std::string> buffer;

	for (auto const &[key, field] : params) {
		buffer.push_back("\"" + key + "\": " + std::visit(var_t_to_string(), field));
	}

	std::string delim = ",\n";
	for (uint32_t i = 0; i < indentation; i++) delim += "\t";
	s += join(buffer, delim);

	return s;
}

template <class T>
T get(Params &params, const std::string& key, T defaultv) {
	if (params.count(key))
		return std::get<T>(params[key]);
	
	params[key] = var_t{defaultv};
	return defaultv;
}

template <class T>
T get(Params &params, const std::string& key) {
	return std::get<T>(params[key]);
}

static std::vector<Params> load_json(nlohmann::json data, Params p, bool verbose) {
	if (verbose)
		std::cout << "Loaded: \n" << data.dump() << "\n";

	std::vector<Params> params;

	// Dealing with model parameters
	std::vector<std::map<std::string, var_t>> zparams;
	std::string zparam_key;
	bool found_zparam_key = false;
	for (auto const& [key, val] : data.items()) {
		if (key.find("zparams") != std::string::npos) {
			zparam_key = key;
			found_zparam_key = true;
			break;
		}
	}

	if (found_zparam_key) {
		for (uint32_t i = 0; i < data[zparam_key].size(); i++) {
			zparams.push_back(std::map<std::string, var_t>());
			for (auto const &[key, val] : data[zparam_key][i].items()) {
				if (data.contains(key)) {
					std::stringstream ss;
					ss << "Key " << key << " passed as a zipped parameter and an unzipped parameter; aborting.\n";
					throw std::invalid_argument(ss.str());
				}
				zparams[i][key] = parse_json_type(val);
			}
		}

		data.erase(zparam_key);
	}

	if (zparams.size() > 0) {
		for (uint32_t i = 0; i < zparams.size(); i++) {
			for (auto const &[k, v] : zparams[i]) p[k] = v;
			std::vector<Params> new_params = load_json(data, Params(p), false);
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
			p[key] = parse_json_type(val);
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
			p[vector_key] = parse_json_type(v);

			std::vector<Params> new_params = load_json(data, p, false);
			params.insert(params.end(), new_params.begin(), new_params.end());
		}
	}

	return params;
}

static std::vector<Params> load_json(nlohmann::json data, bool verbose=false) {
	return load_json(data, Params(), verbose);
}

static std::vector<Params> load_json(const std::string& s, bool verbose=false) {
	std::string t(s);
	std::replace(t.begin(), t.end(), '\'', '"');
	return load_json(nlohmann::json::parse(t), verbose);
}

class Sample {
    private:
        double mean;
        double std;
        uint32_t num_samples;

	public:
		Sample() : mean(0.), std(0.), num_samples(0) {}
        Sample(double mean) : mean(mean), std(0.), num_samples(1) {}
		Sample(double mean, double std, uint32_t num_samples) : mean(mean), std(std), num_samples(num_samples) {}

		template<class T>
		Sample(const std::vector<T> &v) {
			num_samples = v.size();
			mean = std::accumulate(v.begin(), v.end(), 0.0);
			double sum = 0.0;
			for (auto const t : v) {
				sum += std::pow(t - mean, 2.0);
			}

			std = std::sqrt(sum/(num_samples - 1.));
		}

		Sample(const std::string &s) {
			if (s.front() == '[' && s.back() == ']') {
				std::string trimmed = s.substr(1, s.length() - 2);
				std::vector<uint32_t> pos;
				for (uint32_t i = 0; i < trimmed.length(); i++) {
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

        uint32_t get_num_samples() const {
			return this->num_samples;
		}
		void set_num_samples(uint32_t num_samples) {
			this->num_samples = num_samples;
		}

        Sample combine(const Sample &other) const {
			uint32_t combined_samples = this->num_samples + other.get_num_samples();
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
			for (uint32_t i = 1; i < samples.size(); i++) {
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

		DataSlide(const std::string &s) {
			std::string trimmed = s;
			uint32_t start_pos = trimmed.find_first_not_of(" \t\n\r");
			uint32_t end_pos = trimmed.find_last_not_of(" \t\n\r");
			trimmed = trimmed.substr(start_pos, end_pos - start_pos + 1);

			nlohmann::json ds_json;
			if (trimmed.empty() || trimmed.front() != '{' || trimmed.back() != '}')
				ds_json = nlohmann::json::parse("{" + trimmed + "}");
			else
				ds_json = nlohmann::json::parse(trimmed);

			for (auto const &[k, val] : ds_json.items()) {
				if (val.type() == nlohmann::json::value_t::array) {
					add_data(k);
					for (auto const &v : val)
						push_data(k, Sample(v.dump()));
				} else
					add_param(k, parse_json_type(val));
			}
		}

		DataSlide(const DataSlide& other) {
			for (auto const& [key, val]: other.params)
				add_param(key, val);

			for (auto const& [key, vals] : other.data) {
				data[key] = std::vector<Sample>();
				for (auto const& val : vals)
					data[key].push_back(val);
			}
		}

		static DataSlide copy_params(const DataSlide& other) {
			DataSlide slide;
			for (auto const& [key, val]: other.params)
				slide.add_param(key, val);

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

		std::vector<double> get_data(const std::string& s) {
			if (!data.count(s))
				return std::vector<double>();

			std::vector<double> d;
			for (auto const &s : data[s])
				d.push_back(s.get_mean());
			
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

		std::string to_string(uint32_t indentation=0, bool pretty=true, bool save_full_sample=false) const {
			std::string tab = pretty ? "\t" : "";
			std::string nline = pretty ? "\n" : "";
			std::string tabs = "";
			for (uint32_t i = 0; i < indentation; i++) tabs += tab;
			
			std::string s = params_to_string(params, indentation);

			if ((!params.empty()) && (!data.empty())) s += "," + nline + tabs;

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

		bool congruent(const DataSlide &ds, const var_t_eq& equality_comparator) {
			if (!params_eq(params, ds.params, equality_comparator)) {
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

		DataSlide combine(const DataSlide &ds, const var_t_eq& equality_comparator) {
			if (!congruent(ds, equality_comparator)) {
				std::stringstream ss;
				ss << "DataSlides not congruent.\n"; 
				ss << to_string() << "\n\n\n" << ds.to_string() << std::endl;
				throw std::invalid_argument(ss.str());
			}

			DataSlide dn(params); 

			for (auto const &[key, samples] : data) {
				dn.add_data(key);
				for (uint32_t i = 0; i < samples.size(); i++)
					dn.push_data(key, samples[i].combine(ds.data.at(key)[i]));
			}

			return dn;
		}
};



class DataFrame {
	private:
		bool qtable_initialized;
		// qtable stores a list of key: {val: corresponding_slide_indices}
		std::map<std::string, std::vector<std::vector<uint32_t>>> qtable;
		std::map<std::string, std::vector<var_t>> key_vals;

		uint32_t corresponding_ind(const var_t& v, const std::vector<var_t>& vals, const std::optional<var_t_eq>& comp = std::nullopt) {
			var_t_eq equality_comparator = comp.value_or(var_t_eq{atol, rtol});

			for (uint32_t i = 0; i < vals.size(); i++) {
				if (equality_comparator(v, vals[i]))
					return i;
			}

			return -1;
		}


		void init_qtable() {
			var_t_eq equality_comparator(atol, rtol);
			key_vals = std::map<std::string, std::vector<var_t>>();

			for (auto const &slide : slides) {
				for (auto const &[key, tar_val] : slide.params) {
					if (!key_vals.count(key))
						key_vals[key] = std::vector<var_t>();
					
					auto result = std::find_if(key_vals[key].begin(), key_vals[key].end(), [tar_val, &equality_comparator](const var_t& val) {
						return equality_comparator(tar_val, val);
					});

					if (result == key_vals[key].end())
						key_vals[key].push_back(tar_val);
				}
			}

			// Setting up qtable indices
			for (auto const &[key, vals] : key_vals)
				qtable[key] = std::vector<std::vector<uint32_t>>(vals.size());

			for (uint32_t n = 0; n < slides.size(); n++) {
				auto slide = slides[n];
				for (auto const &[key, vals] : key_vals) {
					var_t val = slide.params[key];
					uint32_t idx = corresponding_ind(val, vals, equality_comparator);
					if (idx == (uint32_t) -1)
						throw std::invalid_argument("Error in init_qtable.");

					qtable[key][idx].push_back(n);
				}
			}

			qtable_initialized = true;
		}

		void promote_field(const std::string& s) {
			add_param(s, slides.begin()->get_param(s));
			for (auto &slide : slides)
				slide.remove(s);
		}

		std::set<uint32_t> compatible_inds(const Params& constraints) {
			if (!qtable_initialized)
				init_qtable();

			// Check if any keys correspond to mismatched Frame-level parameters, in which case return nothing
			var_t_eq equality_comparator(atol, rtol);
			for (auto const &[key, val] : constraints) {
				if (params.count(key) && !(equality_comparator(params[key], val)))
					return std::set<uint32_t>();
			}

			// Determine which constraints are relevant, i.e. correspond to existing Slide-level parameters
			Params relevant_constraints;
			for (auto const &[key, val] : constraints) {
				if (!params.count(key))
					relevant_constraints[key] = val;
			}

			std::set<uint32_t> inds;
			for (uint32_t i = 0; i < slides.size(); i++) inds.insert(i);
	
			for (auto const &[key, val] : relevant_constraints) {
				// Take set intersection
				std::set<uint32_t> tmp;
				uint32_t idx = corresponding_ind(val, key_vals[key], equality_comparator);
				if (idx == (uint32_t) -1)
					continue;

				for (auto const i : qtable[key][idx]) {
					if (inds.count(i))
						tmp.insert(i);
				}

				inds = tmp;
			}
			
			return inds;
		}


	public:
		double atol;
		double rtol;

		Params params;
		Params metadata;
		std::vector<DataSlide> slides;

		friend class ParallelCompute;
		
		DataFrame() : atol(ATOL), rtol(RTOL) {}

		DataFrame(const std::vector<DataSlide>& slides) : atol(ATOL), rtol(RTOL) {
			for (uint32_t i = 0; i < slides.size(); i++) add_slide(slides[i]); 
		}

		DataFrame(const Params& params, const std::vector<DataSlide>& slides) : atol(ATOL), rtol(RTOL) {
			add_param(params);
			for (uint32_t i = 0; i < slides.size(); i++) add_slide(slides[i]); 
		}

		DataFrame(const std::string& s) {
			nlohmann::json data = nlohmann::json::parse(s);
			for (auto const &[key, val] : data["params"].items())
				params[key] = parse_json_type(val);
			
			if (data.contains("metadata")) {
				for (auto const &[key, val] : data["metadata"].items())
					metadata[key] = parse_json_type(val);
			}

			// TODO use get<T>
			if (metadata.count("atol"))
				atol = std::get<double>(metadata.at("atol"));
			else
				atol = ATOL;

			if (metadata.count("rtol"))
				rtol = std::get<double>(metadata.at("rtol"));
			else
				rtol = RTOL;
		
			for (auto const &slide_str : data["slides"])
				add_slide(DataSlide(slide_str.dump()));
		}

		DataFrame(const DataFrame& other) : atol(other.atol), rtol(other.rtol) {
			for (auto const& [key, val] : other.params)
				params[key] = val;
			
			for (auto const& [key, val] : other.metadata)
				metadata[key] = val;
			
			for (auto const& slide : other.slides)
				add_slide(DataSlide(slide));
		}

		void add_slide(const DataSlide& ds) {
			slides.push_back(ds);
			qtable_initialized = false;
		}

		bool remove_slide(uint32_t i) {
			if (i >= slides.size())
				return false;

			slides.erase(slides.begin() + i);
			qtable_initialized = false;
			return true;
		}

		template <typename T>
		void add_metadata(const std::string& s, const T & t) {
			metadata[s] = t;
		}

		void add_metadata(const Params &params) {
			for (auto const &[key, field] : params)
				add_metadata(key, field);
		}

		template <typename T>
		void add_param(const std::string& s, const T & t) { 
			params[s] = t; 

			qtable_initialized = false;
		}


		void add_param(const Params &params) {
			for (auto const &[key, field] : params)
				add_param(key, field);

			qtable_initialized = false;
		}

		bool contains(const std::string& s) const {
			return params.count(s);
		}

		var_t get(const std::string& s) const {
			if (params.count(s)) return get_param(s);
			else return get_metadata(s);
		}

		var_t get_param(const std::string& s) const {
			return params.at(s);
		}

		var_t get_metadata(const std::string& s) const {
			return metadata.at(s);
		}

		bool remove(const std::string& s) {
			if (params.count(s)) return remove_param(s);
			else return remove_metadata(s);
		}

		bool remove_param(const std::string& s) {
			qtable_initialized = false;
			return params.erase(s);
		}

		bool remove_metadata(const std::string& s) {
			return metadata.erase(s);
		}

		std::string to_string() const {
			std::string s = "";

			s += "{\n\t\"params\": {\n";

			s += params_to_string(params, 2);

			s += "\n\t},\n\t\"metadata\": {\n";

			s += params_to_string(metadata, 2);

			s += "\n\t},\n\t\"slides\": [\n";

			int num_slides = slides.size();
			std::vector<std::string> buffer;
			for (int i = 0; i < num_slides; i++) {
				buffer.push_back("\t\t{\n" + slides[i].to_string(3) + "\n\t\t}");
			}

			s += join(buffer, ",\n");

			s += "\n\t]\n}\n";

			return s;
		}

		// TODO use nlohmann?
		void write_json(std::string filename) const {
			std::string s = to_string();

			// Save to file
			if (!std::remove(filename.c_str())) std::cout << "Deleting old data\n";

			std::ofstream output_file(filename);
			output_file << s;
			output_file.close();
		}

		bool field_congruent(std::string s) const {
			if (slides.size() == 0) return true;

			DataSlide first_slide = slides[0];

			if (!first_slide.contains(s)) return false;

			var_t first_slide_val = first_slide.get_param(s);

			var_t_eq equality_comparator(atol, rtol);
			for (auto slide : slides) {
				if (!slide.contains(s)) return false;
				if (!equality_comparator(slide.get_param(s), first_slide_val)) return false;
			}

			return true;
		}

		void promote_params() {
			var_t_eq equality_comparator(atol, rtol);
			if (slides.size() == 0) return;

			DataSlide first_slide = slides[0];

			std::vector<std::string> keys;
			for (auto const &[key, _] : first_slide.params) keys.push_back(key);
			for (auto key : keys)
				if (field_congruent(key)) promote_field(key);
		}

		DataFrame filter(const std::vector<Params>& constraints, bool invert = false) {
			std::set<uint32_t> inds;
			for (auto const &constraint : constraints) {
				auto c_inds = compatible_inds(constraint);
				std::set<uint32_t> ind_union;

				std::set_union(
					inds.begin(), inds.end(),
					c_inds.begin(), c_inds.end(),
					std::inserter(ind_union, ind_union.begin())
				);
				inds = ind_union;
			}

			if (invert) {
				std::vector<uint32_t> all_inds(this->slides.size());
				std::iota(all_inds.begin(), all_inds.end(), 0);

				std::set<uint32_t> all_inds_set(all_inds.begin(), all_inds.end());

				std::set<uint32_t> set_sd;
				std::set_symmetric_difference(
					inds.begin(), inds.end(),
					all_inds_set.begin(), all_inds_set.end(),
					std::inserter(set_sd, set_sd.begin())
				);
				inds = set_sd;
			}
			
			std::vector<DataSlide> slides;
			for (auto i : inds)
				slides.push_back(this->slides[i]);
			
			return DataFrame(params, slides);
		}

		query_result query(const std::vector<std::string>& keys, const Params& constraints, bool unique = false) {
			if (unique) {
				auto result = query(keys, constraints, false);
				return std::visit(make_query_unique(atol, rtol), result);
			}

			// Determine indices of slides which respect the given constraints
			auto inds = compatible_inds(constraints);
			
			// Constraints yield no valid slides, so return nothing
			if (inds.empty())
				return query_result{std::vector<var_t>()};

			// Compile result of query
			std::vector<query_t> result;
			
			for (auto const& key : keys) {
				query_t key_result;
				if (params.count(key)) {
					key_result = query_t{params[key]};
				} else if (slides[*inds.begin()].params.count(key)) {
					std::vector<var_t> param_vals;
					for (auto const i : inds)
						param_vals.push_back(slides[i].params[key]);
					key_result = query_t{param_vals};
				} else {
					std::vector<std::vector<double>> data_vals;
					for (auto const i : inds)
						data_vals.push_back(slides[i].get_data(key));
					key_result = query_t{data_vals};
				}

				result.push_back(key_result);
			}

			if (result.size() == 1)
				return query_result{result[0]};
			else
				return query_result{result};

		}

		void reduce() {
			var_t_eq equality_comparator(atol, rtol);

			std::vector<DataSlide> new_slides;

			std::set<uint32_t> reduced;
			for (uint32_t i = 0; i < slides.size(); i++) {
				if (reduced.count(i))
					continue;

				DataSlide slide(slides[i]);
				auto inds = compatible_inds(slide.params);
				for (auto const j : inds) {
					if (i == j) continue;
					slide = slide.combine(slides[j], equality_comparator);
					reduced.insert(j);
				} 
				new_slides.push_back(slide);
			}

			slides = new_slides;
		}

		DataFrame combine(const DataFrame &other) const {
			if (params.empty() && slides.empty())
				return DataFrame(other);
			else if (other.params.empty() && other.slides.empty())
				return DataFrame(*this);
			
			std::set<std::string> self_frame_params;
			for (auto const& [k, _] : params)
				self_frame_params.insert(k);

			std::set<std::string> other_frame_params;
			for (auto const& [k, _] : other.params)
				other_frame_params.insert(k);

			// both_frame_params is the intersection of keys of params and other.params
			std::set<std::string> both_frame_params;
			for (auto const& k : self_frame_params) {
				if (other_frame_params.count(k))
					both_frame_params.insert(k);
			}

			// Erase keys which appear in both frame params
			std::set<std::string> to_erase1;
			for (auto const& k : self_frame_params) {
				if (other_frame_params.count(k))
					to_erase1.insert(k);
			}

			std::set<std::string> to_erase2;
			for (auto const& k : other_frame_params) {
				if (self_frame_params.count(k))
					to_erase2.insert(k);
			}

			for (auto const& k : to_erase1)
				self_frame_params.erase(k);

			for (auto const& k : to_erase2)
				other_frame_params.erase(k);

			// self_frame_params and other_frame_params now only contain parameters unique to that frame

			DataFrame df;
			Params self_slide_params;
			Params other_slide_params;

			var_t_eq equality_comparator(atol, rtol);
			for (auto const& k : both_frame_params) {
				if (equality_comparator(params.at(k), other.params.at(k)))
					df.add_param(k, params.at(k));
				else {
					self_slide_params[k] = params.at(k);
					other_slide_params[k] = other.params.at(k);
				}
			}

			for (auto const& k : self_frame_params)
				self_slide_params[k] = params.at(k);
			
			for (auto const& k : other_frame_params)
				other_slide_params[k] = other.params.at(k);

			for (auto const& slide : slides) {
				DataSlide ds(slide);
				for (auto const& [k, v] : self_slide_params)
					ds.add_param(k, v);
				
				df.add_slide(ds);
			}

			for (auto const& slide : other.slides) {
				DataSlide ds(slide);
				for (auto const& [k, v] : other_slide_params)
					ds.add_param(k, v);
				
				df.add_slide(ds);
			}

			df.promote_params();
			df.reduce();

			return df;
		}
};

#define DEFAULT_NUM_RUNS 1
#define DEFAULT_SERIALIZE false

class Config {
	private:
		uint32_t num_runs;

	public:
		bool serialize;
		Params params;

		Config(Params &params) : params(params) {
			num_runs = get<int>(params, "num_runs", DEFAULT_NUM_RUNS);
			serialize = get<int>(params, "serialize", DEFAULT_SERIALIZE);
		}

		Config(Config &c) : Config(c.params) {}

		virtual ~Config() {}

		std::string to_string() const {
			return "{" + params_to_string(params) + "}";
		}

		uint32_t get_nruns() const { return num_runs; }

		// To implement
		virtual std::string write_serialize() const {
			return "No implementation for config.serialize() provided.\n";
		}

		virtual DataSlide compute(uint32_t num_threads)=0;
		virtual std::shared_ptr<Config> clone()=0;
		virtual std::shared_ptr<Config> deserialize(Params&, const std::string&) { return clone(); } // By default, just clone config
};

class ParallelCompute {
	private:
		uint32_t percent_finished;
		uint32_t prev_percent_finished;

		std::vector<std::shared_ptr<Config>> configs;
		DataFrame serialize_df;

		void print_progress(uint32_t i, uint32_t N, std::optional<std::chrono::high_resolution_clock::time_point> run_start = std::nullopt) {
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

		typedef std::pair<DataSlide, std::optional<std::string>> compute_result_t;

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
			uint32_t idx = 0;
			for (uint32_t i = 0; i < num_configs; i++) {
				// Cloning and discarding calls constructors which emplace default values into params of configs[i]
				// This is a gross hack
				// TODO fix
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				std::vector<std::string> serializations;
				for (uint32_t j = 0; j < nruns; j++) {
					std::shared_ptr<Config> cfg = configs[i]->clone();
					results[idx] = ParallelCompute::thread_compute(cfg, num_threads_per_task);
					idx++;

					if (verbose)
						print_progress(i, total_runs, run_start);
				}
			}

			return results;
		}

		// Return a pair of slides and corresponding state (optional) serializations
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
			uint32_t idx = 0;
			for (uint32_t i = 0; i < num_configs; i++) {
				// Cloning and discarding calls constructors which emplace default values into params of configs[i]
				// This is a gross hack
				// TODO fix
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				for (uint32_t j = 0; j < nruns; j++) {
					std::shared_ptr<Config> cfg = configs[i]->clone();
					futures[idx] = threads.submit(ParallelCompute::thread_compute, cfg, num_threads_per_task);
					idx++;
				}
			}

			for (uint32_t i = 0; i < total_runs; i++) {
				results[i] = futures[i].get();
				
				if (verbose)
					print_progress(i, total_runs, run_start);
			}

			return results;
		}

	public:
		DataFrame df;
		uint32_t num_threads;
		uint32_t num_threads_per_task;

		double atol;
		double rtol;

		bool average_congruent_runs;
		bool serialize;


		ParallelCompute(Params& metaparams, std::vector<std::shared_ptr<Config>> configs) : configs(configs) {
			num_threads = get<int>(metaparams, "num_threads", 1);
			num_threads_per_task = get<int>(metaparams, "num_threads_per_task", 1);

			serialize = get<int>(metaparams, "serialize", false);

			atol = get<double>(metaparams, "atol", ATOL);
			rtol = get<double>(metaparams, "rtol", RTOL);

			average_congruent_runs = get<int>(metaparams, "average_congruent_runs", true);

			df.add_metadata(metaparams);
			df.atol = atol;
			df.rtol = rtol;

			serialize_df.add_metadata(metaparams);
			serialize_df.atol = atol;
			serialize_df.rtol = rtol;
		}

		void compute(bool verbose=false) {
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

#ifdef SERIAL
			auto results = compute_serial(total_configs, verbose);
#else
			auto results = compute_bspl(total_configs, verbose);
#endif

			if (verbose)
				std::cout << "\n";
			
			var_t_eq equality_comparator(atol, rtol);
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
			df.add_metadata("atol", atol);
			df.add_metadata("rtol", rtol);

			serialize_df.add_metadata("num_threads", (int) num_threads);
			serialize_df.add_metadata("num_jobs", (int) num_jobs);
			serialize_df.add_metadata("total_time", (int) duration.count());
			serialize_df.add_metadata("atol", atol);
			serialize_df.add_metadata("rtol", rtol);
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

		void write_json(std::string filename) const {
			df.write_json(filename);
		}

		void write_serialize_json(std::string filename) const {
			if (serialize)
				serialize_df.write_json(filename);
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

static std::string strip(const std::string &input) {
	std::string whitespace = " \t\n";
	size_t start = input.find_first_not_of(whitespace);
	size_t end = input.find_last_not_of(whitespace);

	if (start != std::string::npos && end != std::string::npos) {
		return input.substr(start, end - start + 1);
	} else {
		return "";
	}
}

static std::vector<std::string> split(const std::string &s, const std::string &delim) {
    size_t pos_start = 0, pos_end, delim_len = delim.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delim, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

static void escape_sequences(std::string &str) {
	std::pair<char, char> const sequences[] {
		{ '\a', 'a' },
		{ '\b', 'b' },
		{ '\f', 'f' },
		{ '\n', 'n' },
		{ '\r', 'r' },
		{ '\t', 't' },
		{ '\v', 'v' },
	};

	for (size_t i = 0; i < str.length(); ++i) {
		char *const c = str.data() + i;

		for (auto const seq : sequences) {
			if (*c == seq.first) {
				*c = seq.second;
				str.insert(i, "\\");
				++i; // to account for inserted "\\"
				break;
			}
		}
	}
}
