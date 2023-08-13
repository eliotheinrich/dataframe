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
#include <stdio.h>
#include <iostream>
#include <variant>
#include <unordered_set>
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
class ParallelCompute;

static std::string join(const std::vector<std::string> &v, const std::string &delim);


// --- DEFINING VALID PARAMETER VALUES ---
typedef std::variant<int, double, std::string> var_t;

#define VAR_T_EPS 0.00001

struct var_t_to_string {
	std::string operator()(const int& i) const { return std::to_string(i); }
	std::string operator()(const double& f) const { return std::to_string(f); }
	std::string operator()(const std::string& s) const { return "\"" + s + "\""; }
};

static bool operator==(const var_t& v, const var_t& t) {
	if (v.index() != t.index()) return false;

	if (v.index() == 0) return std::get<int>(v) == std::get<int>(t);
	else if (v.index() == 1) return std::abs(std::get<double>(v) - std::get<double>(t)) < VAR_T_EPS;
	else return std::get<std::string>(v) == std::get<std::string>(t);
}

static bool operator!=(const var_t& v, const var_t& t) {
	return !(v == t);
}

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

// --- DEFINING VALID QUERY RESULTS ---
typedef std::variant<var_t, std::vector<var_t>, std::vector<std::vector<double>>> query_t;

struct make_query_t_unique {
	query_t operator()(const var_t& v) const { return v; }
	query_t operator()(const std::vector<std::vector<double>>& data) const { return data; }

	query_t operator()(const std::vector<var_t>& vec) const { 
		std::vector<var_t> return_vals;

		for (auto const &val : vec) {
			if (std::find(return_vals.begin(), return_vals.end(), val) == return_vals.end())
				return_vals.push_back(val);
		}

		std::sort(return_vals.begin(), return_vals.end());

		return return_vals;
	}
};

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
				buffer2.push_back(std::to_string(d));
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

struct make_query_unique {
	query_result operator()(const query_t& q) {
		return std::visit(make_query_t_unique(), q);
	}

	query_result operator()(const std::vector<query_t>& results) {
		std::vector<query_t> new_results(results.size());
		std::transform(results.begin(), results.end(), std::back_inserter(new_results),
			[](const query_t& q) { return std::visit(make_query_t_unique(), q); }
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
		std::cout << "Invalid json item type on " << p << "; aborting.\n";
		assert(false);

		return var_t{0};
	}
}

static std::string params_to_string(Params const& params, uint32_t indentation=0) {
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
T get(Params &params, std::string key, T defaultv) {
	if (params.count(key))
		return std::get<T>(params[key]);
	
	params[key] = var_t{defaultv};
	return defaultv;
}

template <class T>
T get(Params &params, std::string key) {
	return std::get<T>(params[key]);
}

static std::vector<Params> load_json(nlohmann::json data, Params p, bool verbose) {
	if (verbose) {
		DO_IF_MASTER(std::cout << "Loaded: \n" << data.dump() << "\n";)
	}

	std::vector<Params> params;

	// Dealing with model parameters
	std::vector<std::map<std::string, var_t>> zparams;
	if (data.contains("zparams")) {
		for (uint32_t i = 0; i < data["zparams"].size(); i++) {
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

static std::vector<Params> load_json(std::string s, bool verbose=false) {
	std::replace(s.begin(), s.end(), '\'', '"');
	return load_json(nlohmann::json::parse(s), verbose);
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

		bool contains(std::string s) const {
			return params.count(s) || data.count(s);
		}

		var_t get_param(std::string s) const {
			return params.at(s);
		}

		template <typename T>
		void add_param(std::string s, T const& t) { 
			params[s] = t; 
		}

		void add_param(Params &params) {
			for (auto const &[key, field] : params) {
				add_param(key, field);
			}
		}

		void add_data(std::string s) { data.emplace(s, std::vector<Sample>()); }

		void push_data(std::string s, Sample sample) {
			data[s].push_back(sample);
		}

		void push_data(std::string s, double d) {
			data[s].push_back(Sample(d));
		}

		void push_data(std::string s, double d, double std, uint32_t num_samples) {
			data[s].push_back(Sample(d, std, num_samples));
		}

		std::vector<double> get_data(std::string s) {
			if (!data.count(s))
				return std::vector<double>();

			std::vector<double> d;
			for (auto const &s : data[s])
				d.push_back(s.get_mean());
			
			return d;
		}

		bool remove(std::string s) {
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
				for (uint32_t i = 0; i < samples.size(); i++) {
					dn.push_data(key, samples[i].combine(ds.data[key][i]));
				}
			}

			return dn;
		}
};



class DataFrame {
	private:
		bool qtable_initialized;
		// qtable stores a list of key: {val: corresponding_slide_indices}
		std::map<std::string, std::map<var_t, std::vector<uint32_t>>> qtable;

		void init_qtable() {
			std::map<std::string, std::unordered_set<var_t>> key_vals;

			for (auto const &slide : slides) {
				for (auto const &[key, val] : slide.params) {
					if (!key_vals.count(key))
						key_vals[key] = std::unordered_set<var_t>();
					
					key_vals[key].insert(val);
				}
			}

			// Setting up keys of qtable
			for (auto const &[key, vals] : key_vals) {
				qtable[key] = std::map<var_t, std::vector<uint32_t>>();
				for (auto const &val : vals) {
					qtable[key][val] = std::vector<uint32_t>();
				}
			}

			for (uint32_t n = 0; n < slides.size(); n++) {
				auto slide = slides[n];
				for (auto const &[key, _] : key_vals)
					qtable[key][slide.params[key]].push_back(n);
			}

			qtable_initialized = true;
		}

		void promote_field(std::string s) {
			add_param(s, slides.begin()->get_param(s));
			for (auto &slide : slides) {
				slide.remove(s);
			}
		}

	public:
		Params params;
		std::vector<DataSlide> slides;
		
		DataFrame() {}

		DataFrame(const std::vector<DataSlide>& slides) {
			for (uint32_t i = 0; i < slides.size(); i++) add_slide(slides[i]); 
		}

		DataFrame(const std::string& s) {
			nlohmann::json data = nlohmann::json::parse(s);
			for (auto const &[key, val] : data["params"].items())
				params[key] = parse_json_type(val);
		
			for (auto const &slide_str : data["slides"]) {
				add_slide(DataSlide(slide_str.dump()));
			}
		}

		DataFrame(const DataFrame& other) {
			for (auto const& [key, val] : other.params)
				params[key] = val;
			
			for (auto const& slide : other.slides)
				add_slide(DataSlide(slide));
		}

		void add_slide(DataSlide ds) {
			slides.push_back(ds);
			qtable_initialized = false;
		}

		template <typename T>
		void add_param(std::string s, T const& t) { 
			params[s] = t; 

			qtable_initialized = false;
		}

		void add_param(Params &params) {
			for (auto const &[key, field] : params) {
				add_param(key, field);
			}

			qtable_initialized = false;
		}

		bool contains(std::string s) const {
			return params.count(s);
		}

		var_t get_param(std::string s) const {
			return params.at(s);
		}

		bool remove(std::string s) {
			qtable_initialized = false;
			return params.erase(s);
		}

		std::string to_string() const {
			std::string s = "";

			s += "{\n\t\"params\": {\n";

			s += params_to_string(params, 2);

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
			if (std::remove(filename.c_str())) std::cout << "Deleting old data\n";

			std::ofstream output_file(filename);
			output_file << s;
			output_file.close();
		}

		bool field_congruent(std::string s) const {
			if (slides.size() == 0) return true;

			DataSlide first_slide = slides[0];

			if (!first_slide.contains(s)) return false;

			var_t first_slide_val = first_slide.get_param(s);

			for (auto slide : slides) {
				if (!slide.contains(s)) return false;
				if (slide.get_param(s) != first_slide_val) return false;
			}

			return true;
		}

		void promote_params() {
			if (slides.size() == 0) return;

			DataSlide first_slide = slides[0];

			std::vector<std::string> keys;
			for (auto const &[key, _] : first_slide.params) keys.push_back(key);
			for (auto key : keys) {
				if (field_congruent(key)) promote_field(key);
			}
		}

		query_result query(std::vector<std::string> keys, std::map<std::string, var_t> constraints, bool unique = false) {
			if (unique) {
				auto result = query(keys, constraints, false);
				return std::visit(make_query_unique(), result);
			}

			if (!qtable_initialized)
				init_qtable();

			// Check if any keys correspond to mismatched Frame-level parameters, in which case return nothing
			for (auto const &[key, val] : constraints) {
				if (params.count(key) && params[key] != val)
					return query_t{std::vector<var_t>()};
			}

			// Determine which constraints are relevant, i.e. correspond to existing Slide-level parameters
			std::map<std::string, var_t> relevant_constraints;
			for (auto const &[key, val] : constraints) {
				if (!params.count(key))
					relevant_constraints[key] = val;
			}

			// Determine indices of slides which respect the given constraints
			std::unordered_set<uint32_t> inds;
			for (uint32_t i = 0; i < slides.size(); i++) inds.insert(i);
	
			for (auto const &[key, val] : relevant_constraints) {
				// Take set intersection
				std::unordered_set<uint32_t> tmp;
				for (auto const i : qtable[key][val]) {
					if (inds.count(i))
						tmp.insert(i);
				}

				inds = tmp;
			}
			

			// Compile result of query
			std::vector<query_t> result;
			
			for (auto const& key : keys) {
				query_t key_result;
				if (params.count(key)) {
					key_result = query_t{params[key]};
				} else if (slides[0].params.count(key)) {
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

		DataFrame combine(const DataFrame &other) const {
			if (params.empty() && slides.empty())
				return DataFrame(other);
			else if (other.params.empty() && other.slides.empty())
				return DataFrame(*this);
			
			std::unordered_set<std::string> self_frame_params;
			for (auto const& [k, _] : params)
				self_frame_params.insert(k);

			std::unordered_set<std::string> other_frame_params;
			for (auto const& [k, _] : other.params)
				other_frame_params.insert(k);

			// both_frame_params is the intersection of keys of params and other.params
			std::unordered_set<std::string> both_frame_params;
			for (auto const& k : self_frame_params) {
				if (other_frame_params.count(k))
					both_frame_params.insert(k);
			}

			// Erase keys which appear in both frame params
			std::unordered_set<std::string> to_erase;
			for (auto const& k : self_frame_params) {
				if (other_frame_params.count(k))
					to_erase.insert(k);
			}

			for (auto const& k : to_erase)
				self_frame_params.erase(k);

			to_erase.clear();
			for (auto const& k : other_frame_params) {
				if (self_frame_params.count(k))
					to_erase.insert(k);
			}

			for (auto const& k : to_erase)
				other_frame_params.erase(k);

			// self_frame_params and other_frame_params now only contain parameters unique to that frame
			

			DataFrame df;
			Params self_slide_params;
			Params other_slide_params;

			for (auto const& k : both_frame_params) {
				if (params.at(k) == other.params.at(k))
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

			return df;
		}
};

class Config {
	protected:
		Params params;
		uint32_t num_runs;

	public:
		friend class ParallelCompute;

		Config(Params &params) : params(params) {
			num_runs = get<int>(params, "num_runs");
		}
		Config(Config &c) : Config(c.params) {}

		virtual ~Config() {}

		std::string to_string() const {
			return "{" + params_to_string(params) + "}";
		}

		// To implement
		virtual uint32_t get_nruns() const { return num_runs; }
		virtual DataSlide compute()=0;
		virtual std::shared_ptr<Config> clone()=0;
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
			uint32_t num_seconds = expected_time % 60;
			uint32_t num_minutes = expected_time/60;
			uint32_t num_hours = num_minutes/60;
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
		std::vector<std::shared_ptr<Config>> configs;
		static DataSlide thread_compute(std::shared_ptr<Config> config) {
			DataSlide slide = config->compute();
			slide.add_param(config->params);
			return slide;
		}

		void compute_ompi(bool verbose) {
#ifdef OMPI
			auto start = std::chrono::high_resolution_clock::now();

			uint32_t num_configs = configs.size();

			std::vector<std::shared_ptr<Config>> total_configs;
			uint32_t total_runs = 0;
			for (uint32_t i = 0; i < num_configs; i++) {
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				total_runs += nruns;
				for (uint32_t j = 0; j < nruns; j++)
					total_configs.push_back(std::move(configs[i]->clone()));
			}

			std::vector<DataSlide> slides(total_runs);

			int world_size, rank;
			int index_buffer;
			int control_buffer;

			MPI_Comm_size(MPI_COMM_WORLD, &world_size);
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);

			if (rank == MASTER) {
				if (verbose) {
					std::cout << "Computing with OMPI.\n";
					std::cout << "num_configs: " << num_configs << std::endl;
					std::cout << "total_runs: " << total_runs << std::endl;
					print_progress(0.);	
				}

				uint32_t num_workers = world_size - 1;
				std::vector<bool> free_processes(num_workers, true);
				bool terminate = false;

				auto run_start = std::chrono::high_resolution_clock::now();
				uint32_t percent_finished = 0;
				uint32_t prev_percent_finished = percent_finished;

				uint32_t completed = 0;

				uint32_t head = 0;
				while (completed < total_runs) {
					// Assign work to all free processes
					for (uint32_t j = 0; j < num_workers; j++) {
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
						slides[index_buffer] = DataSlide(std::string(message_buffer));

						// Mark worker as free
						free_processes[message_source-1] = true;
						completed++;
					}

					// Display progress
					if (verbose) {
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
				for (uint32_t i = 0; i < num_workers; i++) {
					if (free_processes[i])
						MPI_Send(&index_buffer, 1, MPI_INT, i+1, TERMINATE, MPI_COMM_WORLD);
				}

				// Construct final DataFrame and return
				uint32_t idx = 0;
				for (uint32_t i = 0; i < num_configs; i++) {
					DataSlide ds = slides[idx];
					uint32_t nruns = configs[i]->get_nruns();
					for (uint32_t j = 1; j < nruns; j++) {
						idx++;
						ds = ds.combine(slides[idx]);
					}
					idx++;

					df.add_slide(ds);
				}


				if (verbose) {
					print_progress(1., 0);	
					std::cout << std::endl;
				}

				auto stop = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

				df.add_param("num_threads", (int) num_threads);
				df.add_param("num_jobs", (int) total_runs);
				df.add_param("total_time", (int) duration.count());
				df.promote_params();
				if (verbose)
					std::cout << "Total runtime: " << (int) duration.count() << std::endl;

			} else {
				uint32_t idx;
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

		void compute_serial(bool verbose) {
			auto start = std::chrono::high_resolution_clock::now();

			uint32_t num_configs = configs.size();

			std::vector<std::shared_ptr<Config>> total_configs;
			uint32_t total_runs = 0;
			for (uint32_t i = 0; i < num_configs; i++) {
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				total_runs += nruns;
				for (uint32_t j = 0; j < nruns; j++)
					total_configs.push_back(std::move(configs[i]->clone()));
			}

			if (verbose) {
				std::cout << "Computing in serial.\n";
				std::cout << "num_configs: " << num_configs << std::endl;
				std::cout << "total_runs: " << total_runs << std::endl;
				print_progress(0.);	
			}

			std::vector<DataSlide> slides(total_runs);

			uint32_t idx = 0;
			auto run_start = std::chrono::high_resolution_clock::now();
			uint32_t percent_finished = 0;
			uint32_t prev_percent_finished = percent_finished;
			for (uint32_t i = 0; i < num_configs; i++) {
				// Cloning and discarding calls constructors which emplace default values into params of configs[i]
				// This is a gross hack
				// TODO fix
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				for (uint32_t j = 0; j < nruns; j++) {
					std::shared_ptr<Config> cfg = configs[i]->clone();
					DataSlide slide = cfg->compute();
					slide.add_param(cfg->params);
					df.add_slide(slide);
					idx++;

					if (verbose) {
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

			if (verbose) {
				print_progress(1., 0);	
				std::cout << std::endl;
			}


			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

			df.add_param("num_threads", (int) num_threads);
			df.add_param("num_jobs", (int) total_runs);
			df.add_param("total_time", (int) duration.count());
			df.promote_params();

			if (verbose)
				std::cout << "Total runtime: " << (int) duration.count() << std::endl;
		}

		void compute_bspl(bool verbose) {
#ifndef OMPI
#ifndef SERIAL
			auto start = std::chrono::high_resolution_clock::now();

			uint32_t num_configs = configs.size();

			std::vector<std::shared_ptr<Config>> total_configs;
			uint32_t total_runs = 0;
			for (uint32_t i = 0; i < num_configs; i++) {
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				total_runs += nruns;
				for (uint32_t j = 0; j < nruns; j++)
					total_configs.push_back(std::move(configs[i]->clone()));
			}

			if (verbose) {
				std::cout << "Computing with BSPL. " << num_threads << " threads available.\n";
				std::cout << "num_configs: " << num_configs << std::endl;
				std::cout << "total_runs: " << total_runs << std::endl;
				print_progress(0.);	
			}

			std::vector<DataSlide> slides(total_runs);
			BS::thread_pool threads(num_threads);
			std::vector<std::future<DataSlide>> results(total_runs);

			uint32_t idx = 0;
			for (uint32_t i = 0; i < num_configs; i++) {
				// Cloning and discarding calls constructors which emplace default values into params of configs[i]
				// This is a gross hack
				// TODO fix
				configs[i]->clone();
				uint32_t nruns = configs[i]->get_nruns();
				for (uint32_t j = 0; j < nruns; j++) {
					std::shared_ptr<Config> cfg = configs[i]->clone();
					results[idx] = threads.submit(ParallelCompute::thread_compute, cfg);
					idx++;
				}
			}

			auto run_start = std::chrono::high_resolution_clock::now();
			uint32_t percent_finished = 0;
			uint32_t prev_percent_finished = percent_finished;
			for (uint32_t i = 0; i < total_runs; i++) {
				slides[i] = results[i].get();
				
				if (verbose) {
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
			for (uint32_t i = 0; i < num_configs; i++) {
				DataSlide ds = slides[idx];
				uint32_t nruns = configs[i]->get_nruns();
				for (uint32_t j = 1; j < nruns; j++) {
					idx++;
					ds = ds.combine(slides[idx]);
				}
				idx++;

				df.add_slide(ds);
			}

			if (verbose) {
				print_progress(1., 0);	
				std::cout << std::endl;
			}


			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

			df.add_param("num_threads", (int) num_threads);
			df.add_param("num_jobs", (int) total_runs);
			df.add_param("total_time", (int) duration.count());
			df.promote_params();

			if (verbose)
				std::cout << "Total runtime: " << (int) duration.count() << std::endl;
#endif
#endif
		}

	public:
		DataFrame df;
		uint32_t num_threads;

		ParallelCompute(std::vector<std::shared_ptr<Config>> configs, uint32_t num_threads) : configs(std::move(configs)),
																						  num_threads(num_threads) {}

		void compute(bool verbose=false) {
#ifdef OMPI
			compute_ompi(verbose);
#elif defined SERIAL
			compute_serial(verbose);
#else
			compute_bspl(verbose);
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