#pragma once

#include "types.h"

#include <nlohmann/json.hpp>
#include <iostream>

namespace dataframe_utils {

#define ATOL 1e-6
#define RTOL 1e-5


std::string join(const std::vector<std::string> &, const std::string &);
std::vector<std::string> split(const std::string &, const std::string &);
std::string strip(const std::string &input);
void escape_sequences(std::string &);
std::string to_string_with_precision(const double d, const int n);

struct var_t_to_string {
    std::string operator()(const int &i) const;
    std::string operator()(const double &f) const;
    std::string operator()(const std::string &s) const;
};

struct var_t_eq {
	double atol;
	double rtol;

    var_t_eq(double atol=ATOL, double rtol=RTOL) : atol(atol), rtol(rtol) {}

	bool operator()(const var_t& v, const var_t& t) const;
};

bool operator<(const var_t& lhs, const var_t& rhs);

bool params_eq(const Params& lhs, const Params& rhs, const var_t_eq& equality_comparator);

// Debug traps; these should never be called.
//bool operator==(const var_t&, const var_t&) {
//	throw std::invalid_argument("Called operator==(var_t, var_t)!");
//}
//
//bool operator!=(const var_t&, const var_t&) {
//	throw std::invalid_argument("Called operator!=(var_t, var_t)!");
//}
//
//bool operator==(const Params&, const Params&) {
//	throw std::invalid_argument("Called operator==(Params, Params)!");
//}
//
//bool operator!=(const Params&, const Params&) {
//	throw std::invalid_argument("Called operator!=(Params, Params)!");
//}

struct query_t_to_string {
	std::string operator()(const var_t& v) const;

	std::string operator()(const std::vector<var_t>& vec) const;

	std::string operator()(const std::vector<std::vector<double>>& v) const;
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

	query_t operator()(const std::vector<var_t>& vec) const;
};


struct make_query_unique {
	make_query_t_unique query_t_visitor;

	make_query_unique(double atol=ATOL, double rtol=RTOL) {
		query_t_visitor = make_query_t_unique{atol, rtol};
	}

	query_result operator()(const query_t& q) {
		return std::visit(query_t_visitor, q);
	}

	query_result operator()(const std::vector<query_t>& results);
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

std::string params_to_string(const Params& params, uint32_t indentation=0);

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

std::vector<Params> load_json(nlohmann::json data, Params p, bool verbose);
std::vector<Params> load_json(nlohmann::json data, bool verbose=false);
std::vector<Params> load_json(const std::string& s, bool verbose=false);

std::string write_config(const std::vector<Params>& params);

}