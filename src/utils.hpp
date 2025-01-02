#pragma once

#include "types.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <stdexcept>

#include <fmt/format.h>

namespace dataframe {

namespace utils {

#define DF_ATOL 1e-6
#define DF_RTOL 1e-5

struct param_to_string {
  std::string operator()(const std::string& s) {
    return s;
  }

  std::string operator()(const double d) {
    return std::to_string(d);
  }

  std::string operator()(const int i) {
    return std::to_string(i);
  }
};

static std::string join(const std::vector<std::string>& v, const std::string& delim) {
  std::string s = "";
  for (const auto& i : v) {
    if (&i != &v[0]) {
      s += delim;
    }
    s += i;
  }

  return s;
}

static std::vector<std::string> split(const std::string& s, const std::string& delim) {
  size_t pos_start = 0, pos_end, delim_len = delim.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delim, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

static std::string strip(const std::string& input) {
  std::string whitespace = " \t\n";
  size_t start = input.find_first_not_of(whitespace);
  size_t end = input.find_last_not_of(whitespace);

  if (start != std::string::npos && end != std::string::npos) {
    return input.substr(start, end - start + 1);
  } else {
    return "";
  }
}

static void escape_sequences(std::string& str) {
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

    for (auto const& seq : sequences) {
      if (*c == seq.first) {
        *c = seq.second;
        str.insert(i, "\\");
        ++i; // to account for inserted "\\"
        break;
      }
    }
  }
}

static std::string to_string_with_precision(const double d, const int n) {
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << d;
  return std::move(out).str();
}

static bool is_float(const std::string& s) {
  try {
    size_t pos;
    std::stof(s, &pos);
    return pos == s.length();
  } catch (const std::exception& e) {
    return false;
  }
}

static bool is_integer(const std::string& str) {
  try {
    size_t pos = 0;
    std::stoi(str, &pos);
    return pos == str.length();
  } catch (const std::exception& e) {
    return false;
  }
}

struct param_eq {
  double atol;
  double rtol;

  param_eq(double atol=DF_ATOL, double rtol=DF_RTOL) : atol(atol), rtol(rtol) {}

  bool operator()(const Parameter& v, const Parameter& t) const {
    if (v.index() != t.index()) {
      return false;
    }

    if (v.index() == 0) {
      return std::get<std::string>(v) == std::get<std::string>(t);
    } else if (v.index() == 1) {
      return std::get<int>(v) == std::get<int>(t);
    } else { // Comparing doubles is the hard part
      double vd = std::get<double>(v);
      double vt = std::get<double>(t);

      // check absolute tolerance first
      if (std::abs(vd - vt) < atol) {
        return true;
      }

      double max_val = std::max(std::abs(vd), std::abs(vt));

      // both numbers are very small; use absolute comparison
      if (max_val < std::numeric_limits<double>::epsilon()) {
        return std::abs(vd - vt) < atol;
      }

      // resort to relative tolerance
      return std::abs(vd - vt)/max_val < rtol;
    }
  }
};

static bool parameter_comparison(const Parameter& lhs, const Parameter& rhs) {
  if (lhs.index() == 0 && rhs.index() == 0) {
    return std::get<std::string>(lhs) < std::get<std::string>(rhs);
  } else if (lhs.index() == 0 && rhs.index() != 0) {
    return true;
  } else if (lhs.index() != 0 && rhs.index() == 0) {
    return false;
  }

  double d1 = (lhs.index() == 1) ? static_cast<double>(std::get<int>(lhs)) : std::get<double>(lhs);
  double d2 = (rhs.index() == 1) ? static_cast<double>(std::get<int>(rhs)) : std::get<double>(rhs);

  return d1 < d2;
}

struct make_query_t_unique {
  param_eq var_visitor;

  make_query_t_unique(double atol=DF_ATOL, double rtol=DF_RTOL) {
    var_visitor = param_eq{atol, rtol};
  }

  query_t operator()(const Parameter& v) const { 
    return std::vector<Parameter>{v}; 
  }

  query_t operator()(const std::vector<Parameter>& vec) const {
    std::vector<Parameter> return_vals;

    for (auto const &tar_val : vec) {
      auto result = std::find_if(return_vals.begin(), return_vals.end(), 
        [tar_val, &var_visitor=var_visitor](const Parameter& val) {
          return var_visitor(tar_val, val);
        }
      );

      if (result == return_vals.end()) {
        return_vals.push_back(tar_val);
      }
    }

    std::sort(return_vals.begin(), return_vals.end(), parameter_comparison);

    return return_vals;
  }

  query_t operator()(const nbarray& data) const { 
    return data;
  }
};

static std::vector<query_t> make_query_unique(const std::vector<query_t>& results, const make_query_t_unique& query_t_visitor) {
  std::vector<query_t> new_results;
  new_results.reserve(results.size());
  std::transform(results.begin(), results.end(), std::back_inserter(new_results),
    [&query_t_visitor=query_t_visitor](const query_t& q) { return std::visit(query_t_visitor, q); }
  );

  return new_results;
}

static bool params_eq(const ExperimentParams& lhs, const ExperimentParams& rhs, const param_eq& equality_comparator) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto const &[key, val] : lhs) {
    if (rhs.count(key)) {
      if (!equality_comparator(rhs.at(key), val)) {
        return false;
      }
    } else {
      return false;
    }
  }

  return true;
}

template <class T>
T get(ExperimentParams &params, const std::string& key, T defaultv) {
  if (params.count(key)) {
    return std::get<T>(params[key]);
  }

  params[key] = Parameter{defaultv};
  return defaultv;
}

template <class T>
T get(const ExperimentParams &params, const std::string& key) {
  if (!params.count(key)) {
    throw std::runtime_error(fmt::format("Key \"{}\" not found in ExperimentParams.", key));
  }
  return std::get<T>(params.at(key));
}

static void emplace(SampleMap& data, const std::string& key, const std::vector<std::vector<double>>& d) {
  data.emplace(key, d);
}

static void emplace(SampleMap& data, const std::string& key, const std::vector<double>& d) {
  std::vector<std::vector<double>> d_transpose(d.size(), std::vector<double>(1));
  for (size_t i = 0; i < d.size(); i++) {
    d_transpose[i][0] = d[i];
  }
  emplace(data, key, d_transpose);
}

static void emplace(SampleMap& data, const std::string& key, double d) {
  emplace(data, key, std::vector<double>{d});
}

ExperimentParams load_params(const std::string& filename);

std::string params_to_string(const ExperimentParams& params);

std::vector<byte_t> pkl_params(const ExperimentParams& params);
void load_params_from_pkl(ExperimentParams& params, const std::vector<byte_t>& data);

}
}
