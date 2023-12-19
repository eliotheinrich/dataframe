#pragma once

#include "types.h"

#include <glaze/glaze.hpp>
#include <nlohmann/json.hpp>

#include <iostream>
#include <sstream>
#include <set>

namespace dataframe {

#define ATOL 1e-6
#define RTOL 1e-5

  namespace utils {

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

    struct qvar_t_eq {
      double atol;
      double rtol;

      qvar_t_eq(double atol=ATOL, double rtol=RTOL) : atol(atol), rtol(rtol) {}

      bool operator()(const qvar_t& v, const qvar_t& t) const {
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

    struct var_t_eq {
      double atol;
      double rtol;

      var_t_eq(double atol=ATOL, double rtol=RTOL) : atol(atol), rtol(rtol) {}

      bool operator()(const var_t& v, const var_t& t) const {
        if (v.index() != t.index()) {
          return false;
        }

        if (v.index() == 0) { // comparing doubles is the tricky part
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
        } else {
          return std::get<std::string>(v) == std::get<std::string>(t);
        }
      }
    };

    static bool operator<(const qvar_t& lhs, const qvar_t& rhs) {	
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

    static bool operator<(const var_t& lhs, const var_t& rhs) {	
      if (lhs.index() == 1 && rhs.index() == 1) {
        return std::get<std::string>(lhs) < std::get<std::string>(rhs);
      } else if (lhs.index() == 1 && rhs.index() != 1) {
        return true;
      } else if (lhs.index() != 1 && rhs.index() == 1) {
        return false;
      }

      double d1 = std::get<double>(lhs);
      double d2 = std::get<double>(rhs);

      return d1 < d2;
    }

    struct var_to_qvar {
      double atol;

      qvar_t operator()(const std::string& s) {
        return qvar_t{s};
      }

      qvar_t operator()(double d) {
        int di = std::round(d);
        if (std::abs(d - static_cast<double>(di)) < atol) {
          return qvar_t{di};
        } else {
          return qvar_t{d};
        }
      }
    };

    struct make_query_t_unique {
      qvar_t_eq var_visitor;

      make_query_t_unique(double atol=ATOL, double rtol=RTOL) {
        var_visitor = qvar_t_eq{atol, rtol};
      }

      query_t operator()(const qvar_t& v) const { 
        return std::vector<qvar_t>{v}; 
      }

      query_t operator()(const std::vector<qvar_t>& vec) const {
        std::vector<qvar_t> return_vals;

        for (auto const &tar_val : vec) {
          auto result = std::find_if(return_vals.begin(), return_vals.end(), [tar_val, &var_visitor=var_visitor](const qvar_t& val) {
            return var_visitor(tar_val, val);
          });

          if (result == return_vals.end()) {
            return_vals.push_back(tar_val);
          }
        }

        std::sort(return_vals.begin(), return_vals.end());

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

    static bool params_eq(const Params& lhs, const Params& rhs, const var_t_eq& equality_comparator) {
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
    T get(Params &params, const std::string& key, T defaultv) {
      if (params.count(key)) {
        return std::get<T>(params[key]);
      }

      params[key] = var_t{defaultv};
      return defaultv;
    }

    template <>
    inline int get<int>(Params &params, const std::string& key, int defaultv) { 
      if (params.count(key)) {
        return std::round(std::get<double>(params[key]));
      }

      params[key] = var_t{static_cast<double>(defaultv)};
      return defaultv;
    }

    template <class T>
    T get(const Params &params, const std::string& key) {
      if (!params.count(key)) {
        std::string error_message = "Key \"" + key + "\" not found in Params.";
        throw std::invalid_argument(error_message);
      }
      return std::get<T>(params.at(key));
    }

    template <>
    inline int get<int>(const Params &params, const std::string& key) {
      return std::round(get<double>(params, key));
    }

    template <class json_object>
    static var_t parse_json_type(json_object p) {
    // Deprecated json deserialization

      if ((p.type() == nlohmann::json::value_t::number_integer) || 
          (p.type() == nlohmann::json::value_t::number_unsigned) ||
          (p.type() == nlohmann::json::value_t::boolean)) {
        return var_t{static_cast<double>(p)};
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
  }
}
