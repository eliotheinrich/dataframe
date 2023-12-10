#pragma once

#include "types.h"

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

    struct var_t_to_string {
      std::string operator()(const int &i) const {
        return std::to_string(i); 
      }

      std::string operator()(const double &f) const {
        for (int exp = 6; exp <= 30; exp += 6) {
          double threshold = std::pow(10.0, -static_cast<double>(exp));
          if (f > threshold) {
            return to_string_with_precision(f, exp);
          }
        }

        return "0.000000";
      }

      std::string operator()(const std::string &s) const {
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
        if (v.index() != t.index()) {
          return false;
        }

        if (v.index() == 0) {
          return std::get<int>(v) == std::get<int>(t);
        } else if (v.index() == 1) { // comparing doubles is the tricky part
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

    static bool operator<(const var_t& lhs, const var_t& rhs) {	
      if (lhs.index() == 2 && rhs.index() == 2) {
        return std::get<std::string>(lhs) < std::get<std::string>(rhs);
      } else if (lhs.index() == 2 && rhs.index() != 2) {
        return true;
      } else if (lhs.index() != 2 && rhs.index() == 2) {
        return false;
      }

      double d1 = 0., d2 = 0.;
      if (lhs.index() == 0) {
        d1 = double(std::get<int>(lhs));
      } else if (lhs.index() == 1) {
        d1 = std::get<double>(lhs);
      } else {
        d1 = 0.;
      }

      if (rhs.index() == 0) {
        d2 = double(std::get<int>(rhs));
      } else if (rhs.index() == 1) {
        d2 = std::get<double>(rhs);
      }

      return d1 < d2;
    }

    struct query_t_to_string {
      std::string operator()(const var_t& v) const {
        return std::visit(var_t_to_string(), v); 
      }

      std::string operator()(const std::vector<var_t>& vec) const {
        var_t_to_string vt;
        std::vector<std::string> buffer(vec.size());
        for (auto const val : vec) {
          buffer.push_back(std::visit(vt, val));
        }
        return "[" + join(buffer, ", ") + "]";
      }

      std::string operator()(const nbarray& arr) const {
        var_t_to_string vt;

        size_t N = arr.size();

        std::vector<std::string> buffer1(N);
        for (size_t i = 0; i < N; i++) {
          size_t M = arr[i].size();
          std::vector<std::string> buffer2(M);
          for (size_t j = 0; j < M; j++) {
            size_t K = arr[i][j].size();
            std::vector<std::string> buffer3(K);
            for (size_t k = 0; k < K; k++) {
              double d = arr[i][j][k];
              buffer3[k] = std::visit(vt, var_t{d});
            }
            buffer2[j] = "[" + join(buffer3, ", ") + "]";
          }

          buffer1[i] = "[" + join(buffer2, ", ") + "]";
        }

        return "[" + join(buffer1, "\n") + "]";
      }
    };

    struct query_to_string {
      std::string operator()(const query_t& q) { 
        return std::visit(query_t_to_string(), q); 
      }

      std::string operator()(const std::vector<query_t>& results) { 
        std::vector<std::string> buffer;
        for (auto const& q : results) {
          buffer.push_back(std::visit(query_t_to_string(), q));
        }

        return "[" + join(buffer, ", ") + "]";
      }
    };

    struct make_query_t_unique {
      var_t_eq var_visitor;

      make_query_t_unique(double atol=ATOL, double rtol=RTOL) {
        var_visitor = var_t_eq{atol, rtol};
      }

      query_t operator()(const var_t& v) const { 
        return std::vector<var_t>{v}; 
      }

      query_t operator()(const std::vector<var_t>& vec) const {
        std::vector<var_t> return_vals;

        for (auto const &tar_val : vec) {
          auto result = std::find_if(return_vals.begin(), return_vals.end(), [tar_val, &var_visitor=var_visitor](const var_t& val) {
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

    static std::string params_to_string(const Params& params, uint32_t indentation=0) {
      std::string s = "";

      for (uint32_t i = 0; i < indentation; i++) {
        s += "\t";
      }

      std::vector<std::string> buffer;

      for (auto const &[key, field] : params) {
        buffer.push_back("\"" + key + "\": " + std::visit(var_t_to_string(), field));
      }

      std::string delim = ",\n";
      for (uint32_t i = 0; i < indentation; i++) {
        delim += "\t";
      }

      s += join(buffer, delim);

      return s;
    }

    template <class T>
    T get(Params &params, const std::string& key, T defaultv) {
      if (params.count(key)) {
        return std::get<T>(params[key]);
      }

      params[key] = var_t{defaultv};
      return defaultv;
    }

    template <class T>
    T get(Params &params, const std::string& key) {
      if (!params.count(key)) {
        std::string error_message = "Key \"" + key + "\" not found in Params.";
        throw std::invalid_argument(error_message);
      }
      return std::get<T>(params[key]);
    }

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

    static std::vector<Params> parse_config(nlohmann::json data, Params p, bool verbose) {
      if (verbose) {
        std::cout << "Loaded: \n" << data.dump() << "\n";
      }

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
          for (auto const &[k, v] : zparams[i]) {
            p[k] = v;
          }
          std::vector<Params> new_params = parse_config(data, Params(p), false);
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

      for (auto key : scalars) {
        data.erase(key);
      }

      if (!contains_vector) {
        params.push_back(p);
      } else {
        auto vals = data[vector_key];
        data.erase(vector_key);
        for (auto v : vals) {
          p[vector_key] = parse_json_type(v);

          std::vector<Params> new_params = parse_config(data, p, false);
          params.insert(params.end(), new_params.begin(), new_params.end());
        }
      }

      return params;

    }

    static std::vector<Params> parse_config(nlohmann::json data, bool verbose=false) {
      return parse_config(data, Params(), verbose);
    }

    static std::vector<Params> parse_config(const std::string& s, bool verbose=false) {
      std::string t(s);
      std::replace(t.begin(), t.end(), '\'', '"');
      return parse_config(nlohmann::json::parse(t), verbose);
    }

    static std::string paramset_to_string(const std::vector<Params>& params) {
      std::set<std::string> keys;
      std::map<std::string, std::set<var_t>> vals;
      for (auto const &p : params) {
        for (auto const &[k, v] : p) {
          keys.insert(k);
          if (!vals.count(k)) {
            vals[k] = std::set<var_t>();
          }

          vals[k].insert(v);
        }
      }

      std::string s = "{\n";
      std::vector<std::string> buffer1;
      for (auto const &key : keys) {
        std::string b1 = "\t\"" + key + "\": ";

        if (vals[key].size() > 1) {
          b1 += "[";
        }

        std::vector<std::string> buffer2;
        std::vector<var_t> sorted_vals(vals[key].begin(), vals[key].end());
        std::sort(sorted_vals.begin(), sorted_vals.end());

        for (auto val : sorted_vals) {
          buffer2.push_back(std::visit(var_t_to_string(), val));
        }

        b1 += join(buffer2, ", ");

        if (vals[key].size() > 1) {
          b1 += "]";
        }

        buffer1.push_back(b1);
      }

      s += join(buffer1, ",\n");

      s += "\n}";

      return s;
    }

  }
}
