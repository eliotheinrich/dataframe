#pragma once

#include "utils.hpp"
#include "Sample.hpp"
#include <stdexcept>

namespace dataframe {
  class DataFrame;

  class DataSlide {
    public:
      friend DataFrame;

      Params params;
      std::map<std::string, std::vector<std::vector<Sample>>> data;

      struct glaze {
        static constexpr auto value = glz::object(
          "params", &DataSlide::params,
          "data", &DataSlide::data
        );
      };

      DataSlide() {}

      DataSlide(Params &params) : params(params) {}

      DataSlide(const std::string &s) {
        auto pe = glz::read_json(*this, s);
        if (pe) {

          std::string error_message = "Error parsing DataSlide: \n" + glz::format_error(pe, s);
          throw std::invalid_argument(error_message);
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
      void add_param(const std::string& s, const T t) { 
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
      
      void push_data(const std::string& s, const std::vector<double>& samples) {
        std::vector<Sample> sample_vec(samples.size());
        for (uint32_t i = 0; i < samples.size(); i++) {
          sample_vec[i] = Sample(samples[i]);
        }
        push_data(s, sample_vec);
      }

      void push_data(const std::string& s, const Sample& sample) {
        std::vector<Sample> sample_vec{sample};
        push_data(s, sample_vec);
      }

      void push_data(const std::string &s, const double mean) {
        Sample sample(mean);
        push_data(s, sample);
      }

      void push_data(const std::string &s, const double mean, const double std, const uint32_t num_samples) {
        Sample sample(mean, std, num_samples);
        push_data(s, sample);
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

      std::string to_string() const {
        return glz::write_json(*this);
      }

      bool congruent(const DataSlide &ds, const utils::var_t_eq& equality_comparator) {
        auto incongruent_key = first_incongruent_key(ds, equality_comparator);
        return incongruent_key == std::nullopt;
      }
      
      std::optional<std::string> first_incongruent_key(const DataSlide &other, const utils::var_t_eq& equality_comparator) const {
        for (auto const& [key, val] : params) {
          if (!other.params.count(key) || !equality_comparator(other.params.at(key), val)) {
            return key;
          }
        }
        
        for (auto const& [key, val] : other.params) {
          if (!params.count(key) || !equality_comparator(params.at(key), val)) {
            return key;            
          }
        }
        
        
        for (auto const &[key, samples] : data) {
          if (!other.data.count(key)) {
            return key;
          }
          if (other.data.at(key).size() != data.at(key).size()) {
            return key;
          }
        }
        for (auto const &[key, val] : other.data) {
          if (!data.count(key)) {
            return key;
          }
        }
        
        return std::nullopt;
      }

      DataSlide combine(const DataSlide &ds, double atol=ATOL, double rtol=RTOL) {
        utils::var_t_eq equality_comparator(atol, rtol);
        auto key = first_incongruent_key(ds, equality_comparator);

        if (!(key == std::nullopt)) {
          std::stringstream ss;
          ss << "DataSlides not congruent at key \"" << key.value() << "\".\n"; 
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

    private:
      static DataSlide deserialize(const std::string& s) {
        // Deprecated json deserialization

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

        DataSlide slide;
        for (auto const &[key, val] : ds_json.items()) {
          if (val.type() == nlohmann::json::value_t::array) {
            slide.add_data(key);

            for (auto const &v : val) {
              std::vector<Sample> samples = Sample::read_samples(v);
              slide.push_data(key, samples);
            }
          } else {
            slide.add_param(key, utils::parse_json_type(val));
          }
        }

        return slide;
      }
  };

  template <>
  inline void DataSlide::add_param(const std::string& s, const int t) { 
    params[s] = static_cast<double>(t); 
  }

}
