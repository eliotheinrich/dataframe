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
      std::map<std::string, std::vector<std::vector<double>>> samples;

      DataSlide() {}

      DataSlide(Params &params) : params(params) {}

      DataSlide(const std::string &s);

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

        for (auto const& [key, vals] : other.samples) {
          add_samples(key);
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
        return params.contains(s) || data.contains(s) || samples.contains(s);
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

      void add_samples(const std::string& s) {
        samples.emplace(s, std::vector<std::vector<double>>());
      }

      void push_data(const std::string& s, const std::vector<Sample>& sample_vec) {
        data[s].push_back(sample_vec);
      }
      
      void push_data(const std::string& s, const std::vector<double>& double_vec) {
        if (data.contains(s)) {
          std::vector<Sample> sample_vec(double_vec.size());
          for (uint32_t i = 0; i < sample_vec.size(); i++) {
            sample_vec[i] = Sample(sample_vec[i]);
          }
          push_data(s, sample_vec);
        } else {
          samples[s].push_back(double_vec);
        }
      }

      void push_data(const std::string& s, const Sample& sample) {
        push_data(s, {sample});
      }

      void push_data(const std::string &s, const double d) {
        if (data.contains(s)) {
          Sample sample(d);
          push_data(s, sample);
        } else {
          samples[s].push_back({d});
        }
      }

      void push_data(const std::string &s, const double mean, const double std, const uint32_t num_samples) {
        Sample sample(mean, std, num_samples);
        push_data(s, sample);
      }

      std::vector<std::vector<double>> get_data(const std::string& s) const {
        if (data.contains(s)) {
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
        } else if (samples.contains(s)) {
          return samples.at(s);
        } else {
          return std::vector<std::vector<double>>();
        }
      }

      std::vector<std::vector<double>> get_std(const std::string& s) const {
        if (!data.contains(s)) {
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
        if (params.contains(s)) { 
          return params.erase(s);
        } else if (data.contains(s)) {
          return data.erase(s);
        } else if (samples.contains(s)) {
          return samples.erase(s);
        }
        return false;
      }

      std::string to_string() const;

      bool congruent(const DataSlide &ds, const utils::var_t_eq& equality_comparator) {
        auto incongruent_key = first_incongruent_key(ds, equality_comparator);
        return incongruent_key == std::nullopt;
      }
      
      std::optional<std::string> first_incongruent_key(const DataSlide &other, const utils::var_t_eq& equality_comparator) const {
        for (auto const& [key, val] : params) {
          if (!other.params.contains(key) || !equality_comparator(other.params.at(key), val)) {
            return key;
          }
        }
        
        for (auto const& [key, val] : other.params) {
          if (!params.contains(key) || !equality_comparator(params.at(key), val)) {
            return key;            
          }
        }
        
        
        for (auto const &[key, _] : data) {
          if (!other.data.contains(key)) {
            return key;
          }
          if (other.data.at(key).size() != data.at(key).size()) {
            return key;
          }
        }
        for (auto const &[key, _] : other.data) {
          if (!data.contains(key)) {
            return key;
          }
        }

        for (auto const &[key, _] : samples) {
          if (!other.samples.contains(key)) {
            return key;
          }
        }
        for (auto const &[key, _] : other.samples) {
          if (!samples.contains(key)) {
            return key;
          }
        }
        
        return std::nullopt;
      }

      DataSlide combine(const DataSlide &other, double atol=DF_ATOL, double rtol=DF_RTOL) {
        utils::var_t_eq equality_comparator(atol, rtol);
        auto key = first_incongruent_key(other, equality_comparator);

        if (!(key == std::nullopt)) {
          std::stringstream ss;
          ss << "DataSlides not congruent at key \"" << key.value() << "\".\n"; 
          ss << to_string() << "\n\n\n" << other.to_string() << std::endl;
          std::string error_message = ss.str();
          throw std::invalid_argument(error_message);
        }
        
        DataSlide dn(params); 

        for (auto const &[key, val] : data) {
          dn.add_data(key);
          for (uint32_t i = 0; i < val.size(); i++) {
            size_t s1 = val[i].size();
            size_t s2 = other.data.at(key)[i].size();
            if (s1 != s2) {
              std::string error_message = "Samples with key '" + key + "' have incongruent length ("
                                        + std::to_string(s1) + " and " + std::to_string(s2) + ")"
                                        + " and cannot be combined.";
              throw std::invalid_argument(error_message);
            }

            dn.push_data(key, Sample::combine_samples(val[i], other.data.at(key)[i]));
          }
        }

        for (auto const &[key, v] : samples) {
          dn.add_data(key);
          for (size_t i = 0; i < v.size(); i++) {
            dn.push_data(key, v[i]);
          }
          for (size_t i = 0; i < other.samples.at(key).size(); i++) {
            dn.push_data(key, other.samples.at(key)[i]);
          }
        }

        return dn;
      }

    private:
      static DataSlide deserialize(const std::string& s);
  };

  template <>
  inline void DataSlide::add_param(const std::string& s, const int t) { 
    params[s] = static_cast<double>(t); 
  }

}
