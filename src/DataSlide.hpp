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

      // data and samples are stored in the format rows x length, where
      // rows correspond to sampled properties corresponding to key
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
          add_data(key, vals.size());
          push_samples_to_data(key, vals);
        }

        for (auto const& [key, vals] : other.samples) {
          add_samples(key, vals.size());
          push_samples(key, vals);
        }
      }

      static DataSlide copy_params(const DataSlide& other) {
        DataSlide slide;
        for (auto const& [key, val]: other.params) {
          slide.add_param(key, val);
        }

        return slide;
      }

      bool contains(const std::string& key) const {
        return params.contains(key) || data.contains(key) || samples.contains(key);
      }

      var_t get_param(const std::string& key) const {
        return params.at(key);
      }

      template <typename T>
      void add_param(const std::string& key, const T val) { 
        params[key] = val; 
      }

      void add_param(const Params &params) {
        for (auto const &[key, field] : params) {
          add_param(key, field);
        }
      }

      void add_data(const std::string& key, size_t width) {
        data.emplace(key, std::vector<std::vector<Sample>>(width));
      }

      void add_data(const std::string& key) { 
        add_data(key, 1);
      }

      void add_data(const data_t& sample) {
        for (auto const& [key, vals] : sample) {
          add_data(key, vals.size());
        }
      }

      void push_samples_to_data(const data_t& sample, bool avg=false) {
        for (auto const& [key, vals] : sample) {
          push_samples_to_data(key, vals, avg);
        }
      }

      void push_samples_to_data(const std::string& key, const std::vector<std::vector<double>>& sample, bool avg=false) {
        size_t width = data[key].size();
        if (sample.size() != width) {
          std::string error_message = "(1) Error pushing sample at key " + key + "; data[" + key + "] has width "
                                    + std::to_string(width) + " but provided sample has width "
                                    + std::to_string(sample.size()) + ".";
          throw std::invalid_argument(error_message);
        }

        std::vector<Sample> sample_vec(width);
        for (size_t i = 0; i < width; i++) {
          sample_vec[i] = Sample(sample[i]);
        }

        if (avg) {
          for (size_t i = 0; i < width; i++) {
            if (data[key][i].size() != 1) {
              std::string error_message = "data[" + key + "][" + std::to_string(i) + "] has width "
                                        + std::to_string(data[key][i].size()) + "; cannot perform average.";
              throw std::invalid_argument(error_message);
            }

            Sample s1 = data[key][i][0];
            Sample s2 = sample_vec[i];
            data[key][i][0] = s1.combine(s2);
          }
        } else {
          push_samples_to_data(key, sample_vec);
        }
      }

      void push_samples_to_data(const std::string& key, const std::vector<std::vector<Sample>>& sample) {
        size_t width = data[key].size();
        if (sample.size() != width) {
          std::string error_message = "Error pushing sample at key " + key + "; data[" + key + "] has width "
                                    + std::to_string(width) + " but provided sample has width "
                                    + std::to_string(sample.size()) + ".";
          throw std::invalid_argument(error_message);
        }

        for (size_t i = 0; i < width; i++) {
          data[key][i].insert(data[key][i].end(), sample[i].begin(), sample[i].end());
        }
      }

      void push_samples_to_data(const std::string& key, const std::vector<Sample>& sample_vec) {
        size_t width = data[key].size();
        if (sample_vec.size() != width) {
          std::string error_message = "(2) Error pushing sample at key " + key + "; data[" + key + "] has width "
                                    + std::to_string(width) + " but provided sample has width "
                                    + std::to_string(sample_vec.size()) + ".";
          throw std::invalid_argument(error_message);
        }

        for (size_t i = 0; i < width; i++) {
          data[key][i].push_back(sample_vec[i]);
        }
      }
      
      void push_samples_to_data(const std::string& key, const std::vector<double>& double_vec) {
          std::vector<Sample> sample_vec(double_vec.size());
          for (uint32_t i = 0; i < sample_vec.size(); i++) {
            sample_vec[i] = Sample(double_vec[i]);
          }
          push_samples_to_data(key, sample_vec);
      }

      void push_samples_to_data(const std::string& key, const Sample& sample) {
        push_samples_to_data(key, std::vector<Sample>{sample});
      }

      void push_samples_to_data(const std::string& key, const double d) {
        push_samples_to_data(key, Sample(d));
      }

      void push_samples_to_data(const std::string& key, const double mean, const double std, const uint32_t length) {
        push_samples_to_data(key, Sample(mean, std, length));
      }

      void add_samples(const std::string& key, size_t width) {
        samples.emplace(key, std::vector<std::vector<double>>(width));
      }

      void add_samples(const std::string& key) {
        add_samples(key, 1);
      }

      // data_t : num_rows x length
      void add_samples(const data_t& sample) {
        // vals = vector<vector>
        for (auto const& [key, vals] : sample) {
          add_samples(key, vals.size());
        }
      }


      void push_samples(const data_t& sample) {
        for (auto const& [key, vals] : sample) {
          push_samples(key, vals);
        }
      }

      void push_samples(const std::string& key, const std::vector<std::vector<double>>& sample) {
        size_t width = samples[key].size();
        if (sample.size() != width) {
          std::string error_message = "Error pushing sample at key " + key + "; samples[" + key + "] has width "
                                    + std::to_string(width) + " but provided sample has width "
                                    + std::to_string(sample.size()) + ".";
          throw std::invalid_argument(error_message);
        }

        for (size_t i = 0; i < width; i++) {
          samples[key][i].insert(samples[key][i].end(), sample[i].begin(), sample[i].end());
        }
      }

      void push_samples(const std::string& key, const std::vector<double>& double_vec) {
        samples[key].push_back(double_vec);
      }

      void push_samples(const std::string &key, const double d) {
        samples[key].push_back(std::vector<double>(d));
      }

      std::vector<std::vector<double>> get_data(const std::string& key) const {
        if (data.contains(key)) {
          size_t width = data.at(key).size();
          if (width == 0) {
            return std::vector<std::vector<double>>();
          }

          size_t length = data.at(key)[0].size();
          std::vector<std::vector<double>> d(width, std::vector<double>(length));

          for (uint32_t i = 0; i < width; i++) {
            std::vector<Sample> di = data.at(key)[i];
            if (di.size() != length) {
              throw std::invalid_argument("Stored data is not square.");
            }

            for (uint32_t j = 0; j < length; j++) {
              d[i][j] = di[j].get_mean();
            }
          }

          return d;
        } else if (samples.contains(key)) {
          return samples.at(key);
        } else {
          return std::vector<std::vector<double>>();
        }
      }

      std::vector<std::vector<double>> get_std(const std::string& key) const {
        if (!data.contains(key)) {
          return std::vector<std::vector<double>>();
        }

        size_t width = data.at(key).size();
        if (width == 0) {
          return std::vector<std::vector<double>>();
        }

        size_t length = data.at(key)[0].size();
        std::vector<std::vector<double>> d(width, std::vector<double>(length));

        for (uint32_t i = 0; i < width; i++) {
          std::vector<Sample> di = data.at(key)[i];
          if (di.size() != length) {
            throw std::invalid_argument("Stored data is not square.");
          }

          for (uint32_t j = 0; j < length; j++) {
            d[i][j] = di[j].get_std();
          }
        }

        return d;
      }

      bool remove(const std::string& key) {
        if (params.contains(key)) { 
          return params.erase(key);
        } else if (data.contains(key)) {
          return data.erase(key);
        } else { 
          return samples.erase(key);
        }
      }

      std::string to_string() const;

      std::string describe() const;

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

        if (key != std::nullopt) {
          std::string error_message = "DataSlides not congruent at key \"" + key.value() + "\".\n"
                                    + to_string() + "\n\n\n" + other.to_string() + "\n";
          throw std::invalid_argument(error_message);
        }
        
        DataSlide dn(params); 

        for (auto const &[key, val] : data) {
          size_t width1 = val.size();
          size_t width2 = other.data.at(key).size();
          if (width1 != width2) {
            std::string error_message = "Samples with key '" + key + "' have incongruent width ("
                                      + std::to_string(width1) + " and " + std::to_string(width2) + ")"
                                      + " and cannot be combined.";
            throw std::invalid_argument(error_message);
          }

          dn.add_data(key, width1);

          std::vector<std::vector<Sample>> combined_samples(width1);
          for (uint32_t i = 0; i < width1; i++) {
            size_t length1 = val[i].size();
            size_t length2 = other.data.at(key)[i].size();
            if (length1 != length2) {
              std::string error_message = "Samples with key '" + key + "' have incongruent length ("
                                        + std::to_string(length1) + " and " + std::to_string(length2) + ")"
                                        + " and cannot be combined.";
              throw std::invalid_argument(error_message);
            }

            combined_samples[i].resize(length1);
            
            for (size_t j = 0; j < length1; j++) {
              Sample s1 = val[i][j];
              Sample s2 = other.data.at(key)[i][j];
              combined_samples[i][j] = s1.combine(s2);
            }
          }

          dn.push_samples_to_data(key, combined_samples);
        }

        for (auto const &[key, val] : samples) {
          size_t width1 = val.size();
          size_t width2 = other.samples.at(key).size();
          if (width1 != width2) {
            std::string error_message = "Samples with key '" + key + "' have incongruent width ("
                                      + std::to_string(width1) + " and " + std::to_string(width2) + ")"
                                      + " and cannot be combined.";
            throw std::invalid_argument(error_message);
          }

          dn.add_samples(key, width1);
          dn.push_samples(key, val);
          dn.push_samples(key, other.samples.at(key));
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
