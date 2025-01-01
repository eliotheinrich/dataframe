#pragma once

#include <fmt/format.h>

#include "utils.hpp"
#include "Sample.hpp"

#include <stdexcept>

namespace dataframe {
  class DataFrame;

  class DataSlide {
    public:
      friend DataFrame;

      ExperimentParams params;

      // data and samples are stored in the format rows x length, where
      // rows correspond to sampled properties corresponding to key
      // i.e. if a vector v = (vx, vy, vz) is sampled, the data is stored as
      // vx1 vx2 vx3 ...
      // vy1 vy2 vy3 ...
      // vz1 vz2 vz3 ...
      std::map<std::string, std::vector<std::vector<Sample>>> data;
      std::map<std::string, std::vector<std::vector<double>>> samples;

      // For storing extraneous data, e.g. serialized simulator states associated with this slide
      std::vector<byte_t> buffer;

      DataSlide() {}

      DataSlide(ExperimentParams &params) : params(params) {}

      DataSlide(const std::string &s);

      DataSlide(const DataSlide& other) : buffer(other.buffer) {
        for (auto const& [key, val]: other.params) {
          add_param(key, val);
        }

        for (auto const& [key, vals] : other.data) {
          add_data(key, vals.size());
          data[key] = vals;
        }

        for (auto const& [key, vals] : other.samples) {
          add_samples(key, vals.size());
          samples[key] = vals;
        }
      }

      DataSlide(const std::vector<byte_t>& bytes);

      ~DataSlide()=default;

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

      Parameter get_param(const std::string& key) const {
        return params.at(key);
      }

      template <typename T>
      void add_param(const std::string& key, const T val) { 
        params[key] = val; 
      }

      void add_param(const ExperimentParams &params) {
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
        if (!data.contains(key)) {
          throw std::runtime_error(fmt::format("Data with key {} does not exist.", key));
        }
        size_t width = data[key].size();
        if (sample.size() != width) {
          throw std::runtime_error(
            fmt::format(
              "Error pushing sample at key {}; data[{}] has width {} but provided sample has width {}",
              key, key, width, sample.size()
            )
          );
        }

        std::vector<Sample> sample_vec(width);
        for (size_t i = 0; i < width; i++) {
          sample_vec[i] = Sample(sample[i]);
        }

        if (avg) {
          for (size_t i = 0; i < width; i++) {
            size_t num_samples = data[key][i].size();
            if (num_samples == 0) { // Data is empty; emplace
              data[key][i].push_back(sample_vec[i]);
            } else if (num_samples == 1) { // Average with (single) existing sample
              Sample s1 = data[key][i][0];
              Sample s2 = sample_vec[i];
              data[key][i][0] = s1.combine(s2);
            } else { // Otherwise, throw an error
              throw std::runtime_error(
                fmt::format(
                  "data[{}][{}] has width {}; cannot perform average.",
                  key, i, data[key][i].size()
                )
              );
            }
          }
        } else {
          push_samples_to_data(key, sample_vec);
        }
      }

      void push_samples_to_data(const std::string& key, const std::vector<std::vector<Sample>>& sample) {
        if (!data.contains(key)) {
          throw std::runtime_error(fmt::format("Data with key {} does not exist.", key));
        }
        size_t width = data[key].size();
        if (sample.size() != width) {
          throw std::runtime_error(
            fmt::format(
              "Error pushing sample at key {}; data[{}] has width {} but provided sample has width {}.",
              key, key, width, sample.size()
            )
          );
        }

        for (size_t i = 0; i < width; i++) {
          size_t num_samples = data[key][i].size();
          if (num_samples != sample[i].size()) {
            throw std::runtime_error(
              fmt::format(
                "Error pushing sample at key {}; data[{}][{}] has length {} but provided sample[{}] has length {}.",
                key, key, i, num_samples, i, sample.size()
              )
            );
          }
          for (size_t j = 0; j < num_samples; j++) {
            data[key][i][j] = data[key][i][j].combine(sample[i][j]);
          }
        }
      }

      void push_samples_to_data(const std::string& key, const std::vector<Sample>& sample_vec) {
        if (!data.contains(key)) {
          throw std::runtime_error(fmt::format("Data with key {} does not exist.", key));
        }
        size_t width = data[key].size();
        if (sample_vec.size() != width) {
          throw std::runtime_error(
            fmt::format(
              "Error pushing sample at key {}; data[{}] has width {} but provided sample has width {}.",
              key, key, width, sample_vec.size()
            )
          );
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
        if (!samples.contains(key)) {
          throw std::runtime_error(fmt::format("Samples with key {} does not exist.", key));
        }
        size_t width = samples[key].size();
        if (sample.size() != width) {
          throw std::runtime_error(
            fmt::format(
              "Error pushing sample at key {}; samples[{}] has width {} but provided sample has width {}.",
              key, key, width, sample.size()
            )
          );
        }

        for (size_t i = 0; i < width; i++) {
          samples[key][i].insert(samples[key][i].end(), sample[i].begin(), sample[i].end());
        }
      }

      void push_samples(const std::string& key, const std::vector<double>& double_vec) {
        if (!samples.contains(key)) {
          throw std::runtime_error(fmt::format("Samples with key {} does not exist.", key));
        }

        for (size_t i = 0; i < samples[key].size(); i++) {
          samples[key][i].push_back(double_vec[i]);
        }
      }

      void push_samples(const std::string &key, const double d) {
        if (!samples.contains(key)) {
          throw std::runtime_error(fmt::format("Samples with key {} does not exist.", key));
        }
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
              throw std::runtime_error("Stored data is not square.");
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
            throw std::runtime_error("Stored data is not square.");
          }

          for (uint32_t j = 0; j < length; j++) {
            d[i][j] = di[j].get_std();
          }
        }

        return d;
      }

      std::vector<std::vector<double>> get_num_samples(const std::string& key) const {
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
            throw std::runtime_error("Stored data is not square.");
          }

          for (uint32_t j = 0; j < length; j++) {
            d[i][j] = di[j].get_num_samples();
          }
        }

        return d;
      }

      bool remove_param(const std::string& key) {
        return params.erase(key);
      }

      bool remove_samples(const std::string& key) {
        return samples.erase(key);
      }

      bool remove_data(const std::string& key) {
        return data.erase(key);
      }

      bool remove(const std::string& key) {
        if (params.contains(key)) { 
          return remove_param(key);
        } else if (data.contains(key)) {
          return remove_data(key);
        } else { 
          return remove_samples(key);
        }
      }

      std::vector<byte_t> to_bytes() const;

      std::string to_json() const;

      std::string describe() const;

      bool congruent(const DataSlide &ds, const utils::param_eq& equality_comparator) {
        auto incongruent_key = first_incongruent_key(ds, equality_comparator);
        return incongruent_key == std::nullopt;
      }
      
      std::optional<std::string> first_incongruent_key(const DataSlide &other, const utils::param_eq& equality_comparator) const {
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

      void average_samples_inplace() {
        for (auto const &[key, val] : data) {
          data[key] = Sample::collapse_samples(val);
        }

        std::vector<std::string> to_delete;
        for (auto const &[key, val] : samples) {
          add_data(key, val.size());
          push_samples_to_data(key, val, true);
          to_delete.push_back(key);
        }


        for (auto const& key : to_delete) {
          remove_samples(key);
        }
      }

      DataSlide average_samples() const {
        DataSlide copy(*this);
        copy.average_samples_inplace();
        return copy;
      }

      void combine_data(const DataSlide& other) {
        for (auto const &[key, val] : other.data) {
          size_t width1 = val.size();
          size_t width2 = data.at(key).size();
          if (width1 != width2) {
            throw std::runtime_error(
              fmt::format(
                "Data with key {} have incongruent width ({} and {}) and connot be combined.",
                key, width1, width2
              )
            );
          }

          push_samples_to_data(key, val);
        }

        for (auto const &[key, val] : other.samples) {
          size_t width1 = val.size();
          size_t width2 = samples.at(key).size();
          if (width1 != width2) {
            throw std::runtime_error(
              fmt::format(
                "Samples with key {} have incongruent width ({} and {}) and connot be combined.",
                key, width1, width2
              )
            );
          }

          push_samples(key, val);
        }
      }

      DataSlide combine(const DataSlide &other, double atol=DF_ATOL, double rtol=DF_RTOL) {
        utils::param_eq equality_comparator(atol, rtol);
        auto key = first_incongruent_key(other, equality_comparator);

        if (key != std::nullopt) {
          throw std::runtime_error(
            fmt::format(
              "DataSlides not congruent at key \"{}\".\n{}\n\n\n{}\n",
              key.value(), to_json(), other.to_json()
            )
          );

        }
        
        DataSlide slide(*this); 
        slide.combine_data(other);

        return slide;
      }
  };
}
