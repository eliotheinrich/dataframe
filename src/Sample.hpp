#pragma once

#include "utils.hpp"

#include <vector>
#include <string>
#include <array>

namespace dataframe {
  class Sample {
    public:
      Sample() : Sample(0.0, 0.0, 0) {}
      Sample(double mean) : Sample(mean, 0.0, 1) {}

      Sample(double mean, double std, uint32_t num_samples) {
        set_mean(mean);
        set_std(std);
        set_num_samples(num_samples);
        if (isnan()) {
          throw std::invalid_argument("Attempted to create a Sample containing NaN.");
        }
      }

      Sample(const std::vector<double>& doubles) : Sample() {
        size_t num_samples = doubles.size();
        double mean = 0.0;
        for (size_t i = 0; i < num_samples; i++) {
          mean += doubles[i];
        }
        mean = mean/num_samples;

        double std = 0.0;
        for (size_t i = 0; i < num_samples; i++) {
          std += std::pow(doubles[i] - mean, 2);
        }
        std = std::sqrt(std/num_samples);

        set_mean(mean);
        set_std(std);
        set_num_samples(num_samples);
      }

      bool isnan() const {
        return std::isnan(get_mean()) || std::isnan(get_std());
      }

      Sample(const std::string &s) {
        // Deprecated json deserialization
        
        if (!Sample::is_valid(s)) {
          std::string error_message = "Invalid string \"" + s + "\" provided to Sample(std::string).";
          throw std::invalid_argument(error_message);
        }

        if (s.front() == '[' && s.back() == ']') {
          std::string trimmed = s.substr(1, s.length() - 2);

          std::vector<std::string> elements = utils::split(trimmed, ",");

          set_mean(std::stof(elements[0]));
          set_std(std::stof(elements[1]));
          set_num_samples(std::stoi(elements[2]));
        } else {
          set_mean(std::stof(s));
          set_std(0.);
          set_num_samples(1);
        }

        if (isnan()) {
          throw std::invalid_argument("Attempted to create a Sample containing NaN.");
        }
      }

      static bool is_valid(const std::string& s) {
        if (s.front() == '[' && s.back() == ']') {
          std::string trimmed = s.substr(1, s.length() - 2);
          std::vector<std::string> elements = utils::split(trimmed, ",");

          return elements.size() == 3 && utils::is_float(elements[0]) && utils::is_float(elements[1]) & utils::is_integer(elements[2]);
        } else {
          return utils::is_float(s);
        }
      }

      inline double get_mean() const {
        return data[0];
      }

      inline void set_mean(double mean) {
        data[0] = mean;
      }

      inline double get_std() const {
        return data[1];
      }

      inline void set_std(double std) {
        data[1] = std;
      }

      inline uint32_t get_num_samples() const {
        return static_cast<uint32_t>(data[2]);
      }

      inline void set_num_samples(uint32_t num_samples) {
        data[2] = static_cast<double>(num_samples);
      }

      Sample combine(const Sample &other) const {
        uint32_t combined_samples = get_num_samples() + other.get_num_samples();
        if (combined_samples == 0) {
          return Sample();
        }

        double samples1f = get_num_samples(); 
        double samples2f = other.get_num_samples();
        double combined_samplesf = combined_samples;

        double combined_mean = (samples1f*get_mean() + samples2f*other.get_mean())/combined_samplesf;
        double combined_std = std::pow((samples1f*(std::pow(get_std(), 2) + std::pow(get_mean() - combined_mean, 2))
              + samples2f*(std::pow(other.get_std(), 2) + std::pow(other.get_mean() - combined_mean, 2))
              )/combined_samplesf, 0.5);

        return Sample(combined_mean, combined_std, combined_samples);
      }

      std::string to_string(bool full_sample = false) const {
        if (full_sample) {
          std::string s = "[";
          s += std::to_string(get_mean()) + ", " + std::to_string(get_std()) + ", " + std::to_string(get_num_samples()) + "]";
          return s;
        } else {
          return std::to_string(get_mean());
        }
      }

      static std::vector<Sample> combine_samples(const std::vector<Sample>& samples1, const std::vector<Sample>& samples2) {
        if (samples1.size() != samples2.size()) {
          throw std::invalid_argument("Cannot combine samples; incongruent lentgh.");
        }

        std::vector<Sample> samples(samples1.size());
        for (uint32_t i = 0; i < samples1.size(); i++) {
          samples[i] = samples1[i].combine(samples2[i]);
        }

        return samples;
      }

      static Sample collapse_samples(const std::vector<Sample>& samples) {
        size_t N = samples.size();
        if (N == 0) {
          return Sample();
        }

        Sample s = samples[0];
        for (uint32_t i = 1; i < N; i++) {
          s = s.combine(samples[i]);
        }

        return s;
      }

      static std::vector<Sample> collapse_samples(const std::vector<std::vector<Sample>>& samples) {
        size_t N = samples.size();
        if (N == 0) {
          return std::vector<Sample>();
        }

        size_t M = samples[0].size();

        std::vector<Sample> collapsed_samples(M);
        for (uint32_t i = 0; i < M; i++) {
          Sample s = samples[0][i];
          for (uint32_t j = 1; j < N; j++) {
            s = s.combine(samples[j][i]);
          }

          collapsed_samples[i] = s;
        }

        return collapsed_samples;
      }

      std::array<double, 3> data;
      
      struct glaze {
        static constexpr auto value{&Sample::data};
      };
  };


  // Thin wrapper for map; desirable to overload emplace so that Sample does not need to be exposed
  class data_t {
    private:
      std::map<std::string, std::vector<Sample>> data;

    public:
      data_t()=default;

      void emplace(const std::string &key, const std::vector<Sample> &samples) {
        data.emplace(key, samples);
      }

      void emplace(const std::string &key, const std::vector<double> &doubles) {
        size_t N = doubles.size();
        std::vector<Sample> samples(N);
        for (uint32_t i = 0; i < N; i++) {
          samples[i] = Sample(doubles[i]);
        }

        emplace(key, samples);
      }

      void emplace(const std::string &key, double d) {
        std::vector<Sample> sample{Sample(d)};
        emplace(key, sample);
      }

      void emplace(const std::string &key, Sample s) {
        std::vector<Sample> sample{s};
        emplace(key, sample);
      }
      
      void clear() {
        data.clear();
      }

      void swap(data_t& other) {
        data.swap(other.data);
      }

      auto find(const std::string& key) const {
        return data.find(key);
      }

      auto begin() const {
        return data.begin();
      }

      auto end() const {
        return data.end();
      }

      int count(const std::string &key) const {
        return data.count(key);
      }

      size_t size() const {
        return data.size();
      }

      bool empty() const {
        return data.empty();
      }

      std::vector<Sample> at(const std::string& key) const {
        return data.at(key);
      }

      const std::vector<Sample>& operator[](const std::string& key) const {
        return data.at(key);
      }

      std::vector<Sample>& operator[](const std::string& key) {
        return data[key];
      }

      data_t& operator=(const data_t& other) {
        data = other.data;
        return *this;
      }
  };
}
