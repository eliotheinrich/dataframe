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

      std::map<std::string, DataObject> data;

      // For storing extraneous data, e.g. serialized simulator states associated with this slide
      std::vector<byte_t> buffer;

      DataSlide()=default;
      ~DataSlide()=default;

      DataSlide(ExperimentParams &params) : params(params) {}

      DataSlide(const DataSlide&)=default;
      DataSlide& operator=(const DataSlide&)=default;

      DataSlide(DataSlide&&) noexcept=default;
      DataSlide& operator=(DataSlide&&) noexcept=default;

      DataSlide(const std::vector<byte_t>& bytes);

      static DataSlide copy_params(const DataSlide& other) {
        DataSlide slide;
        for (auto const& [key, val]: other.params) {
          slide.add_param(key, val);
        }

        return slide;
      }

      bool contains(const std::string& key) const {
        return params.contains(key) || data.contains(key);
      }

      Parameter get_param(const std::string& key) const {
        return params.at(key);
      }

      template <typename T>
      void add_param(const std::string& key, const T val) { 
        if (contains(key)) {
          throw std::runtime_error(fmt::format("Tried to add parameter with key {}, but this slide already contains a parameter or data by that key.", key));
        }

        params[key] = val; 
      }

      void add_param(const ExperimentParams &params) {
        for (auto const &[key, field] : params) {
          add_param(key, field);
        }
      }

      std::vector<size_t> get_shape(const std::string& key) const {
        const auto& [shape, values, sampling_data] = data.at(key);
        return shape;
      }

      size_t get_size(const std::string& key) const {
        std::vector<size_t> shape = get_shape(key);
        return dataframe::utils::shape_size(shape);
      }

      static bool shapes_equal(const std::vector<size_t>& s1, const std::vector<size_t>& s2) {
        if (s1.size() != s2.size()) {
          return false;
        }

        for (size_t i = 0; i < s1.size(); i++) {
          if (s1[i] != s2[i]) {
            return false;
          }
        }

        return true;
      }

      void add_data(const std::string& key, const std::vector<double>& values, std::optional<std::vector<size_t>> shape_opt=std::nullopt, std::optional<SamplingData> sampling_data_opt=std::nullopt) {
        if (params.contains(key)) {
          throw std::runtime_error(fmt::format("Tried to add data with key {}, but this slide already contains a parameter with that key.", key));
        }

        if (shape_opt) {
          std::vector<size_t> shape = shape_opt.value();
          if (dataframe::utils::shape_size(shape) != values.size()) {
            throw std::runtime_error(fmt::format("Shape {} provided for data with size {}.", shape, values.size()));
          }
        }

        if (data.contains(key)) {
          const auto& [existing_shape, existing_values, existing_sampling_data] = data.at(key);
          if (shape_opt) {
            if (!shapes_equal(existing_shape, shape_opt.value())) {
              throw std::runtime_error("Mismatched shape in add_data.");
            }
          }

          auto [new_values, new_error, new_nsamples] = combine_values(values, existing_values, sampling_data_opt, existing_sampling_data);
          data[key] = {existing_shape, new_values, SamplingData{new_error, new_nsamples}};
        } else {
          std::vector<size_t> shape;
          if (shape_opt) {
            shape = shape_opt.value();
          } else {
            shape = {values.size()};
          }

          DataObject obj = {shape, values, std::nullopt};
          data.emplace(key, obj);
        }
      }

      // TODO avoid copying
      void add_data(const std::string& key, const DataObject& data) {
        add_data(key, data.values, data.shape, data.sampling_data);
      }

      void add_data(const SampleMap& samples) {
        for (const auto &[key, data] : samples) {
          add_data(key, data.values, data.shape, data.sampling_data);
        }
      }
  

      void concat_data(const std::string& key, const std::vector<double>& values, const std::vector<size_t>& shape) {
        if (params.contains(key)) {
          throw std::runtime_error(fmt::format("Tried to add data with key {}, but this slide already contains a parameter with that key.", key));
        }

        size_t data_size = dataframe::utils::shape_size(shape);
        size_t num_new_samples = values.size() / data_size;
        
        if (num_new_samples * data_size != values.size()) {
          throw std::runtime_error("Passed data of invalid shape to concat_data.");
        }


        if (data.contains(key)) {
          auto& [existing_shape, existing_values, sampling_data] = data.at(key);

          std::vector<size_t> real_existing_shape;
          size_t num_existing_samples;
          if (existing_shape.size() == shape.size()) {
            real_existing_shape = existing_shape;
            num_existing_samples = 1;
          } else if (existing_shape.size() == shape.size() + 1) {
            real_existing_shape = std::vector<size_t>(existing_shape.begin() + 1, existing_shape.end());
            num_existing_samples = existing_shape[0];
          }

          if (!shapes_equal(real_existing_shape, shape)) {
            throw std::runtime_error("Passed data of invalid shape to concat_data.");
          }

          existing_values.insert(existing_values.end(), values.begin(), values.end());

          std::vector<size_t> new_shape = shape;
          new_shape.insert(new_shape.begin(), num_new_samples + num_existing_samples);
          existing_shape = new_shape;

          if (sampling_data) {
            auto& [error, nsamples] = sampling_data.value();
            std::vector<double> zeros(num_new_samples * data_size, 0.0);
            error.insert(error.end(), zeros.begin(), zeros.end());

            std::vector<size_t> ones(num_new_samples * data_size, 1);
            nsamples.insert(nsamples.end(), ones.begin(), ones.end());
          }
        } else {
          std::vector<size_t> new_shape = shape;
          new_shape.insert(new_shape.begin(), num_new_samples);
          add_data(key, values, new_shape);
        }
      }

      static std::tuple<std::vector<double>, std::vector<double>, std::vector<size_t>> combine_values(
          const std::vector<double>& values1, const std::vector<double>& values2, 
          std::optional<SamplingData> sampling_data1, std::optional<SamplingData> sampling_data2
      ) {
        size_t N = values1.size();
        if (values2.size() != N) {
          throw std::runtime_error("Mismatched values size.");
        }

        std::vector<double> error1;
        std::vector<size_t> nsamples1;
        if (sampling_data1) {
          error1 = sampling_data1->std;
          nsamples1 = sampling_data1->nsamples;
        } else {
          error1 = std::vector<double>(N, 0.0);
          nsamples1 = std::vector<size_t>(N, 1);
        }

        std::vector<double> error2;
        std::vector<size_t> nsamples2;
        if (sampling_data2) {
          error2 = sampling_data2->std;
          nsamples2 = sampling_data2->nsamples;
        } else {
          error2 = std::vector<double>(N, 0.0);
          nsamples2 = std::vector<size_t>(N, 1);
        }

        std::vector<double> new_values(N);
        std::vector<double> new_std(N);
        std::vector<size_t> new_nsamples(N);

        for (size_t i = 0; i < N; i++) {
          size_t N1 = nsamples1[i];
          size_t N2 = nsamples2[i];
          size_t total_num_samples = N1 + N2;

          double v = (values1[i]*nsamples1[i] + values2[i]*nsamples2[i]) / total_num_samples;
          double s1 = std::pow(error1[i], 2);
          double s2 = std::pow(error2[i], 2);
          double s = std::pow(((N1 - 1) * s1 + (N2 - 1) * s2 + double(N1*N2)/(N1 + N2) * std::pow(values1[i] - values2[i], 2))/(total_num_samples - 1), 0.5);

          new_values[i] = v;
          new_std[i] = s;
          new_nsamples[i] = total_num_samples;
        }

        return {new_values, new_std, new_nsamples};
      }

      void combine(const DataSlide &other, double atol=DF_ATOL, double rtol=DF_RTOL) {
        utils::param_eq equality_comparator(atol, rtol);
        auto key = first_incongruent_key(other, equality_comparator);

        if (key != std::nullopt) {
          throw std::runtime_error(fmt::format("DataSlides not congruent at key \"{}\"", key.value()));
        }

        // Slides are now guaranteed to be congruent
        for (auto const& [key, val] : other.data) {
          auto const& [shape1, values1, sampling_data1] = data.at(key);
          auto const& [shape2, values2, sampling_data2] = val;
          auto [values, error, num_samples] = combine_values(values1, values2, sampling_data1, sampling_data2);
          DataObject obj = {shape1, values, SamplingData{error, num_samples}};
          data[key] = obj;
        }
      }

      std::vector<double> get_data(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, sampling_data] = data.at(key);
          return values;
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      std::vector<double> get_std(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, sampling_data] = data.at(key);
          if (sampling_data) {
            const auto& [error, nsamples] = sampling_data.value();
            return error;
          } else {
            return std::vector<double>(dataframe::utils::shape_size(shape), 0.0);
          }
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      std::vector<double> get_standard_error(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, sampling_data] = data.at(key);
          if (sampling_data) {
            const auto& [error, nsamples] = sampling_data.value();
            std::vector<double> sderror(error.begin(), error.end());
            for (size_t i = 0; i < error.size(); i++) {
              sderror[i] /= std::sqrt(nsamples[i]);
            }
            return sderror;
          } else {
            return std::vector<double>(dataframe::utils::shape_size(shape), 0.0);
          }
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      std::vector<size_t> get_num_samples(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, sampling_data] = data.at(key);
          if (sampling_data) {
            const auto& [error, nsamples] = sampling_data.value();
            return nsamples;
          } else {
            return std::vector<size_t>(dataframe::utils::shape_size(shape), 1);
          }
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      bool remove_param(const std::string& key) {
        return params.erase(key);
      }

      bool remove_data(const std::string& key) {
        return data.erase(key);
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

          const auto& [shape1, values1, sampling_data1] = data.at(key);
          const auto& [shape2, values2, sampling_data2] = other.data.at(key);

          // Dimension mismatch
          if (shape1.size() != shape2.size()) {
            return key;
          }

          // Shape mismatch
          for (size_t i = 0; i < shape1.size(); i++) {
            if (shape1[i] != shape2[i]) {
              return key;
            }
          }
        }

        for (auto const &[key, _] : other.data) {
          if (!data.contains(key)) {
            return key;
          }

          // TODO check if this section is redundant
          const auto& [shape1, values1, sampling_data1] = other.data.at(key);
          const auto& [shape2, values2, sampling_data2] = data.at(key);

          // Dimension mismatch
          if (shape1.size() != shape2.size()) {
            return key;
          }

          // Shape mismatch
          for (size_t i = 0; i < shape1.size(); i++) {
            if (shape1[i] != shape2[i]) {
              return key;
            }
          }
        }
        
        return std::nullopt;
      }
  };
}
