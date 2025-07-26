#pragma once

#include <fmt/format.h>

#include "utils.hpp"
#include "Sample.hpp"

#include <nanobind/intrusive/counter.h>

#include <stdexcept>

namespace dataframe {
  class DataSlide {
    public:
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

      void add_data(const std::string& key, ndarray<double>& values, std::optional<ndarray<double>> error_opt=std::nullopt, std::optional<ndarray<size_t>> nsamples_opt=std::nullopt) {
        size_t N = values.size();
        std::vector<size_t> shape = dataframe::utils::get_shape(values);

        std::optional<std::vector<double>> error = std::nullopt;
        if (error_opt) {
          error = std::vector<double>(error_opt->data(), error_opt->data() + N);
        }

        std::optional<std::vector<size_t>> nsamples = std::nullopt;
        if (nsamples_opt) {
          nsamples = std::vector<size_t>(nsamples_opt->data(), nsamples_opt->data() + N);
        }

        std::vector<double> values_copy(values.data(), values.data() + N);

        add_data(key, std::make_tuple(std::move(shape), std::move(values_copy), std::move(error), std::move(nsamples)));
      }

      void add_data(const std::string& key, DataObject&& object) {
        if (params.contains(key)) {
          throw std::runtime_error(fmt::format("Tried to add data with key {}, but this slide already contains a parameter with that key.", key));
        }

        if (data.contains(key)) {
          combine_values(key, std::move(object));
        } else {
          data[key] = std::move(object);
        }
      }

      void concat_data(const std::string& key, ndarray<double>& values, std::optional<std::vector<size_t>> shape_opt=std::nullopt, std::optional<ndarray<double>> error_opt=std::nullopt, std::optional<ndarray<size_t>> nsamples_opt=std::nullopt) {
        size_t N = values.size();
        std::vector<size_t> shape = shape_opt ? shape_opt.value() : dataframe::utils::get_shape(values);

        std::optional<std::vector<double>> error = std::nullopt;
        if (error_opt) {
          error = std::vector<double>(error_opt->data(), error_opt->data() + N);
        }

        std::optional<std::vector<size_t>> nsamples = std::nullopt;
        if (nsamples_opt) {
          nsamples = std::vector<size_t>(nsamples_opt->data(), nsamples_opt->data() + N);
        }

        concat_data(key, std::make_tuple(std::move(shape), std::vector<double>(values.data(), values.data() + N), std::move(error), std::move(nsamples)));
      }

      void concat_data(const std::string& key, DataObject&& object) {
        if (params.contains(key)) {
          throw std::runtime_error(fmt::format("Tried to add data with key {}, but this slide already contains a parameter with that key.", key));
        }

        const auto& [shape, values, error_opt, nsamples_opt] = object;
        size_t data_size = dataframe::utils::shape_size(shape);
        size_t num_new_samples = values.size() / data_size;

        if (data.contains(key)) {
          if (num_new_samples * data_size != values.size()) {
            throw std::runtime_error("Passed data of invalid shape to concat_data.");
          }

          auto& [existing_shape, existing_values, existing_error_opt, existing_nsamples_opt] = data.at(key);

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
            throw std::runtime_error("Mismatched shape in add_data.");
          }

          existing_values.insert(existing_values.end(), values.begin(), values.end());

          std::vector<size_t> new_shape = shape;
          new_shape.insert(new_shape.begin(), num_new_samples + num_existing_samples);
          existing_shape = new_shape;

          // Append to sampling data (if it exists)
          // TODO pad with zeros/ones if previously sampling_data did not exist
          if (existing_error_opt && existing_nsamples_opt && error_opt && nsamples_opt) {
            existing_error_opt->insert(existing_error_opt->end(), error_opt->begin(), error_opt->end());
            existing_nsamples_opt->insert(existing_nsamples_opt->end(), nsamples_opt->begin(), nsamples_opt->end());
          }
        } else {
          std::vector<size_t> new_shape = shape;
          new_shape.insert(new_shape.begin(), num_new_samples);
          add_data(key, std::make_tuple(std::move(new_shape), std::move(values), std::move(error_opt), std::move(nsamples_opt)));
        }
      }

      void combine_values(const std::string& key, const DataObject& obj) {
        auto& [shape,  values1, error1_opt, nsamples1_opt] = data.at(key);
        const auto& [shape_, values2, error2_opt, nsamples2_opt] = obj;

        size_t N = values1.size();
        if (!shapes_equal(shape, shape_)) {
          throw std::runtime_error("Mismatched values size in combine_values.");
        }

        const std::vector<double>& error1 = error1_opt ? error1_opt.value() : std::vector<double>(N, 0.0);
        const std::vector<double>& error2 = error2_opt ? error2_opt.value() : std::vector<double>(N, 0.0);

        const std::vector<size_t>& nsamples1 = nsamples1_opt ? *nsamples1_opt : std::vector<size_t>(N, 1);
        const std::vector<size_t>& nsamples2 = nsamples2_opt ? *nsamples2_opt : std::vector<size_t>(N, 1);

        std::vector<double> new_std;
        std::vector<size_t> new_nsamples;
        new_std.reserve(N);
        new_nsamples.reserve(N);

        for (size_t i = 0; i < N; i++) {
          const size_t N1 = nsamples1[i];
          const size_t N2 = nsamples2[i];
          const size_t total_num_samples = N1 + N2;

          const double v = (values1[i]*nsamples1[i] + values2[i]*nsamples2[i]) / total_num_samples;
          const double s1 = error1[i] * error1[i];
          const double s2 = error2[i] * error2[i];
          const double diff = values1[i] - values2[i];
          const double s = ((N1 - 1) * s1 + (N2 - 1) * s2 + static_cast<double>(N1*N2)/(N1 + N2) * diff * diff)/(total_num_samples - 1);

          values1[i] = v;
          new_std.push_back(std::sqrt(s));
          new_nsamples.push_back(total_num_samples);
        }

        error1_opt = std::move(new_std);
        nsamples1_opt = std::move(new_nsamples);
      }

      void combine(const DataSlide &other, double atol=DF_ATOL, double rtol=DF_RTOL) {
        utils::param_eq equality_comparator(atol, rtol);
        auto key = first_incongruent_key(other, equality_comparator);

        if (key != std::nullopt) {
          throw std::runtime_error(fmt::format("DataSlides not congruent at key \"{}\"", key.value()));
        }

        // Slides are now guaranteed to be congruent
        for (auto const& [key, val] : other.data) {
          combine_values(key, val);
        }
      }

      ndarray<double> get_data(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, error, nsamples] = data.at(key);
          return dataframe::utils::to_ndarray(values, shape);
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      ndarray<double> get_std(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, error, nsamples] = data.at(key);
          size_t N = values.size();
          if (error) {
            return dataframe::utils::to_ndarray(error.value(), shape);
          } else {
            std::vector<double> zeros(N, 0.0);
            return dataframe::utils::to_ndarray(zeros, shape);
          }
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      ndarray<double> get_standard_error(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, error_opt, nsamples_opt] = data.at(key);
          size_t N = values.size();
          if (error_opt && nsamples_opt) {
            const std::vector<double>& error = error_opt.value();
            const std::vector<size_t>& nsamples = nsamples_opt.value();

            std::vector<double> sderror(N);
            for (size_t i = 0; i < N; i++) {
              sderror[i] = error[i] / std::sqrt(nsamples[i]);
            }
            return dataframe::utils::to_ndarray(sderror, shape);
          } else {
            std::vector<double> zeros(N, 0.0);
            return dataframe::utils::to_ndarray(zeros, shape);
          }
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      ndarray<size_t> get_num_samples(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [shape, values, error, nsamples] = data.at(key);
          size_t N = values.size();
          if (nsamples) {
            return dataframe::utils::to_ndarray(nsamples.value(), shape);
          } else {
            std::vector<size_t> ones(N, 1);
            return dataframe::utils::to_ndarray(ones, shape);
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

      std::string describe() const;

      bool congruent(const DataSlide &ds, const utils::param_eq& equality_comparator) {
        auto incongruent_key = first_incongruent_key(ds, equality_comparator);
        return incongruent_key == std::nullopt;
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

          const auto& [shape1, values1, error1, nsamples1] = data.at(key);
          const auto& [shape2, values2, error2, nsamples2] = other.data.at(key);

          if (!shapes_equal(shape1, shape2)) {
            return key;
          }
        }

        for (auto const &[key, _] : other.data) {
          if (!data.contains(key)) {
            return key;
          }

          // TODO check if this section is redundant
          const auto& [shape1, values1, error1, nsamples1] = other.data.at(key);
          const auto& [shape2, values2, error2, nsamples2] = data.at(key);

          if (!shapes_equal(shape1, shape2)) {
            return key;
          }
        }
        
        return std::nullopt;
      }
  };
}
