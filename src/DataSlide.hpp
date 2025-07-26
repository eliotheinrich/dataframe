#pragma once

#include <fmt/format.h>

#include "utils.hpp"
#include "Sample.hpp"

#include <nanobind/intrusive/counter.h>

#include <stdexcept>

namespace dataframe {
  class DataSlide : public nanobind::intrusive_base {
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

      // Copy and take ownership of a raw pointer
      template <typename T=double>
      nanobind::ndarray<nanobind::numpy, T> to_ndarray(const T* values, const std::vector<size_t>& shape) const {
        size_t k = dataframe::utils::shape_size(shape);

        T* buffer = new T[k];
        std::memcpy(buffer, values, k * sizeof(T));

        return nanobind::ndarray<nanobind::numpy, T>(buffer, shape.size(), shape.data(), nanobind::find(*this));
      }

      // Copy and take ownership of a vector
      template <typename T=double>
      nanobind::ndarray<nanobind::numpy, T> to_ndarray(const std::vector<T>& values, const std::vector<size_t>& shape) const {
        return to_ndarray(values.data(), shape);
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

      void add_data(const std::string& key, ndarray<double>& values, std::optional<ndarray<double>> error=std::nullopt, std::optional<ndarray<size_t>> nsamples=std::nullopt) {
        add_data(key, std::make_tuple(values, error, nsamples));
      }

      DataObject take_ownership(const DataObject& object) {
        const auto& [values, error_opt, nsamples_opt] = object; 

        std::vector<size_t> shape = dataframe::utils::get_shape(values);

        size_t N = values.size();

        ndarray<double> owned_values = to_ndarray(values.data(), shape);
        std::optional<ndarray<double>> owned_error = std::nullopt;
        if (error_opt) {
          owned_error = to_ndarray(error_opt->data(), shape);
        }

        std::optional<ndarray<size_t>> owned_nsamples = std::nullopt;
        if (error_opt) {
          owned_nsamples = to_ndarray(nsamples_opt->data(), shape);
        }

        return std::make_tuple(owned_values, owned_error, owned_nsamples);
      }

      void add_data(const std::string& key, const DataObject& object_) {
        DataObject object = take_ownership(object_);

        if (params.contains(key)) {
          throw std::runtime_error(fmt::format("Tried to add data with key {}, but this slide already contains a parameter with that key.", key));
        }

        if (data.contains(key)) {
          combine_values(key, object);
          //data[key] = combine_values(data.at(key), data_object);
        } else {
          data.emplace(key, object);
        }
      }

      //void concat_data(const std::string& key, ndarray<double>& values, std::optional<ndarray<double>> error_opt=std::nullopt, std::optional<ndarray<size_t>> nsamples_opt=std::nullopt) {
      //  if (params.contains(key)) {
      //    throw std::runtime_error(fmt::format("Tried to add data with key {}, but this slide already contains a parameter with that key.", key));
      //  }

      //  size_t data_size = values.size();
      //  size_t num_new_samples = values.size() / data_size;
      //  
      //  if (num_new_samples * data_size != values.size()) {
      //    throw std::runtime_error("Passed data of invalid shape to concat_data.");
      //  }

      //  if (data.contains(key)) {
      //    auto& [existing_shape, existing_values, existing_sampling_data] = data.at(key);

      //    std::vector<size_t> real_existing_shape;
      //    size_t num_existing_samples;
      //    if (existing_shape.size() == shape.size()) {
      //      real_existing_shape = existing_shape;
      //      num_existing_samples = 1;
      //    } else if (existing_shape.size() == shape.size() + 1) {
      //      real_existing_shape = std::vector<size_t>(existing_shape.begin() + 1, existing_shape.end());
      //      num_existing_samples = existing_shape[0];
      //    }

      //    if (!shapes_equal(real_existing_shape, shape)) {
      //      throw std::runtime_error("Mismatched shape in add_data.");
      //    }

      //    existing_values.insert(existing_values.end(), values.begin(), values.end());

      //    std::vector<size_t> new_shape = shape;
      //    new_shape.insert(new_shape.begin(), num_new_samples + num_existing_samples);
      //    existing_shape = new_shape;

      //    // Append to sampling data (if it exists)
      //    // TODO pad with zeros/ones if previously sampling_data did not exist
      //    if (existing_sampling_data && sampling_data_opt) {
      //      auto& [existing_error, existing_nsamples] = existing_sampling_data.value();
      //      auto& [error, nsamples] = sampling_data_opt.value();
      //      existing_error.insert(existing_error.end(), error.begin(), error.end());
      //      existing_nsamples.insert(existing_nsamples.end(), nsamples.begin(), nsamples.end());
      //    }
      //  } else {
      //    std::vector<size_t> new_shape = shape;
      //    new_shape.insert(new_shape.begin(), num_new_samples);
      //    add_data(key, values, new_shape);
      //  }
      //}

      //void concat_data(const std::string& key, const DataObject& data) {
      //  auto& [values, shape, sampling_data] = data;
      //  concat_data(key, values, shape, sampling_data);
      //}

      void combine_values(const std::string& key, const DataObject& obj) {
        auto& [values1, error1_opt, nsamples1_opt] = data.at(key);
        auto& [values2, error2_opt, nsamples2_opt] = obj;

        size_t N = values1.size();
        if (values2.size() != N) {
          throw std::runtime_error("Mismatched values size in combine_values.");
        }

        std::vector<size_t> shape = dataframe::utils::get_shape(values1);

        std::vector<double> zeros(N, 0.0);
        std::vector<size_t> ones(N, 0);

        ndarray<double> error1    = to_ndarray(zeros, shape);
        ndarray<size_t> nsamples1 = to_ndarray(ones, shape);

        if (error1_opt) {
          error1 = error1_opt.value();
        }

        if (nsamples1_opt) {
          nsamples1 = nsamples1_opt.value();
        }

        ndarray<double> error2    = to_ndarray(zeros, shape);
        ndarray<size_t> nsamples2 = to_ndarray(ones, shape);

        if (error2_opt) {
          error2 = error2_opt.value();
        }

        if (nsamples2_opt) {
          nsamples2 = nsamples2_opt.value();
        }

        ndarray<double> new_values   = to_ndarray(zeros, shape);
        ndarray<double> new_std      = to_ndarray(zeros, shape);
        ndarray<size_t> new_nsamples = to_ndarray(ones, shape);

        double* values1_ptr = values1.data();
        double* error1_ptr = error1.data();
        size_t* nsamples1_ptr = nsamples1.data();

        double* values2_ptr = values2.data();
        double* error2_ptr = error2.data();
        size_t* nsamples2_ptr = nsamples2.data();

        double* new_values_ptr = new_values.data();
        double* new_std_ptr = new_std.data();
        size_t* new_nsamples_ptr = new_nsamples.data();

        for (size_t i = 0; i < N; i++) {
          size_t N1 = nsamples1_ptr[i];
          size_t N2 = nsamples2_ptr[i];
          size_t total_num_samples = N1 + N2;

          double v = (values1_ptr[i]*nsamples1_ptr[i] + values2_ptr[i]*nsamples2_ptr[i]) / total_num_samples;
          double s1 = std::pow(error1_ptr[i], 2);
          double s2 = std::pow(error2_ptr[i], 2);
          double s = std::pow(((N1 - 1) * s1 + (N2 - 1) * s2 + double(N1*N2)/(N1 + N2) * std::pow(values1_ptr[i] - values2_ptr[i], 2))/(total_num_samples - 1), 0.5);

          new_values_ptr[i] = v;
          new_std_ptr[i] = s;
          new_nsamples_ptr[i] = total_num_samples;
        }

        data[key] = std::make_tuple(new_values, new_std, new_nsamples);
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
          //data[key] = combine_values(data.at(key), val);
        }
      }

      ndarray<double> get_data(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [values, error, nsamples] = data.at(key);
          return values;
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      ndarray<double> get_std(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [values, error, nsamples] = data.at(key);
          if (error) {
            return error.value();
          } else {
            auto shape = dataframe::utils::get_shape(values);
            std::vector<double> zeros(dataframe::utils::shape_size(shape), 0.0);
            return to_ndarray(zeros, shape);
          }
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      ndarray<double> get_standard_error(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [values, error_opt, nsamples_opt] = data.at(key);
          size_t N = values.size();
          auto shape = dataframe::utils::get_shape(values);
          if (error_opt && nsamples_opt) {
            double* error = error_opt->data();
            size_t* nsamples = nsamples_opt->data();

            std::vector<double> sderror(N);
            for (size_t i = 0; i < N; i++) {
              sderror[i] = error[i] / std::sqrt(nsamples[i]);
            }
            return to_ndarray(sderror, shape);
          } else {
            std::vector<double> zeros(dataframe::utils::shape_size(shape), 0.0);
            return to_ndarray(zeros, shape);
          }
        } else {
          throw std::runtime_error(fmt::format("Could not find data with key {}.", key));
        }
      }

      ndarray<size_t> get_num_samples(const std::string& key) const {
        if (data.contains(key)) {
          const auto& [values, error, nsamples] = data.at(key);
          if (nsamples) {
            return nsamples.value();
          } else {
            auto shape = dataframe::utils::get_shape(values);
            std::vector<size_t> ones(dataframe::utils::shape_size(shape), 1);
            return to_ndarray(ones, shape);
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

          const auto& [values1, error1, nsamples1] = data.at(key);
          const auto& [values2, error2, nsamples2] = other.data.at(key);

          auto shape1 = dataframe::utils::get_shape(values1);
          auto shape2 = dataframe::utils::get_shape(values2);

          if (!shapes_equal(shape1, shape2)) {
            return key;
          }
        }

        for (auto const &[key, _] : other.data) {
          if (!data.contains(key)) {
            return key;
          }

          // TODO check if this section is redundant
          const auto& [values1, error1, nsamples1] = other.data.at(key);
          const auto& [values2, error2, nsamples2] = data.at(key);

          auto shape1 = dataframe::utils::get_shape(values1);
          auto shape2 = dataframe::utils::get_shape(values2);

          if (!shapes_equal(shape1, shape2)) {
            return key;
          }
        }
        
        return std::nullopt;
      }
  };
}
