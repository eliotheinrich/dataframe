#pragma once

#include "DataSlide.hpp"

#include <optional>
#include <iterator>
#include <set>
#include <fstream>
#include <numeric>

namespace dataframe {

  class DataFrame {
    public:
      double atol;
      double rtol;

      ExperimentParams params;
      ExperimentParams metadata;
      std::vector<DataSlide> slides;

      DataFrame() {
        init_tolerance();
      }

      ~DataFrame()=default;

      DataFrame(double atol, double rtol) : atol(atol), rtol(rtol) {}

      DataFrame(DataFrame&&) noexcept=default;
      DataFrame& operator=(DataFrame&&) noexcept=default;

      DataFrame(const std::vector<DataSlide>& slides) : atol(DF_ATOL), rtol(DF_RTOL) {
        for (uint32_t i = 0; i < slides.size(); i++) {
          add_slide(slides[i]);
        }

        init_tolerance();
      }

      DataFrame(const ExperimentParams& params, const std::vector<DataSlide>& slides) : atol(DF_ATOL), rtol(DF_RTOL) {
        add_param(params);
        for (uint32_t i = 0; i < slides.size(); i++) {
          add_slide(slides[i]); 
        }

        init_tolerance();
      }

      DataFrame(const std::vector<byte_t>& bytes);

      DataFrame(const DataFrame& other) : atol(other.atol), rtol(other.rtol), params(other.params), metadata(other.metadata), slides(other.slides) {
        init_tolerance();
      }

      static DataFrame from_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);

        std::streampos file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(file_size);
        file.read(buffer.data(), file_size);
        file.close();

        std::vector<byte_t> data(file_size);
        for (int i = 0; i < file_size; i++) {
          data[i] = static_cast<byte_t>(buffer[i]);
        }

        return DataFrame(data);
      }

      void add_slide(const DataSlide& ds) {
        slides.push_back(ds);
      }

      bool remove_slide(uint32_t i) {
        if (i >= slides.size()) {
          return false;
        }

        slides.erase(slides.begin() + i);
        return true;
      }

      template <typename T>
      void add_metadata(const std::string& s, const T t) {
        metadata[s] = t;
      }

      void add_metadata(const ExperimentParams &params) {
        for (auto const &[key, field] : params) {
          add_metadata(key, field);
        }
      }

      template <typename T>
      void add_param(const std::string& s, const T t) { 
        params[s] = t; 
      }

      void add_param(const ExperimentParams &params) {
        for (auto const &[key, field] : params) {
          add_param(key, field);
        }
      }

      bool contains(const std::string& s) const {
        return params.contains(s) || metadata.contains(s);
      }

      Parameter get_param(const std::string& s) const {
        return params.at(s);
      }

      Parameter get_metadata(const std::string& s) const {
        return metadata.at(s);
      }

      bool remove_param(const std::string& s) {
        return params.erase(s);
      }

      bool remove_metadata(const std::string& s) {
        return metadata.erase(s);
      }

      std::string describe(size_t num_slides=0) const;

      std::string to_json() const;

      std::vector<byte_t> to_bytes() const;

      void write(const std::string& filename) const {
        std::vector<std::string> components = utils::split(filename, ".");
        std::string extension = components[components.size() - 1];
        if (extension == "eve") {
          auto content = to_bytes();
          std::ofstream output_file(filename, std::ios::out | std::ios::binary);
          output_file.write(reinterpret_cast<const char*>(&content[0]), content.size());
          output_file.close();
        } else {
          std::string content = to_json();
          std::ofstream output_file(filename);
          output_file << content;
          output_file.close();
        }
      }

      bool field_congruent(const std::string& s) const {
        if (slides.size() == 0) {
          return true;
        }

        const DataSlide& first_slide = slides[0];

        if (!first_slide.contains(s)) {
          return false;
        }

        Parameter first_slide_val = first_slide.get_param(s);

        utils::param_eq equality_comparator(atol, rtol);
        for (const auto& slide : slides) {
          if (!slide.params.contains(s)) {
            return false;
          }

          if (!equality_comparator(slide.get_param(s), first_slide_val)) {
            return false;
          }
        }

        return true;
      }

      void promote_params() {
        utils::param_eq equality_comparator(atol, rtol);
        if (slides.size() == 0) {
          return;
        }

        const DataSlide& first_slide = slides[0];

        std::vector<std::string> keys;
        for (auto const &[key, _] : first_slide.params) {
          keys.push_back(key);
        }

        for (const auto& key : keys) {
          if (field_congruent(key)) { 
            promote_field(key);
          }
        }
      }

      DataFrame filter(const std::vector<ExperimentParams>& constraints, bool invert = false) {
        std::set<uint32_t> inds;
        for (auto const &constraint : constraints) {
          auto c_inds = compatible_inds(constraint);
          std::set<uint32_t> ind_union;

          std::set_union(
            inds.begin(), inds.end(),
            c_inds.begin(), c_inds.end(),
            std::inserter(ind_union, ind_union.begin())
          );
          inds = ind_union;
        }

        if (invert) {
          std::vector<uint32_t> all_inds(this->slides.size());
          std::iota(all_inds.begin(), all_inds.end(), 0);

          std::set<uint32_t> all_inds_set(all_inds.begin(), all_inds.end());

          std::set<uint32_t> set_sd;
          std::set_symmetric_difference(
            inds.begin(), inds.end(),
            all_inds_set.begin(), all_inds_set.end(),
            std::inserter(set_sd, set_sd.begin())
          );
          inds = set_sd;
        }

        std::vector<DataSlide> slides;
        for (auto i : inds) {
          slides.push_back(this->slides[i]);
        }

        return DataFrame(params, slides);
      }

      enum QueryType {
        Mean,
        StandardDeviation,
        NumSamples,
        StandardError,
      };

      std::vector<query_t> query(
        const std::vector<std::string>& keys, 
        const ExperimentParams& constraints, 
        bool unique=false, 
        QueryType query_type=QueryType::Mean
      ) {
        if (unique) {
          auto result = query(keys, constraints);
          return utils::make_query_unique(result, utils::make_query_t_unique(atol, rtol));
        }

        // Determine indices of slides which respect the given constraints
        auto inds = compatible_inds(constraints);

        // Constraints yield no valid slides, so return nothing
        if (inds.empty()) {
          return std::vector<query_t>();
        }

        // Compile result of query
        std::vector<query_t> result(keys.size());

        for (size_t p = 0; p < keys.size(); p++) {
          const std::string& key = keys[p];
          query_t key_result;
          if (params.contains(key)) { // Frame-level param
            key_result = query_t{params[key]};
          } else if (metadata.contains(key)) { // Metadata param
            key_result = query_t{metadata[key]};
          } else if (slides[inds[0]].params.contains(key)) { // Slide-level param
            std::vector<Parameter> param_vals(inds.size());
            for (size_t i = 0; i < inds.size(); i++) {
              uint32_t j = inds[i];
              param_vals[i] = slides[j].params[key];
            }
            key_result = query_t{param_vals};
          } else { // Data; check on query_type
            std::vector<size_t> shape = slides[inds[0]].get_shape(key);
            size_t data_size = DataSlide::product(shape);

            if (query_type == QueryType::NumSamples) {
              std::vector<size_t> values(inds.size() * data_size);
              for (size_t i = 0; i < inds.size(); i++) {
                uint32_t j = inds[i];
                auto const& [shapej, valuesj, sampling_dataj] = slides[j].data.at(key);
                if (!DataSlide::shapes_equal(shape, shapej)) {
                  throw std::runtime_error(fmt::format("Ragged shapes detected: {} and {}", shape, shapej));
                }
                const auto& nsamples = slides[j].get_num_samples(key);
                std::copy(nsamples.begin(), nsamples.end(), values.begin() + i * data_size);
              }

              shape.insert(shape.begin(), inds.size());
              key_result = std::make_pair(shape, std::move(values));
            } else {
              std::vector<double> values(inds.size() * data_size);
              for (size_t i = 0; i < inds.size(); i++) {
                uint32_t j = inds[i];
                auto const& [shapej, valuesj, sampling_dataj] = slides[j].data.at(key);
                if (!DataSlide::shapes_equal(shape, shapej)) {
                  throw std::runtime_error(fmt::format("Ragged shapes detected: {} and {}", shape, shapej));
                }

                if (query_type == QueryType::StandardDeviation) {
                  const auto& std = slides[j].get_std(key);
                  std::copy(std.begin(), std.end(), values.begin() + i * data_size);
                } else if (query_type == QueryType::StandardError) {
                  const auto& sderror = slides[j].get_standard_error(key);
                  std::copy(sderror.begin(), sderror.end(), values.begin() + i * data_size);
                } else {
                  const auto& mean = slides[j].get_data(key);
                  std::copy(mean.begin(), mean.end(), values.begin() + i * data_size);
                }
              }

              shape.insert(shape.begin(), inds.size());
              key_result = std::make_pair(shape, std::move(values));
            }
          }

          result[p] = std::move(key_result);
        }
        return result;
      }

      void reduce() {
        std::vector<DataSlide> new_slides;

        std::set<uint32_t> reduced;
        for (uint32_t i = 0; i < slides.size(); i++) {
          if (reduced.contains(i)) {
            continue;
          }

          DataSlide slide = std::move(slides[i]);

          auto inds = compatible_inds(slide.params);
          for (auto const j : inds) {
            if (j == i) {
              continue;
            }

            slide.combine(slides[j]);
            reduced.insert(j);
          } 
          new_slides.push_back(std::move(slide));
        }

        slides = new_slides;
        promote_params();
      }

      DataFrame combine(const DataFrame &other) const {
        if (params.empty() && slides.empty()) {
          return DataFrame(other);
        } else if (other.params.empty() && other.slides.empty()) {
          return DataFrame(*this);
        }


        // Combine matching metadata
        DataFrame df;
        utils::param_eq equality_comparator(atol, rtol);
        for (auto const &[k, v] : metadata) {
          if (other.metadata.contains(k) && equality_comparator(v, other.metadata.at(k))) {
            df.add_metadata(k, v);
          }
        }

        // Inspect params
        std::set<std::string> self_frame_params;
        for (auto const& [k, _] : params) {
          self_frame_params.insert(k);
        }

        std::set<std::string> other_frame_params;
        for (auto const& [k, _] : other.params) {
          other_frame_params.insert(k);
        }

        // both_frame_params is the intersection of keys of params and other.params
        std::set<std::string> both_frame_params;
        for (auto const& k : self_frame_params) {
          if (other_frame_params.contains(k)) {
            both_frame_params.insert(k);
          }
        }

        // Erase keys which appear in both frame params
        std::set<std::string> to_erase1;
        for (auto const& k : self_frame_params) {
          if (other_frame_params.contains(k)) {
            to_erase1.insert(k);
          }
        }

        std::set<std::string> to_erase2;
        for (auto const& k : other_frame_params) {
          if (self_frame_params.contains(k)) {
            to_erase2.insert(k);
          }
        }

        for (auto const& k : to_erase1) {
          self_frame_params.erase(k);
        }

        for (auto const& k : to_erase2) {
          other_frame_params.erase(k);
        }

        // self_frame_params and other_frame_params now only contain parameters unique to that frame

        ExperimentParams self_slide_params;
        ExperimentParams other_slide_params;

        for (auto const& k : both_frame_params) {
          if (equality_comparator(params.at(k), other.params.at(k))) {
            df.add_param(k, params.at(k));
          } else {
            self_slide_params[k] = params.at(k);
            other_slide_params[k] = other.params.at(k);
          }
        }

        for (auto const& k : self_frame_params) {
          self_slide_params[k] = params.at(k);
        }

        for (auto const& k : other_frame_params) {
          other_slide_params[k] = other.params.at(k);
        }

        for (auto const& slide : slides) {
          DataSlide ds(slide);
          for (auto const& [k, v] : self_slide_params) {
            ds.add_param(k, v);
          }

          df.add_slide(ds);
        }

        for (auto const& slide : other.slides) {
          DataSlide ds(slide);
          for (auto const& [k, v] : other_slide_params) {
            ds.add_param(k, v);
          }

          df.add_slide(ds);
        }

        return df;
      }

    private:
      void init_tolerance() {
        if (metadata.contains("atol")) {
          atol = std::get<double>(metadata.at("atol"));
        } else {
          atol = DF_ATOL;
        }

        if (metadata.contains("rtol")) {
          rtol = std::get<double>(metadata.at("rtol"));
        } else {
          rtol = DF_RTOL;
        }
      }

      uint32_t corresponding_ind(
        const Parameter& v, 
        const std::vector<Parameter>& vals, 
        const std::optional<utils::param_eq>& comp = std::nullopt
      ) {
        utils::param_eq equality_comparator = comp.value_or(utils::param_eq{atol, rtol});

        for (uint32_t i = 0; i < vals.size(); i++) {
          if (equality_comparator(v, vals[i])) {
            return i;
          }
        }

        return -1;
      }

      void promote_field(const std::string& s) {
        add_param(s, slides.begin()->get_param(s));
        for (auto &slide : slides) {
          slide.remove_param(s);
        }
      }

      std::vector<uint32_t> compatible_inds(const ExperimentParams& constraints) {
        // Check if any keys correspond to mismatched Frame-level parameters, in which case return nothing
        utils::param_eq equality_comparator(atol, rtol);
        for (auto const &[key, val] : constraints) {
          if (params.contains(key) && !(equality_comparator(params[key], val))) {
            return std::vector<uint32_t>();
          }
        }

        // Determine which constraints are relevant, i.e. correspond to existing Slide-level parameters
        ExperimentParams relevant_constraints;
        for (auto const &[key, val] : constraints) {
          if (!params.contains(key)) {
            relevant_constraints[key] = val;
          }
        }

        std::set<uint32_t> inds;
        for (uint32_t i = 0; i < slides.size(); i++) {
          inds.insert(i);
        }

        for (const auto& [key, val] : relevant_constraints) {
          std::vector<size_t> to_remove;
          for (const auto i : inds) {
            if (slides[i].params.contains(key)) {
              if (!equality_comparator(slides[i].params.at(key), val)) {
                to_remove.push_back(i);
              }
            } else {
              to_remove.push_back(i);
            }
          }

          for (const auto i : to_remove) {
            inds.erase(i);
          }
        }

        std::vector<uint32_t> inds_vec(inds.begin(), inds.end());
        std::sort(inds_vec.begin(), inds_vec.end());
        return inds_vec;
      }
  };

  template <>
  inline void DataFrame::add_metadata<int>(const std::string& s, const int t) { 
    add_metadata(s, static_cast<double>(t));
  }

  template <>
  inline void DataFrame::add_param<int>(const std::string& s, const int t) { 
    add_param(s, static_cast<double>(t));
  }
}

