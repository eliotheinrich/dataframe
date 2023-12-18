#pragma once

#include "DataSlide.hpp"

#include <optional>
#include <set>
#include <fstream>

namespace dataframe {

  class DataFrame {
    public:
      double atol;
      double rtol;

      Params params;
      Params metadata;
      std::vector<DataSlide> slides;

      struct glaze {
        static constexpr auto value = glz::object(
          "params", &DataFrame::params,
          "metadata", &DataFrame::metadata,
          "slides", &DataFrame::slides
        );
      };

      DataFrame() : atol(ATOL), rtol(RTOL) {}

      DataFrame(double atol, double rtol) : atol(atol), rtol(rtol) {}

      DataFrame(const std::vector<DataSlide>& slides) : atol(ATOL), rtol(RTOL) {
        for (uint32_t i = 0; i < slides.size(); i++) {
          add_slide(slides[i]);
        }
      }

      DataFrame(const Params& params, const std::vector<DataSlide>& slides) : atol(ATOL), rtol(RTOL) {
        add_param(params);
        for (uint32_t i = 0; i < slides.size(); i++) {
          add_slide(slides[i]); 
        }
      }

      DataFrame(const std::vector<std::byte>& data) {
        auto pe = glz::read_binary(*this, data);
        if (pe) {
          std::string error_message = "Error parsing DataFrame from binary.";
          throw std::invalid_argument(error_message);
        }
      }

      DataFrame(const std::string& s) {
        auto pe = glz::read_json(*this, s); // try json deserialization
        if (pe) {
          pe = glz::read_binary(*this, s); // try binary deserialization
            if (pe) {
              try {
                *this = deserialize(s); // try deprecated deserialization
              } catch (const std::runtime_error &e) {
                std::string error_message = "Error parsing DataFrame: \n" + glz::format_error(pe, s);
                throw std::invalid_argument(error_message);
              }
          }
        }

        if (metadata.count("atol")) {
          atol = std::get<double>(metadata.at("atol"));
        } else {
          atol = ATOL;
        }

        if (metadata.count("rtol")) {
          rtol = std::get<double>(metadata.at("rtol"));
        } else {
          rtol = RTOL;
        }

        init_qtable();
      }

      DataFrame(const DataFrame& other) : atol(other.atol), rtol(other.rtol) {
        for (auto const& [key, val] : other.params) {
          params[key] = val;
        }

        for (auto const& [key, val] : other.metadata) {
          metadata[key] = val;
        }

        for (auto const& slide : other.slides) {
          add_slide(DataSlide(slide));
        }
      }

      void add_slide(const DataSlide& ds) {
        slides.push_back(ds);
        qtable_initialized = false;
      }

      bool remove_slide(uint32_t i) {
        if (i >= slides.size()) {
          return false;
        }

        slides.erase(slides.begin() + i);
        qtable_initialized = false;
        return true;
      }

      template <typename T>
      void add_metadata(const std::string& s, const T t) {
        metadata[s] = t;
      }

      void add_metadata(const Params &params) {
        for (auto const &[key, field] : params) {
          add_metadata(key, field);
        }
      }

      template <typename T>
      void add_param(const std::string& s, const T t) { 
        params[s] = t; 
        qtable_initialized = false;
      }


      void add_param(const Params &params) {
        for (auto const &[key, field] : params) {
          add_param(key, field);
        }

        qtable_initialized = false;
      }

      bool contains(const std::string& s) const {
        return params.count(s);
      }

      var_t get(const std::string& s) const {
        if (params.count(s)) {
          return get_param(s);
        } else {
          return get_metadata(s);
        }
      }

      var_t get_param(const std::string& s) const {
        return params.at(s);
      }

      var_t get_metadata(const std::string& s) const {
        return metadata.at(s);
      }

      bool remove(const std::string& s) {
        if (params.count(s)) {
          return remove_param(s);
        } else {
          return remove_metadata(s);
        }
      }

      bool remove_param(const std::string& s) {
        qtable_initialized = false;
        return params.erase(s);
      }

      bool remove_metadata(const std::string& s) {
        return metadata.erase(s);
      }

      std::string to_string() const {
        return glz::write_json(*this);
      }

      std::string to_binary() const {
        return glz::write_binary(*this);
      }

      std::string to_json() const {
        return glz::prettify(glz::write_json(*this), false, 2);
      }

      void write(const std::string& filename) const {
        std::vector<std::string> components = utils::split(filename, ".");
        std::string extension = components[components.size() - 1];
        std::string content;
        if (extension == "json") {
          content = to_json();
        } else if (extension == "eve") {
          content = to_binary();
        } else {
          content = to_json();
        }

        // Save to file
        std::ofstream output_file(filename);
        output_file << content;
        output_file.close();
      }

      bool field_congruent(const std::string& s) const {
        if (slides.size() == 0) {
          return true;
        }

        DataSlide first_slide = slides[0];

        if (!first_slide.contains(s)) {
          return false;
        }

        var_t first_slide_val = first_slide.get_param(s);

        utils::var_t_eq equality_comparator(atol, rtol);
        for (auto slide : slides) {
          if (!slide.contains(s)) {
            return false;
          }

          if (!equality_comparator(slide.get_param(s), first_slide_val)) {
            return false;
          }
        }

        return true;
      }

      void promote_params() {
        utils::var_t_eq equality_comparator(atol, rtol);
        if (slides.size() == 0) {
          return;
        }

        DataSlide first_slide = slides[0];

        std::vector<std::string> keys;
        for (auto const &[key, _] : first_slide.params) {
          keys.push_back(key);
        }

        for (auto key : keys) {
          if (field_congruent(key)) { 
            promote_field(key);
          }
        }
      }

      DataFrame filter(const std::vector<Params>& constraints, bool invert = false) {
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

      typedef std::variant<std::string, std::vector<std::string>> query_key_t;
      std::vector<query_t> query(
        const query_key_t& keys_var, 
        const Params& constraints, 
        bool unique=false, 
        bool error=false
      ) {
        std::vector<std::string> keys;
        if (keys_var.index() == 0) {
          keys = std::vector<std::string>{std::get<std::string>(keys_var)};
        } else {
          keys = std::get<std::vector<std::string>>(keys_var);
        }
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
        std::vector<query_t> result;

        for (auto const& key : keys) {
          query_t key_result;
          if (params.count(key)) {
            key_result = query_t{params[key]};
          } else if (metadata.count(key)) {
            key_result = query_t{metadata[key]};
          } else if (slides[*inds.begin()].params.count(key)) {
            std::vector<var_t> param_vals;
            for (auto const i : inds) {
              param_vals.push_back(slides[i].params[key]);
            }
            key_result = query_t{param_vals};
          } else {
            std::vector<std::vector<std::vector<double>>> data_vals;

            if (error) {
              for (auto const i : inds) {
                data_vals.push_back(slides[i].get_std(key));
              }
            } else {
              for (auto const i : inds) {
                data_vals.push_back(slides[i].get_data(key));
              }
            }

            key_result = data_vals;
          }

          result.push_back(key_result);
        }

        return result;
      }

      void reduce() {
        std::vector<DataSlide> new_slides;

        std::set<uint32_t> reduced;
        for (uint32_t i = 0; i < slides.size(); i++) {
          if (reduced.count(i)) {
            continue;
          }

          DataSlide slide(slides[i]);
          auto inds = compatible_inds(slide.params);
          for (auto const j : inds) {
            if (i == j) {
              continue;
            }
            slide = slide.combine(slides[j], atol, rtol);
            reduced.insert(j);
          } 
          new_slides.push_back(slide);
        }

        slides = new_slides;
      }

      // TODO seems to be a bug here
      DataFrame combine(const DataFrame &other) const {
        if (params.empty() && slides.empty()) {
          return DataFrame(other);
        } else if (other.params.empty() && other.slides.empty()) {
          return DataFrame(*this);
        }


        // Combine matching metadata
        DataFrame df;
        utils::var_t_eq equality_comparator(atol, rtol);
        for (auto const &[k, v] : metadata) {
          if (other.metadata.count(k) && equality_comparator(v, other.metadata.at(k))) {
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
          if (other_frame_params.count(k)) {
            both_frame_params.insert(k);
          }
        }

        // Erase keys which appear in both frame params
        std::set<std::string> to_erase1;
        for (auto const& k : self_frame_params) {
          if (other_frame_params.count(k)) {
            to_erase1.insert(k);
          }
        }

        std::set<std::string> to_erase2;
        for (auto const& k : other_frame_params) {
          if (self_frame_params.count(k)) {
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

        Params self_slide_params;
        Params other_slide_params;

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

        df.promote_params();
        df.reduce();

        return df;
      }

    private:
      bool qtable_initialized;
      // qtable stores a list of key: {val: corresponding_slide_indices}
      std::map<std::string, std::vector<std::vector<uint32_t>>> qtable;
      std::map<std::string, std::vector<var_t>> key_vals;

      static DataFrame deserialize(const std::string& s) {
        // Deprecated json deserialization
        DataFrame frame;

        nlohmann::json data = nlohmann::json::parse(s);
        for (auto const &[key, val] : data["params"].items()) {
          frame.params[key] = utils::parse_json_type(val);  
        }

        if (data.contains("metadata")) {
          for (auto const &[key, val] : data["metadata"].items()) {
            frame.metadata[key] = utils::parse_json_type(val);
          }
        }

        if (frame.metadata.count("atol")) {
          frame.atol = std::get<double>(frame.metadata.at("atol"));
        } else {
          frame.atol = ATOL;
        }

        if (frame.metadata.count("rtol")) {
          frame.rtol = std::get<double>(frame.metadata.at("rtol"));
        } else {
          frame.rtol = RTOL;
        }

        for (auto const &slide_str : data["slides"]) {
          frame.add_slide(DataSlide::deserialize(slide_str.dump()));
        }

        return frame;
      }

      uint32_t corresponding_ind(
        const var_t& v, 
        const std::vector<var_t>& vals, 
        const std::optional<utils::var_t_eq>& comp = std::nullopt
      ) {
        utils::var_t_eq equality_comparator = comp.value_or(utils::var_t_eq{atol, rtol});

        for (uint32_t i = 0; i < vals.size(); i++) {
          if (equality_comparator(v, vals[i])) {
            return i;
          }
        }

        return -1;
      }

      // Initialize query table; is called anytime a query is made after the frame has been changed.
      void init_qtable() {
        utils::var_t_eq equality_comparator(atol, rtol);
        key_vals = std::map<std::string, std::vector<var_t>>();

        for (auto const &slide : slides) {
          for (auto const &[key, tar_val] : slide.params) {
            if (!key_vals.count(key)) {
              key_vals[key] = std::vector<var_t>();
            }

            auto result = std::find_if(
              key_vals[key].begin(), key_vals[key].end(), 
              [tar_val, &equality_comparator](const var_t& val) {
                return equality_comparator(tar_val, val);
              }
            );

            if (result == key_vals[key].end()) {
              key_vals[key].push_back(tar_val);
            }
          }
        }

        // Setting up qtable indices
        for (auto const &[key, vals] : key_vals) {
          qtable[key] = std::vector<std::vector<uint32_t>>(vals.size());
        }

        for (uint32_t n = 0; n < slides.size(); n++) {
          auto slide = slides[n];
          for (auto const &[key, vals] : key_vals) {
            var_t val = slide.params[key];
            uint32_t idx = corresponding_ind(val, vals, equality_comparator);
            if (idx == (uint32_t) -1) {
              throw std::invalid_argument("Error in init_qtable.");
            }

            qtable[key][idx].push_back(n);
          }
        }

        qtable_initialized = true;

      }

      void promote_field(const std::string& s) {
        add_param(s, slides.begin()->get_param(s));
        for (auto &slide : slides) {
          slide.remove(s);
        }
      }

      std::set<uint32_t> compatible_inds(const Params& constraints) {
        if (!qtable_initialized) {
          init_qtable();
        }

        // Check if any keys correspond to mismatched Frame-level parameters, in which case return nothing
        utils::var_t_eq equality_comparator(atol, rtol);
        for (auto const &[key, val] : constraints) {
          if (params.count(key) && !(equality_comparator(params[key], val))) {
            return std::set<uint32_t>();
          }
        }

        // Determine which constraints are relevant, i.e. correspond to existing Slide-level parameters
        Params relevant_constraints;
        for (auto const &[key, val] : constraints) {
          if (!params.count(key)) {
            relevant_constraints[key] = val;
          }
        }

        std::set<uint32_t> inds;
        for (uint32_t i = 0; i < slides.size(); i++) {
          inds.insert(i);
        }

        for (auto const &[key, val] : relevant_constraints) {
          // Take set intersection
          std::set<uint32_t> tmp;
          uint32_t idx = corresponding_ind(val, key_vals[key], equality_comparator);
          if (idx == (uint32_t) -1) {
            continue;
          }

          for (auto const i : qtable[key][idx]) {
            if (inds.count(i)) {
              tmp.insert(i);
            }
          }

          inds = tmp;
        }

        return inds;
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

