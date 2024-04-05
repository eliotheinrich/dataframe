#include "DataFrame.hpp"

#include <nlohmann/json.hpp>
#include <glaze/glaze.hpp>

using namespace dataframe;

template<>
struct glz::meta<DataFrame> {
  static constexpr auto value = glz::object(
    "params", &DataFrame::params,
    "metadata", &DataFrame::metadata,
    "slides", &DataFrame::slides
  );
};

template <>
struct glz::meta<DataSlide> {
  static constexpr auto value = glz::object(
    "params", &DataSlide::params,
    "data", &DataSlide::data
  );
};

DataSlide::DataSlide(const std::string &s) {
  auto pe = glz::read_json(*this, s);
  if (pe) {
    std::string error_message = "Error parsing DataSlide: \n" + glz::format_error(pe, s);
    throw std::invalid_argument(error_message);
  }
}

std::string DataSlide::to_string() const {
  return glz::write_json(*this);
}

template <class json_object>
static var_t parse_json_type(json_object p) {
  if ((p.type() == nlohmann::json::value_t::number_integer) || 
      (p.type() == nlohmann::json::value_t::number_unsigned) ||
      (p.type() == nlohmann::json::value_t::boolean)) {
    return var_t{static_cast<double>(p)};
  }  else if (p.type() == nlohmann::json::value_t::number_float) {
    return var_t{(double) p};
  } else if (p.type() == nlohmann::json::value_t::string) {
    return var_t{std::string(p)};
  } else {
    std::stringstream ss;
    ss << "Invalid json item type on " << p << "; aborting.\n";
    throw std::invalid_argument(ss.str());
  }
}

static inline std::vector<Sample> read_samples(const nlohmann::json& arr) {
  // Deprecated json deserialization

  if (!arr.is_array()) {
    throw std::invalid_argument("Invalid value passed to read_samples.");
  }

  size_t num_elements = arr.size();

  // Need to assume at least one element exists for the remainder
  if (num_elements == 0) {
    return std::vector<Sample>();
  }

  std::string arr_str = arr.dump();

  if (Sample::is_valid(arr_str)) {
    return std::vector<Sample>{Sample(arr_str)};
  }

  std::vector<Sample> samples;
  samples.reserve(num_elements);

  for (auto const& el : arr) {
    // Check that dimension is consistent
    std::string s = el.dump();
    if (!Sample::is_valid(s)) {
      std::string error_message = "Invalid string " + s + " passed to read_samples.";
      throw std::invalid_argument(error_message);
    }

    samples.push_back(Sample(s));
  }

  return samples;
}

DataSlide DataSlide::deserialize(const std::string& s) {
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
        std::vector<Sample> samples = read_samples(v);
        slide.push_data(key, samples);
      }
    } else {
      slide.add_param(key, parse_json_type(val));
    }
  }

  return slide;
}

DataFrame::DataFrame(const std::vector<uint8_t>& data) {
  auto pe = glz::read_binary(*this, data);
  if (pe) {
    std::string error_message = "Error parsing DataFrame from binary.";
    throw std::invalid_argument(error_message);
  }

  init_tolerance();
  init_qtable();
}

DataFrame::DataFrame(const std::string& s) {
  auto pe = glz::read_json(*this, s); // try json deserialization
  if (pe) {
    try {
      *this = deserialize(s); // try deprecated deserialization
    } catch (const std::runtime_error &e) {
      std::string error_message = "Error parsing DataFrame: \n" + glz::format_error(pe, s);
      throw std::invalid_argument(error_message);
    }
  }

  if (metadata.count("atol")) {
    atol = std::get<double>(metadata.at("atol"));
  } else {
    atol = DF_ATOL;
  }

  if (metadata.count("rtol")) {
    rtol = std::get<double>(metadata.at("rtol"));
  } else {
    rtol = DF_RTOL;
  }

  init_tolerance();
  init_qtable();
}

std::string DataFrame::to_string() const {
  return glz::write_json(*this);
}

std::vector<std::byte> DataFrame::to_binary() const {
  std::vector<std::byte> data;
  glz::write_binary(*this, data);
  return data;
}

std::string DataFrame::to_json() const {
  return glz::prettify(glz::write_json(*this), false, 2);
}

DataFrame DataFrame::deserialize(const std::string& s) {
  // Deprecated json deserialization
  DataFrame frame;

  nlohmann::json data = nlohmann::json::parse(s);
  for (auto const &[key, val] : data["params"].items()) {
    frame.params[key] = parse_json_type(val);  
  }

  if (data.contains("metadata")) {
    for (auto const &[key, val] : data["metadata"].items()) {
      frame.metadata[key] = parse_json_type(val);
    }
  }

  if (frame.metadata.count("atol")) {
    frame.atol = std::get<double>(frame.metadata.at("atol"));
  } else {
    frame.atol = DF_ATOL;
  }

  if (frame.metadata.count("rtol")) {
    frame.rtol = std::get<double>(frame.metadata.at("rtol"));
  } else {
    frame.rtol = DF_RTOL;
  }

  for (auto const &slide_str : data["slides"]) {
    frame.add_slide(DataSlide::deserialize(slide_str.dump()));
  }

  return frame;
}
