#include "DataFrame.hpp"

#include <glaze/glaze.hpp>
#include <fmt/core.h>

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
    "data", &DataSlide::data,
    "samples", &DataSlide::samples,
    "buffer", &DataSlide::buffer
  );
};

DataSlide::DataSlide(const std::string &s) {
  auto pe = glz::read_json(*this, s);
  if (pe) {
    throw std::invalid_argument(fmt::format("Error parsing DataSlide: \n{}", glz::format_error(pe, s)));
  }
}

DataSlide::DataSlide(const std::vector<byte_t>& bytes) {
  auto pe = glz::read_beve(*this, bytes);
  if (pe) {
    throw std::invalid_argument("Error parsing DataSlide from binary.");
  }
}

std::vector<byte_t> DataSlide::to_bytes() const {
  std::vector<byte_t> data;
  glz::write_beve(*this, data);
  return data;
}

std::string DataSlide::to_json() const {
  static constexpr auto partial = glz::json_ptrs("/params", "/data", "/samples");

  std::string s;
  glz::write_json<partial>(*this, s);
  return glz::prettify_json(s);
}

// TODO redo with fmt
std::string DataSlide::describe() const {
  std::string s = fmt::format("params: {},\n", glz::write_json(params).value_or("error"));

  //std::string s = "params: " + glz::write_json(params).value_or("error") + "},\n";
  s += "data: { ";
  std::vector<std::string> buffer;
  for (auto const& [key, d] : data) {
    buffer.push_back(key + ": " + std::to_string(d.size()));
  }

  for (size_t i = 0; i < buffer.size(); i++) {
    s += buffer[i];
    if (i != buffer.size() - 1) {
      s += ", ";
    }
  }

  s += " }, \n";
  s += "samples: { ";

  buffer.clear();
  for (auto const& [key, d] : samples) {
    buffer.push_back(key + ": " + std::to_string(d.size()));
  }

  for (size_t i = 0; i < buffer.size(); i++) {
    s += buffer[i];
    if (i != buffer.size() - 1) {
      s += ", ";
    }
  }

  s += " }\n";

  return s;
}

DataFrame::DataFrame(const std::vector<byte_t>& bytes) {
  auto pe = glz::read_beve(*this, bytes);
  if (pe) {
    throw std::invalid_argument("Error parsing DataFrame from binary.");
  }

  init_tolerance();
  init_qtable();
}

DataFrame::DataFrame(const std::string& s) {
  auto pe = glz::read_json(*this, s); // try json deserialization
  if (pe) {
    throw std::invalid_argument(fmt::format("Error parsing DataFrame: \n{}", glz::format_error(pe, s)));
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

std::string DataFrame::describe(size_t num_slides) const {
  std::string s = fmt::format("params: {},\n", glz::write_json(params).value_or("error"));
  s += fmt::format("metadata: {},\n", glz::write_json(metadata).value_or("error"));

  s += fmt::format("number of slides: {}\n", slides.size());
  size_t m = std::min(slides.size(), num_slides);
  for (size_t i = 0; i < m; i++) {
    s += fmt::format("\nslide {}: \n{}", i, slides[i].describe());
  }
  return s;
}

std::vector<byte_t> DataFrame::to_bytes() const {
  std::vector<byte_t> data;
  glz::write_beve(*this, data);
  return data;
}

std::string DataFrame::to_json() const {
  // For now, don't print slides when displaying DataFrame
  static constexpr auto partial = glz::json_ptrs("/params", "/metadata");

  std::string s;
  glz::write_json<partial>(*this, s);
  return glz::prettify_json(s);
}

