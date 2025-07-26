#include "DataFrame.hpp"

#include <glaze/glaze.hpp>
#include <fmt/core.h>

using namespace dataframe;
using namespace dataframe::utils;

//using DataObjectSerialized = std::tuple<
//  std::vector<size_t>,
//  std::vector<double>,
//  std::optional<std::vector<double>>,
//  std::optional<std::vector<size_t>>
//>;
//
//DataObject from_serialized(const DataObjectSerialized& serialized) {
//  auto& [shape, values, error_opt, nsamples_opt] = serialized;
//
//  ndarray<double> values_ = to_ndarray(values, shape);
//  std::optional<ndarray<double>> error_opt_ = std::nullopt;
//  if (error_opt) {
//    error_opt_ = to_ndarray(error_opt.value(), shape);
//  }
//
//  std::optional<ndarray<size_t>> nsamples_opt_ = std::nullopt;
//  if (nsamples_opt) {
//    nsamples_opt_ = to_ndarray(nsamples_opt.value(), shape);
//  }
//
//  return std::make_tuple(values_, error_opt_, nsamples_opt_); 
//}
//
//DataObjectSerialized to_serialized(const DataObject& data) {
//  const auto& [values, error_opt, nsamples_opt] = data;
//
//  size_t N = values.size();
//
//  double* values_ptr = values.data();
//  std::vector<double> values_(values_ptr, values_ptr + N);
//
//  std::optional<std::vector<double>> error_opt_ = std::nullopt;
//  if (error_opt) {
//    double* error_ptr = error_opt->data();
//    error_opt_ = std::vector<double>(error_ptr, error_ptr + N);
//  }
//
//  std::optional<std::vector<size_t>> nsamples_opt_ = std::nullopt;
//  if (nsamples_opt) {
//    size_t* nsamples_ptr = nsamples_opt->data();
//    nsamples_opt_ = std::vector<size_t>(nsamples_ptr, nsamples_ptr + N);
//  }
//
//  std::vector<size_t> shape = get_shape(values);
//
//  return std::make_tuple(shape, values_, error_opt_, nsamples_opt_); 
//}

struct DataSlideSerialized {
  std::map<std::string, Parameter> params;
  std::vector<byte_t> buffer;
  std::map<std::string, DataObject> data;

  DataSlideSerialized()=default;
  DataSlideSerialized(const DataSlide& slide) : params(slide.params), buffer(slide.buffer), data(slide.data) {}
};

DataSlide from_serialized(const DataSlideSerialized& serialized) {
  DataSlide slide;
  slide.params = serialized.params;
  slide.buffer = serialized.buffer;
  slide.data = serialized.data;
  return slide;
}

struct DataFrameSerialized {
  std::map<std::string, Parameter> params;
  std::map<std::string, Parameter> metadata;
  std::vector<DataSlideSerialized> slides;

  DataFrameSerialized()=default;
  DataFrameSerialized(const DataFrame& frame) : params(frame.params), metadata(frame.metadata) {
    size_t num_slides = frame.slides.size();
    std::vector<DataSlideSerialized> serialized_slides(num_slides);
    for (size_t i = 0; i < num_slides; i++) {
      serialized_slides[i] = DataSlideSerialized(frame.slides[i]);
    }

    slides = serialized_slides;
  }
};

DataFrame from_serialized(const DataFrameSerialized& serialized) {
  DataFrame frame;
  frame.params = serialized.params;
  frame.metadata = serialized.metadata;
  size_t num_slides = serialized.slides.size();
  std::vector<DataSlide> slides(num_slides);
  for (size_t i = 0; i < num_slides; i++) {
    slides[i] = from_serialized(serialized.slides[i]);
  }

  frame.slides = slides;

  return frame;
}

template <>
struct glz::meta<DataSlideSerialized> {
  static constexpr auto value = glz::object(
    "params", &DataSlideSerialized::params,
    "data",   &DataSlideSerialized::data,
    "buffer", &DataSlideSerialized::buffer
  );
};

template<>
struct glz::meta<DataFrameSerialized> {
  static constexpr auto value = glz::object(
    "params",   &DataFrameSerialized::params,
    "metadata", &DataFrameSerialized::metadata,
    "slides",   &DataFrameSerialized::slides
  );
};

template <>
struct glz::meta<DataSlide> {
  static constexpr auto value = glz::object(
    "params", &DataSlide::params,
    "buffer", &DataSlide::buffer
  );
};

template<>
struct glz::meta<DataFrame> {
  static constexpr auto value = glz::object(
    "params",   &DataFrame::params,
    "metadata", &DataFrame::metadata
  );
};


DataSlide::DataSlide(const std::vector<byte_t>& bytes) {
  DataSlideSerialized serialized;
  auto parse_error = glz::read_beve(serialized, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error parsing DataSlide from binary: \n{}", glz::format_error(parse_error, bytes)));
  }

  *this = from_serialized(serialized);
}

std::vector<byte_t> DataSlide::to_bytes() const {
  DataSlideSerialized serialized(*this);
  std::vector<byte_t> data;
  auto write_error = glz::write_beve(serialized, data);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing DataSlide to binary: \n{}", glz::format_error(write_error, data)));
  }

  return data;
}

//std::string DataSlide::to_json() const {
//  static constexpr auto partial = glz::json_ptrs("/params", "/data");
//
//  std::string s;
//  auto write_error = glz::write_json<partial>(*this, s);
//  if (write_error) {
//    throw std::runtime_error(fmt::format("Error writing DataSlide to json: \n{}", glz::format_error(write_error, s)));
//  }
//  return glz::prettify_json(s);
//}

std::string DataSlide::describe() const {
  std::string s = fmt::format("params: {},\n", glz::write_json(params).value_or("error"));

  //s += "data: { ";
  //std::vector<std::string> buffer;
  //for (auto const& [key, d] : data) {
  //  auto const& [shape, values, sampling_data] = d;
  //  buffer.push_back(fmt::format("{}: {}", key, shape));
  //}

  //for (size_t i = 0; i < buffer.size(); i++) {
  //  s += buffer[i];
  //  if (i != buffer.size() - 1) {
  //    s += ", ";
  //  }
  //}

  //s += " }\n";

  return s;
}

DataFrame::DataFrame(const std::vector<byte_t>& bytes) {
  DataFrameSerialized serialized;
  auto parse_error = glz::read_beve(serialized, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error parsing DataFrame from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
  
  *this = from_serialized(serialized);

  init_tolerance();
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
  DataFrameSerialized serialized(*this);
  std::vector<byte_t> data;
  auto write_error = glz::write_beve(serialized, data);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing DataFrame to binary: \n{}", glz::format_error(write_error, data)));
  }

  return data;
}

std::string DataFrame::to_json() const {
  // For now, don't print slides when displaying DataFrame
  static constexpr auto partial = glz::json_ptrs("/params", "/metadata");

  std::string s;
  auto write_error = glz::write_json<partial>(*this, s);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing DataFrame to json: \n{}", glz::format_error(write_error, s)));
  }
  return glz::prettify_json(s);
}

