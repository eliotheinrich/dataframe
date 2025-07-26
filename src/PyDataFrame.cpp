#include "PyDataFrame.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/shared_ptr.h>

#include "test.hpp"

using namespace nanobind::literals;
using namespace dataframe;
using namespace dataframe::utils;

using py_query_result = std::variant<query_t, std::vector<query_t>>;

NB_MODULE(dataframe_bindings, m) {
  m.def("load_params", &utils::load_params);

  nanobind::class_<TestSampler>(m, "TestSampler")
    .def(nanobind::init<ExperimentParams&>())
    .def_static("create_and_emplace", [](ExperimentParams& params) {
      TestSampler sampler(params);
      return std::make_pair(sampler, params);
    })
    .def("get_samples", [](TestSampler& self, int t) {
      SampleMap samples;
      self.add_samples(t, samples);
      return samples;
    });

  nanobind::class_<DataSlide>(m, "DataSlide_")
    .def(nanobind::init<>())
    .def(nanobind::init<ExperimentParams&>())
    .def(nanobind::init<const DataSlide&>())
    .def("__init__", [](DataSlide* t, nanobind::bytes&& bytes) {
      auto byte_vec = convert_bytes(bytes);
      new (t) DataSlide(std::move(byte_vec));
    })
    .def_rw("params", &DataSlide::params)
    //.def_rw("data", &DataSlide::data)
    .def("add_param", [](DataSlide& self, const std::string& key, const Parameter& param) { self.add_param(key, param); })
    .def("add_param", [](DataSlide& self, const ExperimentParams& params) { self.add_param(params); })
    .def("_add_data", [](DataSlide& self, const std::string& key, ndarray<double> values, std::optional<ndarray<double>> errors_opt, std::optional<ndarray<size_t>> nsamples_opt) { 
      self.add_data(key, values, errors_opt, nsamples_opt);
    }, "key"_a, "values"_a, "error"_a = nanobind::none(), "nsamples"_a = nanobind::none())
    .def("_concat_data", [](DataSlide& self, const std::string& key, ndarray<double> values, std::optional<std::vector<size_t>> shape_opt, std::optional<ndarray<double>> errors_opt, std::optional<ndarray<size_t>> nsamples_opt) { 
      self.concat_data(key, values, shape_opt, errors_opt, nsamples_opt);
    }, "key"_a, "values"_a, "shape"_a, "error"_a = nanobind::none(), "nsamples"_a = nanobind::none())
    .def("get_data", [](const DataSlide& self, const std::string& key) {
      return self.get_data(key);
    })
    .def("get_std", [](const DataSlide& self, const std::string& key) {
      return self.get_std(key);
    })
    .def("get_standard_error", [](const DataSlide& self, const std::string& key) {
      return self.get_standard_error(key);
    })
    .def("get_num_samples", [](const DataSlide& self, const std::string& key) {
      return self.get_num_samples(key);
    })
    .def("remove", &DataSlide::remove_param)
    .def("_inject_buffer", [](DataSlide& slide, const nanobind::bytes& bytes) {
      slide.buffer = convert_bytes(bytes);
    })
    .def("_get_buffer", [](const DataSlide& slide) {
      return convert_bytes(slide.buffer);
    })
    .def("__contains__", &DataSlide::contains)
    .def("__getitem__", [](const DataSlide& self, const std::string& key) {
      try {
        return self.get_param(key);
      } catch (const std::runtime_error& e) {
        throw nanobind::key_error(e.what());
      }
    })
    .def("__setitem__", [](DataSlide& self, const std::string& key, const Parameter& param) { self.add_param(key, param); })
    .def("__str__", &DataSlide::describe)
    .def("__getstate__", [](const DataSlide& slide){ return convert_bytes(slide.to_bytes()); })
    .def("__setstate__", [](DataSlide& slide, const nanobind::bytes& bytes){ new (&slide) DataSlide(convert_bytes(bytes)); })
    .def("describe", &DataSlide::describe)
    .def("congruent", &DataSlide::congruent)
    .def("combine", &DataSlide::combine, "other"_a, "atol"_a = DF_ATOL, "rtol"_a = DF_RTOL);

  auto _query = [](DataFrame& df, const std::vector<std::string>& keys, const ExperimentParams& constraints, bool unique, DataFrame::QueryType query_type) {
    std::vector<query_t> results = df.query(keys, constraints, unique, query_type);

    if (results.size() == 1) {
      return py_query_result{results[0]};
    }

    return py_query_result{results};
  };

  nanobind::class_<DataFrame>(m, "DataFrame")
    .def(nanobind::init<>())
    .def(nanobind::init<const std::vector<DataSlide>&>())
    .def(nanobind::init<const ExperimentParams&, const std::vector<DataSlide>&>())
    .def(nanobind::init<const DataFrame&>())
    .def(nanobind::init<double, double>())
    .def("__init__", [](DataFrame* t, const nanobind::bytes& bytes) {
      auto byte_vec = convert_bytes(bytes);
      new (t) DataFrame(std::move(byte_vec));
    })
    .def_rw("params", &DataFrame::params)
    .def_rw("metadata", &DataFrame::metadata)
    //.def_rw("slides", &DataFrame::slides)
    .def_rw("atol", &DataFrame::atol)
    .def_rw("rtol", &DataFrame::rtol)
    .def("add_slide", &DataFrame::add_slide)
    .def("add_param", [](DataFrame& self, const std::string& key, const Parameter& param) { self.add_param(key, param); })
    .def("add_param", [](DataFrame& self, const ExperimentParams& params) { self.add_param(params); })
    .def("add_metadata", [](DataFrame& self, const std::string& key, const Parameter& param) { self.add_metadata(key, param); })
    .def("add_metadata", [](DataFrame& self, const ExperimentParams& params) { self.add_metadata(params); })
    .def("remove_param", &DataFrame::remove_param)
    .def("remove_metadata", &DataFrame::remove_metadata)
    .def("__contains__", &DataFrame::contains)
    .def("__getitem__", [](const DataFrame& self, const std::string& key) {
      if (self.params.contains(key)) {
        return self.params.at(key);
      } else if (self.metadata.contains(key)) {
        return self.metadata.at(key);
      } else {
        throw nanobind::key_error(fmt::format("Could not find key {} in params or metadata.", key).c_str());
      }
    })
    .def("__len__", [](const DataFrame& self) { return self.slides.size(); })
    .def("__setitem__", [](DataFrame& self, const std::string& key, const Parameter& param) { self.add_param(key, param); })
    .def("__str__", &DataFrame::to_json)
    .def("__add__", &DataFrame::combine)
    .def("__getstate__", [](const DataFrame& frame){ return convert_bytes(frame.to_bytes()); })
    .def("__setstate__", [](DataFrame& frame, const nanobind::bytes& bytes){ new (&frame) DataFrame(convert_bytes(bytes)); })
    .def("describe", &DataFrame::describe, "num_slides"_a = 0)
    .def("write", &DataFrame::write)
    .def("promote_params", &DataFrame::promote_params)
    .def("reduce", &DataFrame::reduce)
    .def("query", [&_query](DataFrame& df, const std::vector<std::string>& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::Mean);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::copy)
    .def("query_std", [&_query](DataFrame& df, const std::vector<std::string>& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::StandardDeviation);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::move)
    .def("query_sde", [&_query](DataFrame& df, const std::vector<std::string>& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::StandardError);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::move)
    .def("query_nsamples", [&_query](DataFrame& df, const std::vector<std::string>& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::NumSamples);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::move);

}
