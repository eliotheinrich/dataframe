#include "PyDataFrame.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/trampoline.h>

using namespace nanobind::literals;
using namespace dataframe;

// Types which are interfaced with python
template <typename T=double>
using py_nbarray = nanobind::ndarray<nanobind::numpy, T>;
using py_query_t = std::variant<Parameter, std::vector<Parameter>, py_nbarray<double>, py_nbarray<size_t>>;
using py_query_result = std::variant<py_query_t, std::vector<py_query_t>>;

struct query_t_to_py {
  py_query_t operator()(Parameter&& v) const { 
    return std::move(v);
  }
  py_query_t operator()(std::vector<Parameter>&& v) const { 
    return std::move(v); 
  }
  py_query_t operator()(ndarray<double>&& data) const {
    return py_query_t{to_nbarray(std::move(data))};
  }
  py_query_t operator()(ndarray<size_t>&& data) const {
    return py_query_t{to_nbarray(std::move(data))};
  }
};

NB_MODULE(dataframe_bindings, m) {
  m.def("load_params", &utils::load_params);

  // Need to statically cast overloaded templated methods
  void (DataSlide::*ds_add_param1)(const ExperimentParams&) = &DataSlide::add_param;
  void (DataSlide::*ds_add_param2)(const std::string&, const Parameter&) = &DataSlide::add_param;

  nanobind::class_<DataObject>(m, "DataObject")
    .def_ro("shape", &DataObject::shape)
    .def_ro("values", &DataObject::values)
    .def_ro("sampling_data", &DataObject::sampling_data);

  nanobind::class_<DataSlide>(m, "DataSlide")
    .def(nanobind::init<>())
    .def(nanobind::init<ExperimentParams&>())
    .def(nanobind::init<const DataSlide&>())
    .def("__init__", [](DataSlide* t, nanobind::bytes&& bytes) {
      auto byte_vec = convert_bytes(bytes);
      new (t) DataSlide(std::move(byte_vec));
    })
    .def_rw("params", &DataSlide::params)
    .def_rw("data", &DataSlide::data)
    .def("add_param", ds_add_param1)
    .def("add_param", ds_add_param2)
    .def("add_data", [](DataSlide& self, const std::string& key, const std::vector<double>& values, std::optional<std::vector<size_t>> shape, std::optional<std::vector<double>> std, std::optional<std::vector<size_t>> nsamples) { 
      if (std && nsamples) {
        self.add_data(key, values, shape, SamplingData{std.value(), nsamples.value()}); 
      } else if (std || nsamples) {
        throw std::runtime_error("Cannot pass only one of std and nsamples.");
      } else {
        self.add_data(key, values, shape); 
      }
    }, "key"_a, "values"_a, "shape"_a = nanobind::none(), "error"_a = nanobind::none(), "nsamples"_a = nanobind::none())
    .def("add_data", [](DataSlide& self, const std::string& key, const DataObject& data) {
      self.add_data(key, data);
    })
    .def("concat_data", [](DataSlide& self, const std::string& key, const std::vector<double>& values, std::optional<std::vector<size_t>> shape, std::optional<std::vector<double>> std, std::optional<std::vector<size_t>> nsamples) { 
      if (std && nsamples) {
        self.concat_data(key, values, shape, SamplingData{std.value(), nsamples.value()}); 
      } else if (std || nsamples) {
        throw std::runtime_error("Cannot pass only one of std and nsamples.");
      } else {
        self.concat_data(key, values, shape); 
      }
    }, "key"_a, "values"_a, "shape"_a = nanobind::none(), "error"_a = nanobind::none(), "nsamples"_a = nanobind::none())
    .def("concat_data", [](DataSlide& self, const std::string& key, const DataObject& data) {
      self.concat_data(key, data);
    }, "key"_a, "data"_a)
    .def("get_data", [](const DataSlide& self, const std::string& key) {
      auto data = self.get_data(key);
      auto shape = self.get_shape(key);
      return to_nbarray(std::move(ndarray<double>{shape, data}));
    })
    .def("get_std", [](const DataSlide& self, const std::string& key) {
      auto data = self.get_std(key);
      auto shape = self.get_shape(key);
      return to_nbarray(std::move(ndarray<double>{shape, data}));
    })
    .def("get_standard_error", [](const DataSlide& self, const std::string& key) {
      auto data = self.get_standard_error(key);
      auto shape = self.get_shape(key);
      return to_nbarray(std::move(ndarray<double>{shape, data}));
    })
    .def("get_num_samples", [](const DataSlide& self, const std::string& key) {
      auto data = self.get_num_samples(key);
      auto shape = self.get_shape(key);
      return to_nbarray(std::move(ndarray<size_t>{shape, data}));
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
    .def("__setitem__", ds_add_param2)
    .def("__str__", &DataSlide::to_json)
    .def("__getstate__", [](const DataSlide& slide){ return convert_bytes(slide.to_bytes()); })
    .def("__setstate__", [](DataSlide& slide, const nanobind::bytes& bytes){ new (&slide) DataSlide(convert_bytes(bytes)); })
    .def("describe", &DataSlide::describe)
    .def("congruent", &DataSlide::congruent)
    .def("combine", &DataSlide::combine, "other"_a, "atol"_a = DF_ATOL, "rtol"_a = DF_RTOL);

  void (DataFrame::*df_add_param1)(const ExperimentParams&) = &DataFrame::add_param;
  void (DataFrame::*df_add_param2)(const std::string&, const Parameter&) = &DataFrame::add_param;
  void (DataFrame::*df_add_metadata1)(const ExperimentParams&) = &DataFrame::add_metadata;
  void (DataFrame::*df_add_metadata2)(const std::string&, const Parameter&) = &DataFrame::add_metadata;

  auto _query = [](DataFrame& df, const std::vector<std::string>& keys, const ExperimentParams& constraints, bool unique, DataFrame::QueryType query_type) {
    std::vector<query_t> results = df.query(keys, constraints, unique, query_type);

    size_t num_queries = results.size();

    std::vector<py_query_t> py_results(num_queries);
    for (uint32_t i = 0; i < num_queries; i++) {
      py_results[i] = std::visit(query_t_to_py(), std::move(results[i]));
    }

    if (num_queries == 1) {
      return py_query_result{py_results[0]};
    }

    return py_query_result{py_results};
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
    .def_rw("slides", &DataFrame::slides)
    .def_rw("atol", &DataFrame::atol)
    .def_rw("rtol", &DataFrame::rtol)
    .def("add_slide", &DataFrame::add_slide)
    .def("add_param", df_add_param1)
    .def("add_param", df_add_param2)
    .def("add_metadata", df_add_metadata1)
    .def("add_metadata", df_add_metadata2)
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
    .def("__setitem__", df_add_param2)
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
