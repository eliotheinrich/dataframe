#include "PyDataFrame.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/trampoline.h>
#include <nanobind/ndarray.h>

using namespace nanobind::literals;
using namespace dataframe;

  // Types which are interfaced with python
using py_nbarray = nanobind::ndarray<nanobind::numpy, double>;
using py_query_t = std::variant<Parameter, std::vector<Parameter>, py_nbarray>;
using py_query_result = std::variant<py_query_t, std::vector<py_query_t>>;

size_t get_query_size(const query_t& q) {
  if (q.index() != 2) {
    return 0;
  }

  nbarray arr = std::get<nbarray>(q);
  size_t N = arr.size();
  if (N == 0) {
    return 0;
  }

  size_t M = arr[0].size();
  if (M == 0) {
    return 0;
  }

  size_t K = arr[0][0].size();

  return N * M * K;
}

struct query_t_to_py {
  double* my_data;

  query_t_to_py(double data[]) {
    my_data = data;
  }

  py_query_t operator()(const Parameter& v) const { 
    return v;
  }
  py_query_t operator()(const std::vector<Parameter>& v) const { 
    return v; 
  }
  py_query_t operator()(const nbarray &data) const {
    size_t N = data.size();
    if (N == 0) {
      return py_query_t();
    }

    size_t M = data[0].size();
    if (M == 0) {
      return py_query_t();
    }

    size_t K = data[0][0].size();

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < M; j++) {
        for (size_t k = 0; k < K; k++) {
          size_t findex = i*(M*K) + j*K + k;
          my_data[findex] = data[i][j][k];
        }
      }
    }

    nanobind::capsule owner(my_data, [](void *p) noexcept {
        delete[] (double *) p;
    });

    py_nbarray nb_data(my_data, {N, M, K}, owner);
    return py_query_t{nb_data};
  }
};

NB_MODULE(dataframe_bindings, m) {
  m.def("load_params", &utils::load_params);

  // Need to statically cast overloaded templated methods
  void (DataSlide::*ds_add_param1)(const ExperimentParams&) = &DataSlide::add_param;
  void (DataSlide::*ds_add_param2)(const std::string&, const Parameter&) = &DataSlide::add_param;

  void (DataSlide::*push_data1)(const std::string&, const double) = &DataSlide::push_samples_to_data;
  void (DataSlide::*push_data2)(const std::string&, const double, const double, const uint32_t) = &DataSlide::push_samples_to_data;
  void (DataSlide::*push_data3)(const std::string&, const std::vector<double>&) = &DataSlide::push_samples_to_data;
  void (DataSlide::*push_data4)(const std::string&, const std::vector<std::vector<double>>&, bool) = &DataSlide::push_samples_to_data;
  void (DataSlide::*push_data5)(const SampleMap&, bool) = &DataSlide::push_samples_to_data;

  void (DataSlide::*push_samples1)(const std::string&, const double) = &DataSlide::push_samples;
  void (DataSlide::*push_samples2)(const std::string&, const std::vector<double>&) = &DataSlide::push_samples;
  void (DataSlide::*push_samples3)(const std::string&, const std::vector<std::vector<double>>&) = &DataSlide::push_samples;
  void (DataSlide::*push_samples4)(const SampleMap&) = &DataSlide::push_samples;

  nanobind::class_<Sample>(m, "Sample")
    .def(nanobind::init<>())
    .def(nanobind::init<double>())
    .def(nanobind::init<double, double, uint32_t>())
    .def("__str__", &Sample::to_string)
    .def("get_mean", &Sample::get_mean)
    .def("get_std", &Sample::get_std)
    .def("get_num_samples", &Sample::get_num_samples)
    .def("set_mean", &Sample::set_mean)
    .def("set_std", &Sample::set_std)
    .def("set_num_samples", &Sample::set_num_samples);

  nanobind::class_<DataSlide>(m, "DataSlide")
    .def(nanobind::init<>())
    .def(nanobind::init<ExperimentParams&>())
    .def(nanobind::init<const std::string&>())
    .def(nanobind::init<const DataSlide&>())
    .def("__init__", [](DataSlide* t, const nanobind::bytes& bytes) {
      auto byte_vec = convert_bytes(bytes);
      new (t) DataSlide(byte_vec);
    })
    .def_rw("params", &DataSlide::params)
    .def_rw("data", &DataSlide::data)
    .def_rw("samples", &DataSlide::samples)
    .def("to_bytes", &DataSlide::to_bytes)
    .def("add_param", ds_add_param1)
    .def("add_param", ds_add_param2)
    .def("add_data", [](DataSlide& self, const std::string& s, size_t width) { self.add_data(s, width); }, "key"_a, "width"_a = 1)
    .def("add_data", [](DataSlide& self, const SampleMap& sample) { self.add_data(sample); })
    .def("push_samples_to_data", push_data1)
    .def("push_samples_to_data", push_data2)
    .def("push_samples_to_data", push_data3)
    .def("push_samples_to_data", push_data4, "key"_a, "data"_a, "avg"_a = false)
    .def("push_samples_to_data", push_data5, "data"_a, "avg"_a = false)
    .def("add_samples", [](DataSlide& self, const std::string& s, size_t width) { self.add_samples(s, width); }, "key"_a, "width"_a = 1)
    .def("add_samples", [](DataSlide& self, const SampleMap& sample) { self.add_samples(sample); })
    .def("combine", &DataSlide::combine, "other"_a, "atol"_a = DF_ATOL, "rtol"_a = DF_RTOL)
    .def("combine_data", &DataSlide::combine_data)
    .def("push_samples", push_samples1)
    .def("push_samples", push_samples2)
    .def("push_samples", push_samples3)
    .def("push_samples", push_samples4)
    .def("remove", &DataSlide::remove)
    .def("_get_buffer", [](const DataSlide& slide) {
      return convert_bytes(slide.buffer);
    })
    .def("__contains__", &DataSlide::contains)
    .def("__getitem__", &DataSlide::get_param)
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

  auto _query = [](DataFrame& df, const auto& keys, const ExperimentParams& constraints, bool unique, DataFrame::QueryType query_type) {
    std::vector<query_t> results = df.query(keys, constraints, unique, query_type);

    // Allocate space for query results
    size_t num_queries = results.size();
    std::vector<double*> datas(num_queries);
    for (uint32_t i = 0; i < num_queries; i++) {
      size_t query_size = get_query_size(results[i]);
      datas[i] = new double[query_size];
    }

    std::vector<py_query_t> py_results(num_queries);
    for (uint32_t i = 0; i < num_queries; i++) {
      py_results[i] = std::visit(query_t_to_py(datas[i]), results[i]);
    }

    if (num_queries == 1) {
      return py_query_result{py_results[0]};
    }

    return py_query_result{py_results};
  };

  nanobind::class_<DataFrame>(m, "DataFrame")
    .def(nanobind::init<>())
    .def(nanobind::init<double, double>())
    .def(nanobind::init<const std::vector<DataSlide>&>())
    .def(nanobind::init<const ExperimentParams&, const std::vector<DataSlide>&>())
    .def(nanobind::init<const std::string&>())
    .def(nanobind::init<const DataFrame&>())
    .def("__init__", [](DataFrame* t, const nanobind::bytes& bytes) {
      auto byte_vec = convert_bytes(bytes);
      new (t) DataFrame(byte_vec);
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
    .def("remove", &DataFrame::remove)
    .def("__contains__", &DataFrame::contains)
    .def("__getitem__", &DataFrame::get)
    .def("__setitem__", df_add_param2)
    .def("__str__", &DataFrame::to_json)
    .def("__add__", &DataFrame::combine)
    .def("__getstate__", [](const DataFrame& frame){ return convert_bytes(frame.to_bytes()); })
    .def("__setstate__", [](DataFrame& frame, const nanobind::bytes& bytes){ new (&frame) DataFrame(convert_bytes(bytes)); })
    .def("describe", &DataFrame::describe, "num_slides"_a = 0)
    .def("write", &DataFrame::write)
    .def("promote_params", &DataFrame::promote_params)
    .def("reduce", &DataFrame::reduce)
    .def("query", [&_query](DataFrame& df, const query_key_t& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::Mean);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::copy)
    .def("query_std", [&_query](DataFrame& df, const query_key_t& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::StandardDeviation);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::move)
    .def("query_sde", [&_query](DataFrame& df, const query_key_t& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::StandardError);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::move)
    .def("query_nsamples", [&_query](DataFrame& df, const query_key_t& keys, const ExperimentParams& constraints, bool unique) {
      return _query(df, keys, constraints, unique, DataFrame::QueryType::NumSamples);
    }, "keys"_a, "constraints"_a = ExperimentParams(), "unique"_a = false, nanobind::rv_policy::move);

}
