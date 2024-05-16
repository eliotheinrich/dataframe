#include "Frame.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/trampoline.h>
#include <nanobind/ndarray.h>

#define EXPORT_SIMULATOR_DRIVER(A)                                                   \
  nanobind::class_<dataframe::TimeSamplingDriver<A>>(m, #A)                          \
  .def(nanobind::init<dataframe::Params&>())                                         \
  .def_rw("params", &dataframe::TimeSamplingDriver<A>::params)                       \
  .def("generate_dataslide", &dataframe::TimeSamplingDriver<A>::generate_dataslide);

#define INIT_CONFIG()                                \
  nanobind::class_<dataframe::Config>(m, "Config")   \
  .def(nanobind::init<dataframe::Params&>())         \
  .def_rw("params", &dataframe::Config::params)      \
  .def("get_nruns", &dataframe::Config::get_nruns);

#define EXPORT_CONFIG(A)                                              \
  nanobind::class_<A, dataframe::Config>(m, #A)                       \
  .def(nanobind::init<dataframe::Params&>())                          \
  .def("compute", &A::compute)                                        \
  .def("clone", &A::clone)                                            \
  .def("__getstate__", [](const A& config) { return config.params; }) \
  .def("__setstate__", [](A& config, dataframe::Params& params){ new (&config) A(params); } )



using namespace nanobind::literals;

namespace dataframe {
  // Types which are interfaced with python
  typedef nanobind::ndarray<nanobind::numpy, double> py_nbarray;
  typedef std::variant<qvar_t, std::vector<qvar_t>, py_nbarray> py_query_t;
  typedef std::variant<py_query_t, std::vector<py_query_t>> py_query_result;

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

    py_query_t operator()(const qvar_t& v) const { 
      return v;
    }
    py_query_t operator()(const std::vector<qvar_t>& v) const { 
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

      py_nbarray nb_data(my_data, {N, M, K});
      return py_query_t{nb_data};
    }
  };

  // Provide this function to initialize dataframe in other projects
  void init_dataframe(nanobind::module_ &m) {
    m.def("load_params", &utils::load_params);

    // Need to statically cast overloaded templated methods
    void (DataSlide::*ds_add_param1)(const Params&) = &DataSlide::add_param;
    void (DataSlide::*ds_add_param2)(const std::string&, var_t const&) = &DataSlide::add_param;

    void (DataSlide::*push_data1)(const std::string&, const double) = &DataSlide::push_samples_to_data;
    void (DataSlide::*push_data2)(const std::string&, const double, const double, const uint32_t) = &DataSlide::push_samples_to_data;
    void (DataSlide::*push_data3)(const std::string&, const std::vector<Sample>&) = &DataSlide::push_samples_to_data;

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
      .def(nanobind::init<Params&>())
      .def(nanobind::init<const std::string&>())
      .def(nanobind::init<const DataSlide&>())
      .def_rw("params", &DataSlide::params)
      .def_rw("data", &DataSlide::data)
      .def_rw("samples", &DataSlide::samples)
      .def("add_param", ds_add_param1)
      .def("add_param", ds_add_param2)
      .def("add_data", [](DataSlide& self, const std::string& s) { self.add_data(s); })
      .def("push_samples_to_data", push_data1)
      .def("push_samples_to_data", push_data2)
      .def("push_samples_to_data", push_data3)
      .def("remove", &DataSlide::remove)
      .def("__contains__", &DataSlide::contains)
      .def("__getitem__", &DataSlide::get_param)
      .def("__setitem__", ds_add_param2)
      .def("__str__", &DataSlide::to_string)
      .def("__getstate__", [](const DataSlide& slide){ return slide.to_string(); })
      .def("__setstate__", [](DataSlide& slide, const std::string& s){ new (&slide) DataSlide(s); })
      .def("describe", &DataSlide::describe)
      .def("congruent", &DataSlide::congruent)
      .def("combine", &DataSlide::combine, "other"_a, "atol"_a = DF_ATOL, "rtol"_a = DF_RTOL);

    void (DataFrame::*df_add_param1)(const Params&) = &DataFrame::add_param;
    void (DataFrame::*df_add_param2)(const std::string&, var_t const&) = &DataFrame::add_param;
    void (DataFrame::*df_add_metadata1)(const Params&) = &DataFrame::add_metadata;
    void (DataFrame::*df_add_metadata2)(const std::string&, var_t const&) = &DataFrame::add_metadata;

    nanobind::class_<DataFrame>(m, "DataFrame")
      .def(nanobind::init<>())
      .def(nanobind::init<double, double>())
      .def(nanobind::init<const std::vector<DataSlide>&>())
      .def(nanobind::init<const Params&, const std::vector<DataSlide>&>())
      .def(nanobind::init<const std::string&>())
      .def(nanobind::init<const DataFrame&>())
      .def(nanobind::init<const std::vector<uint8_t>&>())
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
      .def("__getstate__", [](const DataFrame& frame){ return frame.to_string(); })
      .def("__setstate__", [](DataFrame& frame, const std::string& s){ new (&frame) DataFrame(s); })
      .def("describe", &DataFrame::describe)
      .def("write", &DataFrame::write)
      .def("promote_params", &DataFrame::promote_params)
      .def("reduce", &DataFrame::reduce)
      .def("average_samples", &DataFrame::average_samples_inplace)
      .def("filter", &DataFrame::filter, "constraints"_a, "filter"_a = false)
      .def("query", [](DataFrame& df, const DataFrame::query_key_t& keys, const Params& constraints, bool unique, bool error, bool num_samples) {
          std::vector<query_t> results = df.query(keys, constraints, unique, error, num_samples);

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
          } else {
            return py_query_result{py_results};
          }
        }, "keys"_a, "constraints"_a = Params(), "unique"_a = false, "error"_a = false, "num_samples"_a = false, nanobind::rv_policy::move);
  }

}
