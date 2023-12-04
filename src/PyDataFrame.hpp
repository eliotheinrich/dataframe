#include "Frame.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/trampoline.h>
#include <nanobind/ndarray.h>

using namespace nanobind::literals;

namespace dataframe {

typedef nanobind::ndarray<nanobind::numpy, double> py_nbarray;
typedef std::variant<var_t, std::vector<var_t>, py_nbarray> py_query_t;
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
	
    py_query_t operator()(const var_t& v) const { 
		return v;
	}
    py_query_t operator()(const std::vector<var_t>& v) const { 
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
	m.def("parse_config", static_cast<std::vector<Params>(*)(const std::string&, bool)>(&utils::parse_config), "data"_a, "verbose"_a = false);
	m.def("paramset_to_string", &utils::paramset_to_string);

	// Need to statically cast overloaded templated methods
	void (DataSlide::*ds_add_param1)(const Params&) = &DataSlide::add_param;
	void (DataSlide::*ds_add_param2)(const std::string&, var_t const&) = &DataSlide::add_param;

	void (DataSlide::*push_data1)(const std::string&, const double) = &DataSlide::push_data;
	void (DataSlide::*push_data2)(const std::string&, const double, const double, const uint32_t) = &DataSlide::push_data;
	void (DataSlide::*push_data3)(const std::string&, const std::vector<Sample>&) = &DataSlide::push_data;

	nanobind::class_<DataSlide>(m, "DataSlide")
		.def(nanobind::init<>())
		.def(nanobind::init<Params&>())
		.def(nanobind::init<const std::string&>())
		.def(nanobind::init<const DataSlide&>())
		.def_rw("params", &DataSlide::params)
		.def_rw("data", &DataSlide::data)
		.def("add_param", ds_add_param1)
		.def("add_param", ds_add_param2)
		.def("add_data", &DataSlide::add_data)
		.def("push_data", push_data1)
		.def("push_data", push_data2)
		.def("push_data", push_data3)
		.def("remove", &DataSlide::remove)
		.def("__contains__", &DataSlide::contains)
		.def("__getitem__", &DataSlide::get_param)
		.def("__setitem__", ds_add_param2)
		.def("__str__", &DataSlide::to_string)
		.def("__getstate__", [](const DataSlide& slide){ return slide.to_string(0, false, true); })
		.def("__setstate__", [](DataSlide& slide, const std::string& s){ new (&slide) DataSlide(s); })
		.def("congruent", &DataSlide::congruent)
		.def("combine", &DataSlide::combine, "other"_a, "atol"_a = ATOL, "rtol"_a = RTOL);
	
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
		.def_rw("params", &DataFrame::params)
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
		.def("__str__", &DataFrame::to_string, "write_error"_a = true)
		.def("__add__", &DataFrame::combine)
		.def("__getstate__", [](const DataFrame& frame){ return frame.to_string(true); })
		.def("__setstate__", [](DataFrame& frame, const std::string& s){ new (&frame) DataFrame(s); })
		.def("write_json", &DataFrame::write_json)
		.def("promote_params", &DataFrame::promote_params)
		.def("reduce", &DataFrame::reduce)
		.def("filter", &DataFrame::filter)
		.def("query", [](DataFrame& df, const std::vector<std::string>& keys, const Params& constraints, bool unique, bool error) {
			std::vector<query_t> results = df.query(keys, constraints, unique, error);

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
		}, nanobind::rv_policy::move);
}

}