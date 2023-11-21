#include "Frame.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/ndarray.h>

// TODO remove this
using namespace nanobind::literals;

namespace dataframe {

typedef nanobind::ndarray<nanobind::numpy, double, nanobind::ndim<3>> py_nbarray;
typedef std::variant<var_t, std::vector<var_t>, py_nbarray> py_query_t;
typedef std::variant<py_query_t, std::vector<py_query_t>> py_query_result;

size_t get_query_size(const query_t& q) {
	if (q.index() != 2)
		return 0;

	nbarray arr = std::get<nbarray>(q);
	size_t N = arr.size();
	if (N == 0)
		return 0;

	size_t M = arr[0].size();
	if (M == 0)
		return 0;

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
		if (N == 0)
			return py_query_t();
		
		size_t M = data[0].size();
		if (M == 0)
			return py_query_t();

		size_t K = data[0][0].size();

		for (size_t i = 0; i < N; ++i) {
    		for (size_t j = 0; j < M; ++j) {
        		for (size_t k = 0; k < K; ++k) {
					size_t findex = i*(M*K) + j*K + k;
            		my_data[findex] = data[i][j][k];
        		}
    		}
		}

		//for (uint32_t i = 0; i < N*M*K; i++) {
		//	std::cout << my_data[i] << "\t";
		//} std::cout << "\n";




		py_nbarray nb_data(my_data, {N, M, K});
		//for (uint32_t i = 0; i < N; i++) {
		//	for (uint32_t j = 0; j < M; j++) {
		//		for (uint32_t k = 0; k < K; k++) {
		//			std::cout << nb_data(i, j, k) << "\t";
		//		}
		//	}
		//}
		//std::cout << "\n";
		return py_query_t{nb_data};
	}
};

//struct query_result_to_py {
//	nanobind::handle my_capsule;
//
//	query_result_to_py(nanobind::handle capsule) {
//		my_capsule = capsule;
//	}
//
//	py_query_result operator()(const query_t& result) const {
//		return std::visit(query_t_to_py(my_capsule), result);
//	}
//
//	py_query_result operator()(const std::vector<query_t>& results) const {
//		size_t N = results.size();
//		std::vector<py_query_t> py_results(N);
//
//		for (uint32_t i = 0; i < N; i++) {
//			std::cout << "i = " << i << std::endl;
//			py_results[i] = std::visit(query_t_to_py(my_capsule), results[i]);
//			if (py_results[i].index() == 2) {
//				std::cout << "query_t_to_python() gone; now result looks like: ";
//				py_nbarray nb_arr = std::get<py_nbarray>(py_results[i]);
//				auto v = nb_arr.view();
//				for (uint32_t i = 0; i < v.shape(0); i++) {
//					for (uint32_t j = 0; j < v.shape(1); j++) {
//						for (uint32_t k = 0; k < v.shape(2); k++) {
//							std::cout << v(i, j, k) << "\t";
//						}
//					}
//				}
//				std::cout << "\n";
//			}
//		}
//
//		return py_results;
//	}
//};
//
//py_query_result parse_query_result(const query_result& result, nanobind::handle capsule) {
//	return std::visit(query_result_to_py(capsule), result);
//}



// Provide this function to initialize dataframe in other projects
void init_dataframe(nanobind::module_ &m) {
	m.def("load_json", static_cast<std::vector<Params>(*)(const std::string&, bool)>(&utils::load_json), "data"_a, "verbose"_a = false);
	m.def("write_config", &utils::write_config);

	// Need to statically cast overloaded templated methods
	void (DataSlide::*ds_add_param1)(const Params&) = &DataSlide::add_param;
	void (DataSlide::*ds_add_param2)(const std::string&, var_t const&) = &DataSlide::add_param;

	void (DataSlide::*push_data1)(const std::string&, const Sample&) = &DataSlide::push_data;
	void (DataSlide::*push_data2)(const std::string&, const std::vector<Sample>&) = &DataSlide::push_data;

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
		.def("remove", &DataSlide::remove)
		.def("__contains__", &DataSlide::contains)
		.def("__getitem__", &DataSlide::get_param)
		.def("__setitem__", ds_add_param2)
		.def("__str__", &DataSlide::to_string, "indentation"_a = 0, "pretty"_a = true, "save_full_sample"_a = false)
		.def("congruent", &DataSlide::congruent)
		.def("combine", &DataSlide::combine);
	
	void (DataFrame::*df_add_param1)(const Params&) = &DataFrame::add_param;
	void (DataFrame::*df_add_param2)(const std::string&, var_t const&) = &DataFrame::add_param;
	void (DataFrame::*df_add_metadata1)(const Params&) = &DataFrame::add_metadata;
	void (DataFrame::*df_add_metadata2)(const std::string&, var_t const&) = &DataFrame::add_metadata;

	//auto py_query = [](DataFrame& df, const std::vector<std::string>& keys, const Params& constraints, bool unique, bool error, nanobind::handle capsule) {
	//	return parse_query_result(df.query(keys, constraints, unique, error), capsule);
	//};


	nanobind::class_<DataFrame>(m, "DataFrame")
		.def(nanobind::init<>())
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
		.def("__str__", &DataFrame::to_string, "record_error"_a = true)
		.def("__add__", &DataFrame::combine)
		.def("write_json", &DataFrame::write_json)
		.def("promote_params", &DataFrame::promote_params)
		.def("filter", &DataFrame::filter)
		//.def("query", &DataFrame::query, "keys"_a, "constraints"_a, "unique"_a = false, "error"_a = false); // DO CASTING HERE
		.def("query", [](DataFrame& df, const std::vector<std::string>& keys, const Params& constraints, bool unique, bool error) {
			std::vector<query_t> results = df.query(keys, constraints, unique, error);

			size_t num_queries = results.size();
			std::vector<double*> datas(num_queries);
			for (uint32_t i = 0; i < num_queries; i++) {
				size_t query_size = get_query_size(results[i]);
				datas[i] = new double[query_size];
			}
		
			std::vector<py_query_t> py_results(num_queries);
			for (uint32_t i = 0; i < num_queries; i++)
				py_results[i] = std::visit(query_t_to_py(datas[i]), results[i]);

			if (num_queries == 1)
				return py_query_result{py_results[0]};
			else
				return py_query_result{py_results};
		}, nanobind::rv_policy::move);
	
	nanobind::class_<ParallelCompute>(m, "ParallelCompute")
		.def(nanobind::init<Params&, std::vector<std::shared_ptr<Config>>>())
		.def_rw("dataframe", &ParallelCompute::df)
		.def_rw("atol", &ParallelCompute::atol)
		.def_rw("rtol", &ParallelCompute::rtol)
		.def("compute", &ParallelCompute::compute, "verbose"_a = false)
		.def("write_json", &ParallelCompute::write_json)
		.def("write_serialize_json", &ParallelCompute::write_serialize_json);
}

}