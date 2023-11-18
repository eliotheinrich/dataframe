#include "Frame.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>

using namespace nanobind::literals;
using namespace dataframe_utils;


// Provide this function to initialize dataframe in other projects
void init_dataframe(nanobind::module_ &m) {
	m.def("load_json", static_cast<std::vector<Params>(*)(const std::string&, bool)>(&load_json), "data"_a, "verbose"_a = false);
	m.def("write_config", &write_config);

	// Need to statically cast overloaded templated methods
	void (DataSlide::*ds_add_param1)(const Params&) = &DataSlide::add_param;
	void (DataSlide::*ds_add_param2)(const std::string&, var_t const&) = &DataSlide::add_param;

	void (DataSlide::*push_data1)(const std::string&, double) = &DataSlide::push_data;
	void (DataSlide::*push_data2)(const std::string&, double, double, uint32_t) = &DataSlide::push_data;

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
		.def("__str__", &DataFrame::to_string)
		.def("__add__", &DataFrame::combine)
		.def("write_json", &DataFrame::write_json)
		.def("promote_params", &DataFrame::promote_params)
		.def("filter", &DataFrame::filter)
		.def("query", &DataFrame::query, "keys"_a, "constraints"_a, "unique"_a = false, "error"_a = false);
	
	nanobind::class_<ParallelCompute>(m, "ParallelCompute")
		.def(nanobind::init<Params&, std::vector<std::shared_ptr<Config>>>())
		.def_rw("dataframe", &ParallelCompute::df)
		.def_rw("atol", &ParallelCompute::atol)
		.def_rw("rtol", &ParallelCompute::rtol)
		.def("compute", &ParallelCompute::compute, "verbose"_a = false)
		.def("write_json", &ParallelCompute::write_json)
		.def("write_serialize_json", &ParallelCompute::write_serialize_json);
}
