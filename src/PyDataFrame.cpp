#include "DataFrame.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;
using namespace nb::literals;

std::string write_config(const std::vector<Params>& params) {
	std::unordered_set<std::string> keys;
	std::map<std::string, std::unordered_set<var_t>> vals;
	for (auto const &p : params) {
		for (auto const &[k, v] : p) {
			keys.insert(k);
			if (!vals.count(k))
				vals[k] = std::unordered_set<var_t>();
			
			vals[k].insert(v);
		}
	}

	std::string s = "{\n";
	std::vector<std::string> buffer1;
	for (auto const &key : keys) {
		std::string b1 = "\t\"" + key + "\": ";
		if (vals[key].size() > 1) b1 += "[";

		std::vector<std::string> buffer2;
		std::vector<var_t> sorted_vals(vals[key].begin(), vals[key].end());
		std::sort(sorted_vals.begin(), sorted_vals.end());
		
		for (auto val : sorted_vals)
			buffer2.push_back(std::visit(var_t_to_string(), val));
		
		b1 += join(buffer2, ", ");
		if (vals[key].size() > 1) b1 += "]";
		buffer1.push_back(b1);
	}

	s += join(buffer1, ",\n");

	s += "\n}";

	return s;
}

// Provide this function to initialize dataframe in other projects
void init_dataframe(nb::module_ &m) {
	m.def("load_json", static_cast<std::vector<Params>(*)(const std::string&, bool)>(&load_json), "data"_a, "verbose"_a = false);
	m.def("write_config", &write_config);

	// Need to statically cast overloaded templated methods
	void (DataSlide::*ds_add_param1)(const Params&) = &DataSlide::add_param;
	void (DataSlide::*ds_add_param2)(const std::string&, var_t const&) = &DataSlide::add_param;

	void (DataSlide::*push_data1)(const std::string&, double) = &DataSlide::push_data;
	void (DataSlide::*push_data2)(const std::string&, double, double, uint32_t) = &DataSlide::push_data;

	nb::class_<DataSlide>(m, "DataSlide")
		.def(nb::init<>())
		.def(nb::init<Params&>())
		.def(nb::init<const std::string&>())
		.def(nb::init<const DataSlide&>())
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
		.def("__str__", &DataSlide::to_string, "indentation"_a = 0, "pretty"_a = true, "save_full_sample"_a = false)
		.def("congruent", &DataSlide::congruent)
		.def("combine", &DataSlide::combine);
	
	void (DataFrame::*df_add_param1)(const Params&) = &DataFrame::add_param;
	void (DataFrame::*df_add_param2)(const std::string&, var_t const&) = &DataFrame::add_param;
	void (DataFrame::*df_add_metadata1)(const Params&) = &DataFrame::add_metadata;
	void (DataFrame::*df_add_metadata2)(const std::string&, var_t const&) = &DataFrame::add_metadata;
	nb::class_<DataFrame>(m, "DataFrame")
		.def(nb::init<>())
		.def(nb::init<const std::vector<DataSlide>&>())
		.def(nb::init<const std::string&>())
		.def(nb::init<const DataFrame&>())
		.def_rw("params", &DataFrame::params)
		.def_rw("slides", &DataFrame::slides)
		.def("add_slide", &DataFrame::add_slide)
		.def("add_param", df_add_param1)
		.def("add_param", df_add_param2)
		.def("add_metadata", df_add_metadata1)
		.def("add_metadata", df_add_metadata2)
		.def("remove", &DataFrame::remove)
		.def("__contains__", &DataFrame::contains)
		.def("__getitem__", &DataFrame::get)
		.def("__str__", &DataFrame::to_string)
		.def("__add__", &DataFrame::combine)
		.def("write_json", &DataFrame::write_json)
		.def("promote_params", &DataFrame::promote_params)
		.def("query", &DataFrame::query, "keys"_a, "constraints"_a, "unique"_a = false);
	
	nb::class_<ParallelCompute>(m, "ParallelCompute")
		.def(nb::init<std::vector<std::shared_ptr<Config>>, uint32_t>(), "configs"_a = std::vector<std::shared_ptr<Config>>(), "num_threads"_a = 1)
		.def_rw("dataframe", &ParallelCompute::df)
		.def("compute", &ParallelCompute::compute, "verbose"_a = false)
		.def("write_json", &ParallelCompute::write_json);
}
