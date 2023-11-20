#pragma once

//#ifdef PS_BUILDING_PYTHON
//#include <nanobind/nanobind.h>
//#include <nanobind/stl/string.h>
//#include <nanobind/stl/variant.h>
//#include <nanobind/stl/vector.h>
//#include <nanobind/stl/map.h>
//#include <nanobind/stl/shared_ptr.h>
//#include <nanobind/ndarray.h>
//#endif


#include <vector>
#include <string>
#include <map>
#include <variant>
#include <utility>
#include <algorithm>

#define THREADPOOL 0
#define OPENMP 1
#define SERIAL 2

enum parallelization_t {
	threadpool,
	openmp,
	serial,
};

// --- DEFINING VALID PARAMETER VALUES ---
typedef std::variant<int, double, std::string> var_t;

// Need to be provide overloaded constructors so that the Sample interface is hidden and
// data can be provided as simple doubles or vectors of doubles
typedef std::map<std::string, var_t> Params;

// --- DEFINING VALID QUERY RESULTS ---
// Options are:
// 1.) Frame-level param -> var_t
// 2.) Slide-level param -> std::vector<var_t>
// 3.) Vector data -> nbarray
//#ifdef PS_BUILDING_PYTHON
//typedef nanobind::ndarray<nanobind::numpy, double, nanobind::ndim<3>> nbarray;
//#else
//typedef std::vector<std::vector<std::vector<double>>> nbarray;
//#endif
typedef std::vector<std::vector<std::vector<double>>> nbarray;
typedef std::variant<var_t, std::vector<var_t>, nbarray> query_t;
//typedef std::variant<query_t, std::vector<query_t>> query_result;

// --- COMPUTE RESULT USED BY PARALLElCOMPUTE --- //
class DataSlide;
typedef std::pair<DataSlide, std::optional<std::string>> compute_result_t;
