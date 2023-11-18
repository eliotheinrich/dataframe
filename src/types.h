#pragma once

#include <vector>
#include <string>
#include <map>
#include <variant>
#include <utility>

#define THREADPOOL 0
#define OPENMP 1
#define SERIAL 2

enum parallelization_t {
	threadpool,
	openmp,
	serial,
};

// --- DEFINING VALID PARAMETER VALUES ---
class Sample;
typedef std::variant<int, double, std::string> var_t;
typedef std::map<std::string, Sample> data_t;
typedef std::map<std::string, var_t> Params;

// --- DEFINING VALID QUERY RESULTS ---
typedef std::variant<var_t, std::vector<var_t>, std::vector<std::vector<double>>> query_t;
typedef std::variant<query_t, std::vector<query_t>> query_result;



// --- COMPUTE RESULT USED BY PARALLElCOMPUTE --- //
class DataSlide;
typedef std::pair<DataSlide, std::optional<std::string>> compute_result_t;
