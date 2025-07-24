#pragma once

#include <vector>
#include <string>
#include <map>
#include <variant>
#include <utility>
#include <optional>
#include <algorithm>
#include <cstdint>

namespace dataframe {
  
using byte_t = char;

// --- DEFINING VALID PARAMETER VALUES ---
using Parameter = std::variant<std::string, int, double>;
using ExperimentParams = std::map<std::string, Parameter>;

// --- DEFINING VALID QUERY RESULTS ---
// Options are:
// 1.) Frame-level param -> var_t
// 2.) Slide-level param -> std::vector<var_t>
// 3.) Vector data -> nbarray

template <typename T=double>
using ndarray = std::pair<std::vector<size_t>, std::vector<T>>;
using query_t = std::variant<Parameter, std::vector<Parameter>, ndarray<double>, ndarray<size_t>>;


// Stores shape, values, and optional sampling data (standard error, number of samples) for Gaussian error combination
struct SamplingData {
  std::vector<double> std;
  std::vector<size_t> nsamples;
};

struct DataObject {
  std::vector<size_t> shape;
  std::vector<double> values;
  std::optional<SamplingData> sampling_data;
};

using SampleMap = std::map<std::string, DataObject>;

}
