#pragma once

#include <vector>
#include <string>
#include <map>
#include <variant>
#include <utility>
#include <optional>
#include <algorithm>
#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace dataframe {
  
using byte_t = char;

// --- DEFINING VALID PARAMETER VALUES ---
using Parameter = std::variant<std::string, int, double>;
using ExperimentParams = std::map<std::string, Parameter>;

template <typename T=double>
using ndarray = nanobind::ndarray<nanobind::numpy, T>;

// Stores shape, values, and optional sampling data (standard error, number of samples) for Gaussian error combination
using DataObject = std::tuple<
  ndarray<double>,
  std::optional<ndarray<double>>,
  std::optional<ndarray<size_t>>
>;

using SampleMap = std::map<std::string, DataObject>;

// --- DEFINING VALID QUERY RESULTS ---
// Options are:
// 1.) Frame-level param -> var_t
// 2.) Slide-level param -> std::vector<var_t>
// 3.) Vector data -> nbarray
using query_t = std::variant<Parameter, std::vector<Parameter>, ndarray<double>, ndarray<size_t>>;

}
