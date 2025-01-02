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

using SampleMap = std::map<std::string, std::vector<std::vector<double>>>;

// --- DEFINING VALID QUERY RESULTS ---
// Options are:
// 1.) Frame-level param -> var_t
// 2.) Slide-level param -> std::vector<var_t>
// 3.) Vector data -> nbarray
using nbarray = std::vector<std::vector<std::vector<double>>>;
using query_t = std::variant<Parameter, std::vector<Parameter>, nbarray>;
using query_key_t = std::variant<std::string, std::vector<std::string>>;

}
