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
  
typedef uint8_t byte_t;

// --- DEFINING VALID PARAMETER VALUES ---
typedef std::variant<double, std::string> var_t;

// Need to be provide overloaded constructors so that the Sample interface is hidden and
// data can be provided as simple doubles or vectors of doubles
typedef std::map<std::string, var_t> Params;

// --- DEFINING VALID QUERY RESULTS ---
// Options are:
// 1.) Frame-level param -> var_t
// 2.) Slide-level param -> std::vector<var_t>
// 3.) Vector data -> nbarray
typedef std::vector<std::vector<std::vector<double>>> nbarray;
typedef std::variant<std::string, int, double> qvar_t;
typedef std::variant<qvar_t, std::vector<qvar_t>, nbarray> query_t;

}
