#include "Frame.h"

#include <nanobind/nanobind.h>
#include <types.h>

nanobind::bytes convert_bytes(const std::vector<dataframe::byte_t>& bytes) {
  nanobind::bytes nb_bytes(bytes.data(), bytes.size());
  return nb_bytes;
}

std::vector<dataframe::byte_t> convert_bytes(const nanobind::bytes& bytes) {
  std::vector<dataframe::byte_t> bytes_vec(bytes.c_str(), bytes.c_str() + bytes.size());
  bytes_vec.push_back('\0');
  return bytes_vec;
}

