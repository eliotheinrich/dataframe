#include "Frame.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <types.h>

#include <iostream>
#include <vector>
#include <type_traits>
#include <stdexcept>

// --- Type trait to detect std::vector ---
template<typename T>
struct is_std_vector : std::false_type {};

template<typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

// --- Get base type of nested vector ---
template<typename T>
struct get_base_type {
  using type = T;
};

template<typename T>
struct get_base_type<std::vector<T>> {
  using type = typename get_base_type<T>::type;
};

template <typename T>
struct vector_depth {
  static constexpr int value = 0; 
};


template <typename T>
struct vector_depth<std::vector<T>> {
  static constexpr int value = vector_depth<T>::value + 1; 
};

// --- Flatten recursive function ---
template<typename T, typename U = typename get_base_type<T>::type>
void flatten_recursive(const T& input, U* out, size_t& k) {
  if constexpr (is_std_vector<T>::value) {
    for (const auto& sub : input) {
      flatten_recursive(sub, out, k);
    }
  } else {
    out[k++] = input;
  }
}

template <typename T>
void extract_shape_recursive(const T& val, size_t* shape, size_t dim) {
  shape[dim] = val.size();

  if constexpr (vector_depth<T>::value > 1) {
    if (val.empty()) {
      return;
    }

    const auto& first = val[0];

    for (const auto& v : val) {
      if (v.size() != first.size()) {
        throw std::runtime_error(fmt::format("Ragged array detected at dimension {}", dim + 1));
      }
    }

    extract_shape_recursive(first, shape, dim + 1);
  }
}

template <typename T>
auto extract_shape(const T& val) {
  size_t dim = vector_depth<T>::value;
  size_t* shape = new size_t[dim];
  extract_shape_recursive(val, shape, 0);
  return std::make_tuple(shape, dim);
}

// --- Public API ---
template<typename T>
auto flatten_nd_vector(const T& input) {
  using Scalar = typename get_base_type<T>::type;

  auto [shape, dim] = extract_shape(input);
  size_t length = 1;
  for (size_t n = 0; n < dim; n++) {
    length *= shape[n];
  }

  Scalar* data = new Scalar[length];
  size_t i = 0;

  flatten_recursive(input, data, i);

  return std::make_tuple(data, shape, dim); 
}

template <typename T, size_t D, size_t ... Idx>
std::initializer_list<T> make_init_list(const std::array<T, D>& arr, std::index_sequence<Idx...>) {
  return {arr[Idx] ... };
}

template <typename T>
nanobind::ndarray<nanobind::numpy, double> to_nbarray(const T& vector) {
  auto [data, shape, dim] = flatten_nd_vector(vector);

  nanobind::capsule owner(data, [](void *p) noexcept {
    delete[] (double *) p;
  });

  return nanobind::ndarray<nanobind::numpy, double>(data, dim, shape, owner);
}

nanobind::bytes convert_bytes(const std::vector<dataframe::byte_t>& bytes) {
  nanobind::bytes nb_bytes(bytes.data(), bytes.size());
  return nb_bytes;
}

std::vector<dataframe::byte_t> convert_bytes(const nanobind::bytes& bytes) {
  std::vector<dataframe::byte_t> bytes_vec(bytes.c_str(), bytes.c_str() + bytes.size());
  bytes_vec.push_back('\0');
  return bytes_vec;
}

