#include <chrono>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <iostream>

#include <Frame.h>

using namespace dataframe;
using namespace dataframe::utils;

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define ASSERT(...) GET_MACRO(__VA_ARGS__, ASSERT_TWO_ARGS, ASSERT_ONE_ARG)(__VA_ARGS__)

#define ASSERT_ONE_ARG(x) if (!(x)) { return false; }

#define ASSERT_TWO_ARGS(x, y) \
  if (!(x)) {                 \
    std::cout << y << "\n";   \
    return false;             \
  }

template <typename T, typename V>
bool is_close_eps(double eps, T first, V second) {
  return std::abs(first - second) < eps;
}

template <typename T, typename V>
bool is_close(T first, V second) {
  return is_close_eps(1e-8, first, second);
}

template <typename T, typename V, typename... Args>
bool is_close_eps(double eps, T first, V second, Args... args) {
  if (!is_close_eps(eps, first, second)) {
    return false;
  } else {
    return is_close_eps(eps, first, args...);
  }
}

template <typename T, typename V, typename... Args>
bool is_close(T first, V second, Args... args) {
  if (!is_close(first, second)) {
    return false;
  } else {
    return is_close(first, args...);
  }
}

bool test_slide() {
  DataSlide slide;
  std::vector<double> test_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<size_t> shape = {2, 3};
  slide.add_data("test", test_data, shape);

  return true;
}


using TestResult = std::tuple<bool, int>;

#define ADD_TEST(x)                                                               \
if (run_all || test_names.contains(#x)) {                                         \
  auto start = std::chrono::high_resolution_clock::now();                         \
  bool passed = x();                                                              \
  auto stop = std::chrono::high_resolution_clock::now();                          \
  int duration = duration_cast<std::chrono::microseconds>(stop - start).count();  \
  tests[#x "()"] = std::make_tuple(passed, duration);                             \
}                                                                                 \

int main(int argc, char *argv[]) {
  std::map<std::string, TestResult> tests;
  std::set<std::string> test_names;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      test_names.insert(argv[i]);
    }
  }


  ADD_TEST(test_slide);

  constexpr char green[] = "\033[1;32m";
  constexpr char black[] = "\033[0m";
  constexpr char red[] = "\033[1;31m";

  auto test_passed_str = [&](bool passed) {
    std::stringstream stream;
    if (passed) {
      stream << green << "PASSED" << black;
    } else {
      stream << red << "FAILED" << black;
    }
    
    return stream.str();
  };

  if (tests.size() == 0) {
    std::cout << "No tests to run.\n";
  } else {
    double total_duration = 0.0;
    for (const auto& [name, result] : tests) {
      auto [passed, duration] = result;
      std::cout << fmt::format("{:>40}: {} ({:.2f} seconds)\n", name, test_passed_str(passed), duration/1e6);
      total_duration += duration;
    }

    std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
  }
}
