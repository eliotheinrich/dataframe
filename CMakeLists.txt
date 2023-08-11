cmake_minimum_required(VERSION 3.15..3.26)

project(dataframe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Python 3.10 
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Finding nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Fetching externals
include(FetchContent)

FetchContent_Declare(
    nlohmann
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    bspl
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(nlohmann bspl)

include_directories(
    src
    ${nlohmann_SOURCE_DIR}/include 
    ${bspl_SOURCE_DIR}/include 
)

nanobind_add_module(
  dataframe_bindings 
  STABLE_ABI
  NB_STATIC
  src/PyDataFrameModule.cpp
)

install(
  TARGETS dataframe_bindings
  LIBRARY DESTINATION dataframe
)