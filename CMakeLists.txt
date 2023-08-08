cmake_minimum_required(VERSION 3.12)

project(dataframe)

find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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

nanobind_add_module(dataframe_bindings src/PyDataFrameModule.cpp)

if (CALL_FROM_SETUP_PY)
  set(DATAFRAME_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
else()
  set(DATAFRAME_INSTALL_PREFIX ${Python_SITELIB})
endif()

install(
  TARGETS dataframe_bindings
  LIBRARY DESTINATION ${DATAFRAME_INSTALL_PREFIX}
  ARCHIVE DESTINATION ${DATAFRAME_INSTALL_PREFIX}
  RUNTIME DESTINATION ${DATAFRAME_INSTALL_PREFIX}
)