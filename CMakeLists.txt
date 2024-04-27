cmake_minimum_required(VERSION 3.27)

project(dataframe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

if (TARGET dataframe)
    return()
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Fetching externals
include(FetchContent)

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG d907f9cfb7197a29fa7cbefab310895593f5b105
)

FetchContent_Declare(
  nlohmann
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG master
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(glaze nlohmann)

include_directories(
  src
  ${glaze_SOURCE_DIR}/include
  ${nlohmann_SOURCE_DIR}/include
)

set(SRC_FILES 
  src/utils.hpp 
  src/utils.cpp
  src/Config.hpp
  src/Sample.hpp 
  src/DataSlide.hpp 
  src/DataFrame.hpp 
  src/DataFrame.cpp
  src/Simulator.hpp
  src/TimeSamplingDriver.hpp
  src/types.h
)

add_library(dataframe ${SRC_FILES}) 

if (DEFINED DF_BUILDING_PY_BINDINGS)
  find_package(Python 3.10 
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
  )

  # Finding nanobind
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(
    dataframe_bindings 
    STABLE_ABI
    NB_STATIC
    ${SRC_FILES}
    src/PyDataFrame.hpp
    src/PyDataFrameModule.cpp
  )

  install(
    TARGETS dataframe_bindings
    LIBRARY DESTINATION dataframe
  )
else()
    add_executable(test src/test.cpp)
    target_link_libraries(test PRIVATE dataframe)
endif()

