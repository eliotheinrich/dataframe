cmake_minimum_required(VERSION 3.27)

project(dataframe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

if (DEFINED DF_MAIN_BUILDING)
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
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG main
    GIT_SHALLOW TRUE
  )

  FetchContent_MakeAvailable(glaze)

  include_directories(
      src
      ${glaze_SOURCE_DIR}/include
  )

  set(SRC_FILES 
    src/utils.hpp 
    src/Config.hpp
    src/Sample.hpp 
    src/DataSlide.hpp 
    src/DataFrame.hpp 
    src/Simulator.hpp
    src/TimeSamplingDriver.hpp
    src/PyDataFrame.hpp
    src/PyDataFrameModule.cpp
    src/types.h
  )

  nanobind_add_module(
    dataframe_bindings 
    STABLE_ABI
    NB_STATIC
    ${SRC_FILES}
  )

  install(
    TARGETS dataframe_bindings
    LIBRARY DESTINATION dataframe
  )

endif()

include(FetchContent)

FetchContent_Declare(
  glaze
  GIT_REPOSITORY https://github.com/stephenberry/glaze.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(glaze)

include_directories(
    src
    ${glaze_SOURCE_DIR}/include
)

add_executable(test src/test.cpp)

