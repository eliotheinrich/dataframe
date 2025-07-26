cmake_minimum_required(VERSION 3.26)

project(dataframe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
string(REPLACE ":" ";" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Fetching externals
find_package(fmt)
add_compile_definitions(-DFMT_HEADER_ONLY=On)

foreach(PATH IN LISTS CMAKE_PREFIX_PATH)
    include_directories("${PATH}/include")
endforeach()

include(FetchContent)

FetchContent_Declare(
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG cc4ce0efe5133c23ec55f07d9e753f7dd1abb0b8
)

FetchContent_MakeAvailable(glaze)

include_directories(
    src
    ${glaze_SOURCE_DIR}/include
)

set(SRC_FILES 
    src/utils.hpp 
    src/utils.cpp
    src/Sample.hpp 
    src/DataSlide.hpp 
    src/DataFrame.hpp 
    src/DataFrame.cpp
    src/types.h
)

if (DEFINED DF_BUILDING_PY_BINDINGS)
    nanobind_add_module(
        dataframe_bindings 
        STABLE_ABI
        NB_STATIC
        ${SRC_FILES}
        src/PyDataFrame.hpp
        src/PyDataFrame.cpp
    )

    install(
        TARGETS dataframe_bindings
        LIBRARY DESTINATION dataframe
    )
elseif (DEFINED DF_BUILDING_BINARIES)
    add_executable(test src/test.cpp)
    target_link_libraries(test PRIVATE dataframe)
endif()

