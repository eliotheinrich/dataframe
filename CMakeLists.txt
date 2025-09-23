cmake_minimum_required(VERSION 3.26)

project(dataframe VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Fetching externals
find_package(fmt REQUIRED)

include(FetchContent)

FetchContent_Declare(
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG cc4ce0efe5133c23ec55f07d9e753f7dd1abb0b8
)

FetchContent_MakeAvailable(glaze)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
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

add_library(dataframe ${SRC_FILES}) 
target_link_libraries(dataframe PUBLIC fmt::fmt)

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
        src/PyDataFrame.cpp
    )

    install(
        TARGETS dataframe_bindings
        LIBRARY DESTINATION dataframe
    )

    target_link_libraries(dataframe_bindings PUBLIC fmt::fmt)
else()
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/dataframeConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_LIST_DIR}/cmake/dataframeConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/dataframeConfig.cmake"
        INSTALL_DESTINATION lib/cmake/dataframe
    )

    install(TARGETS dataframe
            EXPORT dataframeTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin
    )

    install(EXPORT dataframeTargets
        FILE dataframeTargets.cmake
        NAMESPACE dataframe::
        DESTINATION lib/cmake/dataframe
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/dataframeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/dataframeConfigVersion.cmake"
        DESTINATION lib/cmake/dataframe
    )

    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
        DESTINATION include/dataframe
        FILES_MATCHING 
        PATTERN "*.h"
        PATTERN "*.hpp"
    )
endif()
