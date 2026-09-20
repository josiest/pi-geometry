cmake_minimum_required(VERSION 4.4)
project(pi-geometry VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(pi-geometry INTERFACE)
target_sources(pi-geometry
    INTERFACE FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/pi/geometry/affine_transforms.hpp
        include/pi/geometry/vectors.hpp
        include/pi/geometry/linear_algebra.hpp
        include/pi/geometry.hpp)

target_compile_options(pi-geometry INTERFACE -lstdc++exp)

set_target_properties(pi-geometry PROPERTIES
    SOVERSION 0
    EXPORT_NAME geometry)

include(pi-cmake-library-helpers/PiLibraryHelpers.cmake)
install_pi_package(geometry)