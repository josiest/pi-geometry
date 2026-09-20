cmake_minimum_required(VERSION 4.4)
project(pi-affine-transforms VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(pi-affine-transforms INTERFACE)
target_sources(pi-affine-transforms
    INTERFACE FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/pi/geometry/affine_transforms.hpp
        include/pi/geometry/vectors.hpp
        include/pi/geometry/linear_algebra.hpp
        include/pi/geometry.hpp)

set_target_properties(pi-affine-transforms PROPERTIES
    SOVERSION 0
    EXPORT_NAME affine-transforms)

include(pi-cmake-library-helpers/PiLibraryHelpers.cmake)
install_pi_package(affine-transforms)