cmake_minimum_required(VERSION 3.21)

project(Triforce)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-DUSE_FLOAT -pg")

add_executable(Triforce triforce.cpp)

find_package(Kokkos REQUIRED)
# find_library(TestSo TestSo)
# find_library(Xyz Xyz REQUIRED HINTS src)
find_library(optixNSearch optixNSearch HINTS src)
# link_directories(./src)
target_link_libraries(Triforce Kokkos::kokkos)

target_include_directories(Triforce PRIVATE ./src)

# target_link_libraries(Triforce ${TestSo})
# target_link_libraries(Triforce ${xyz})
target_link_libraries(Triforce ${optixNSearch})