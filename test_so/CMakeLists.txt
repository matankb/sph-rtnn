cmake_minimum_required(VERSION 2.8.9)
project (TestSo)
set(CMAKE_CXX_STANDARD 11)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(TestSo SHARED test_so.h test_so.cpp)

install(TARGETS TestSo DESTINATION /usr/lib)
install(FILES test_so.h DESTINATION include)