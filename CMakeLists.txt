# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 16:28
#  
#  Description: GraceQ/tensor project. Main CMake file.
# 
cmake_minimum_required(VERSION 3.12)

project("GraceQ/tensor"
  DESCRIPTION "Low level sparse tensor library with U1 symmetry."
  LANGUAGES CXX)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()


set(CMAKE_CXX_STANDARD 11)

add_definitions("-g")


add_subdirectory(src lib)


# Unittests
enable_testing()
add_subdirectory(tests tests)
