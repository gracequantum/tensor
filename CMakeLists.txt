#  SPDX-License-Identifier: LGPL-3.0-only
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

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

option(GQTEN_USE_MKL_GEMM_BATCH "Use MKL's cblas_dgemm_batch function." ON)
if(GQTEN_USE_MKL_GEMM_BATCH)
  add_definitions(-DGQTEN_USE_MKL_GEMM_BATCH)
endif()


set(CMAKE_CXX_STANDARD 11)

add_definitions("-g")

set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
set(MATH_LIB_LINK_FLAGS "-L$ENV{MKLROOT}/lib -Wl,-rpath,$ENV{MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")

add_subdirectory(external)

add_subdirectory(src lib)


# Unittests
enable_testing()
add_subdirectory(tests tests)
