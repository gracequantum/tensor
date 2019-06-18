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


# Build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()


# Advanced options.
option(GQTEN_USE_MKL_GEMM_BATCH "Use MKL's cblas_dgemm_batch function." OFF)
if(GQTEN_USE_MKL_GEMM_BATCH)
  add_definitions(-DGQTEN_USE_MKL_GEMM_BATCH)
endif()

option(GQTEN_USE_OMP_GEMM_BATCH "Explictly control threading for gemm batch." OFF)
if(GQTEN_USE_OMP_GEMM_BATCH)
  add_definitions(-DGQTEN_USE_OMP_GEMM_BATCH)
endif()

option(GQTEN_TIMING_MODE "Timing mode." OFF)
if(GQTEN_TIMING_MODE)
  add_definitions(-DGQTEN_TIMING_MODE)
endif()

option(GQTEN_CONTRACT_BLOCK_COUNTING "Count the number of blocks when contracting tensors." OFF)
if(GQTEN_CONTRACT_BLOCK_COUNTING)
  add_definitions(-DGQTEN_CONTRACT_BLOCK_COUNTING)
endif()

option(GQTEN_BUILD_UNITTEST "Build unittests for GraceQ/tensor." OFF)


# Compilation and linking control.
set(CMAKE_CXX_STANDARD 11)
add_definitions("-g")
set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
set(MATH_LIB_LINK_FLAGS "-L$ENV{MKLROOT}/lib -Wl,-rpath,$ENV{MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl")

# Find OpenMP
if(APPLE AND GQTEN_USE_OMP_GEMM_BATCH)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
      set(OPENMP_INCLUDES "/usr/local/Cellar/libomp/8.0.0/include")
      set(OPENMP_LIBRARIES "/usr/local/Cellar/libomp/8.0.0/lib")
      set(OpenMP_C "${CMAKE_C_COMPILER}")
      set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
      set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
      set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
      set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
      set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(OPENMP_LIBRARIES "/usr/local/Cellar/libomp/8.0.0/lib")
      set(OPENMP_INCLUDES "/usr/local/Cellar/libomp/8.0.0/include")
      set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
      set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
      set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
      set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    endif()
endif()
if(GQTEN_USE_OMP_GEMM_BATCH)
  find_package(OpenMP REQUIRED)
endif(GQTEN_USE_OMP_GEMM_BATCH)
if (OPENMP_FOUND)
  if(APPLE)
    include_directories("${OPENMP_INCLUDES}")
    link_directories("${OPENMP_LIBRARIES}")
  endif()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

option(GQTEN_OMP_DEV_MODE "Print more information in this mode." OFF)
if(GQTEN_OMP_DEV_MODE)
  add_definitions(-DGQTEN_OMP_DEV_MODE)
endif(GQTEN_OMP_DEV_MODE)


# Subdirectories.
add_subdirectory(external)
add_subdirectory(src lib)


# Unittests
if(GQTEN_BUILD_UNITTEST)
  enable_testing()
  add_subdirectory(tests tests)
endif()
