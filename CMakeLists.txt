#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 16:28
#  
#  Description: GraceQ/tensor project. Main CMake file.
# 

# CMake basic setting
cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project("GraceQ/tensor"
  DESCRIPTION "A high-performance tensor library for the quantum physics community"
  LANGUAGES CXX)


# Build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()


# Advanced options.
option(GQTEN_BUILD_UNITTEST "Build unittests for GraceQ/tensor." OFF)

option(GQTEN_USE_MKL_GEMM_BATCH "Use MKL's cblas_dgemm_batch function." OFF)
if(GQTEN_USE_MKL_GEMM_BATCH)
  add_definitions(-DGQTEN_USE_MKL_GEMM_BATCH)
endif()

option(GQTEN_OPENMP_PARALLEL "Build OpenMP support." OFF)

option(GQTEN_TIMING_MODE "Timing mode." OFF)
if(GQTEN_TIMING_MODE)
  add_definitions(-DGQTEN_TIMING_MODE)
endif()

option(GQTEN_CONTRACT_BLOCK_COUNTING "Count the number of blocks when contracting tensors." OFF)
if(GQTEN_CONTRACT_BLOCK_COUNTING)
  add_definitions(-DGQTEN_CONTRACT_BLOCK_COUNTING)
endif()

option(GQTEN_USE_EXTERNAL_HPTT_LIB "Use external hptt library" OFF)


# Compilation and linking control.
set(CMAKE_CXX_STANDARD 11)
add_definitions("-g")


# Set MKL compile flags and link flags.
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
  set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
  # Link the Intel's OpenMP library to avoid performance issue when the library calls the MKL's gesdd function.
  set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
  set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl)
endif()


# Find OpenMP
if(GQTEN_OPENMP_PARALLEL)
  if(APPLE)
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
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    if(APPLE)
      include_directories("${OPENMP_INCLUDES}")
      link_directories("${OPENMP_LIBRARIES}")
    endif()
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()


# Installation path
set(GQTEN_INCLUDE_DEST "${CMAKE_INSTALL_PREFIX}/include")
set(GQTEN_LIB_DEST "${CMAKE_INSTALL_PREFIX}/lib")


# Build dependencies.
add_subdirectory(external)

# Build library.
set(GQTEN_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
set(GQTEN_SRC_PATH "${PROJECT_SOURCE_DIR}/src")
add_subdirectory(src lib)


# Unittests
if(GQTEN_BUILD_UNITTEST)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(tests tests)
endif()
