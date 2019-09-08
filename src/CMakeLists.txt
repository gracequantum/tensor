#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 22:54
#  
#  Description: GraceQ/tensor project. CMake file to control the library source code.
# 
if(NOT GQTEN_USE_EXTERNAL_HPTT_LIB)
  set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/hptt/include")
endif()


if(GQTEN_MPI_PARALLEL)
  add_library(gqten
    utils.cc
    timer.cc
    ten_trans.cc
    qn.cc
    qnsector.cc
    index.cc
    qnblock.cc
    gqtensor.cc
    ten_ctrct.cc
    ten_lincmb.cc
    ten_svd.cc
    mpi_ten_ctrct.cc)

  target_include_directories(gqten
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
    PRIVATE ${hptt_INCLUDE_DIR}
    PRIVATE ${MPI_CXX_INCLUDE_PATH})

  target_compile_options(gqten
    PRIVATE ${MATH_LIB_COMPILE_FLAGS}
    PRIVATE ${MPI_CXX_COMPILE_FLAGS})
else()
  add_library(gqten
    utils.cc
    timer.cc
    ten_trans.cc
    qn.cc
    qnsector.cc
    index.cc
    qnblock.cc
    gqtensor.cc
    ten_ctrct.cc
    ten_lincmb.cc
    ten_svd.cc)

  target_include_directories(gqten
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
    PRIVATE ${hptt_INCLUDE_DIR})

  target_compile_options(gqten
    PRIVATE ${MATH_LIB_COMPILE_FLAGS})
endif()
