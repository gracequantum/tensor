#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 22:54
#  
#  Description: GraceQ/tensor project. CMake file to control the library source
#               code.
# 


if(GQTEN_MPI_PARALLEL)
  add_library(gqten
    ten_numer_func.cc
    mpi_ten_numer_func.cc
    gqten.cc)

  target_include_directories(gqten
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PUBLIC "${PROJECT_SOURCE_DIR}/external/hptt/include"
    PRIVATE ${MPI_CXX_INCLUDE_PATH})

  target_compile_options(gqten
    PRIVATE ${MATH_LIB_COMPILE_FLAGS}
    PRIVATE ${MPI_CXX_COMPILE_FLAGS})

  target_link_libraries(gqten
    hptt
    ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS}
    ${MATH_LIB_LINK_FLAGS})
else()
  add_library(gqten
    ten_numer_func.cc
    gqten.cc)

  target_include_directories(gqten
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PUBLIC "${PROJECT_SOURCE_DIR}/external/hptt/include")

  target_compile_options(gqten
    PUBLIC ${MATH_LIB_COMPILE_FLAGS})

  target_link_libraries(gqten
    hptt
    "${MATH_LIB_LINK_FLAGS}")
endif()
