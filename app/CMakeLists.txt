#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-06-28 18:47
#  
#  Description: GraceQ/tensor project. Applications CMake file.
# 
if(GQTEN_MPI_PARALLEL)
  add_executable(gqten_mpi_gemm_batch_worker
    gqten_mpi_gemm_batch_worker.cc)

  target_include_directories(gqten_mpi_gemm_batch_worker
    PUBLIC ${CFLAGS}
    PRIVATE ${MPI_CXX_INCLUDE_PATH})

  target_compile_options(gqten_mpi_gemm_batch_worker
    PRIVATE ${MATH_LIB_COMPILE_FLAGS}
    PRIVATE ${MPI_CXX_COMPILE_FLAGS})

  target_link_libraries(gqten_mpi_gemm_batch_worker
    ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS}
    ${MATH_LIB_LINK_FLAGS})
endif()
