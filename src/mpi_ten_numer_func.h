// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-28 09:29
* 
* Description: GraceQ/tensor project. Distributed numerical function for GQTensor, head file.
*/
#ifndef GQTEN_MPI_TEN_NUMER_FUNC_H
#define GQTEN_MPI_TEN_NUMER_FUNC_H


#include "gqten/gqten.h"

#include "mkl.h"
#include "mpi.h"


namespace gqten {


std::vector<QNBlock *> GQTEN_MPI_BlocksCtrctBatch(
    const std::vector<long> &, const std::vector<long> &,
    const double,
    const std::vector<QNBlock *> &, const std::vector<QNBlock *> &,
    MPI_Comm, const int);

void GQTEN_MPI_GemmBatch(
    const long *, const long *, const long *,
    const double **, const double **, double **,
    const long,
    MPI_Comm, const int);


// Inline functions.
inline std::vector<long>
CalcLocalBatchSizes(const long batch_size, const int workers) {
  long remainder = batch_size % workers;
  long local_batch_size = (batch_size - remainder) / workers;
  std::vector<long> local_batch_sizes(workers);
  for (int i = 0; i < workers-1; ++i) {
    local_batch_sizes[i] = local_batch_size;
  }
  local_batch_sizes[workers-1] = local_batch_size + remainder;
  return local_batch_sizes;
}


inline std::vector<long>
CalcTaskOffsets(const std::vector<long> &local_batch_sizes, const int workers) {
  std::vector<long> task_offsets(workers);
  long offset = 0;
  for (int i = 0; i < workers; ++i) {
    task_offsets[i] = offset;
    offset += local_batch_sizes[i];
  }
  return task_offsets;
}


inline void MPI_SendGemmData(
    const long m, const long n, const long k,
    const double *a, const double *b,
    const int worker, MPI_Comm comm) {
  long gemm_info[3];
  gemm_info[0] = m;
  gemm_info[1] = n;
  gemm_info[2] = k;
  MPI_Send(gemm_info, 3, MPI_LONG, worker, 1, comm);
  MPI_Send(a, m*k, MPI_DOUBLE, worker, 2, comm);
  MPI_Send(b, k*n, MPI_DOUBLE, worker, 3, comm);
}


inline void MPI_RecvGemmRes(
    double *c,
    const long m, const long n, 
    const int worker, MPI_Comm comm) {
  MPI_Recv(c, m*n, MPI_DOUBLE, worker, 4, comm, MPI_STATUS_IGNORE); 
}
} /* gqten */ 
#endif /* ifndef GQTEN_MPI_TEN_NUMER_FUNC_H */
