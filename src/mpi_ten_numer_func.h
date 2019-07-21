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


const int kMpiGemmDataSenderCallMpiSendFuncNum = 3;


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


std::vector<std::vector<long>> TaskScheduler(
    const int, const long,
    const long *, const long  *, const long *);


// Inline functions.
inline std::vector<long>
CalcLocalBatchSizes(const std::vector<std::vector<long>> &tasks) {
  auto local_batch_sizes = std::vector<long>(tasks.size());
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    local_batch_sizes[i] = tasks[i].size();
  }
  return local_batch_sizes;
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


inline void MPI_IsendGemmData(
    const long m, const long n, const long k,
    const double *a, const double *b,
    const int worker,
    MPI_Comm comm, MPI_Request *reqs) {
  long gemm_info[3];
  gemm_info[0] = m;
  gemm_info[1] = n;
  gemm_info[2] = k;
  MPI_Isend(gemm_info, 3, MPI_LONG, worker, 1, comm, reqs);
  MPI_Isend(a, m*k, MPI_DOUBLE, worker, 2, comm, reqs+1);
  MPI_Isend(b, k*n, MPI_DOUBLE, worker, 3, comm, reqs+2);
}


inline void MPI_RecvGemmRes(
    double *c,
    const long m, const long n, 
    const int worker, MPI_Comm comm) {
  MPI_Recv(c, m*n, MPI_DOUBLE, worker, 4, comm, MPI_STATUS_IGNORE); 
}


inline void MPI_IrecvGemmRes(
    double *c,
    const long m, const long n,
    const int worker,
    MPI_Comm comm, MPI_Request *req) {
  MPI_Irecv(c, m*n, MPI_DOUBLE, worker, 4, comm, req);
}
} /* gqten */ 
#endif /* ifndef GQTEN_MPI_TEN_NUMER_FUNC_H */
