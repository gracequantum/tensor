// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-28 16:19
* 
* Description: GraceQ/tensor project. Parallel GEMM batch worker.
*/
#include "mkl.h"
#include "mpi.h"


const char kGemmWorkerStatCont = 'c';
const char kGemmWorkerStatStop = 's';


inline void MPI_RecvGemmData(
    long idx, long local_batch_size,
    long *pm, long *pn, long *pk,
    double * &a, double * &b, double * &c) {
  long gemm_info[3];
  MPI_Recv(gemm_info, 3, MPI_LONG, 0, idx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  *pm = gemm_info[0];
  *pn = gemm_info[1];
  *pk = gemm_info[2];
  auto a_size = (*pm) * (*pk);
  a = new double[a_size];
  MPI_Recv(a, a_size, MPI_DOUBLE, 0, idx+local_batch_size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  auto b_size = (*pk) * (*pn);
  b = new double[b_size];
  MPI_Recv(b, b_size, MPI_DOUBLE, 0, idx+2*local_batch_size, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  auto c_size = (*pm) * (*pn);
  c = new double[c_size];
}


inline void MPI_SendGemmRes(long idx, double *c, const long m, const long n) {
  MPI_Send(c, m*n, MPI_DOUBLE, 0, idx, MPI_COMM_WORLD);
}


inline void MPI_RecvGemmWorkerStat(char *pworker_stat) {
  MPI_Recv(pworker_stat, 1, MPI_CHAR, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


inline char CheckWorkStat(void) {
  char stat;
  MPI_RecvGemmWorkerStat(&stat);
  return stat;
}


int main(int argc, char *argv[]) {
  int provided; 
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  /* TODO: provided check */
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  long local_batch_size;
  while (CheckWorkStat() != kGemmWorkerStatStop) {
    MPI_Recv(
        &local_batch_size, 1, MPI_LONG, 0, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    auto local_gemm_batch_a_array = new double *[local_batch_size];
    auto local_gemm_batch_b_array = new double *[local_batch_size];
    auto local_gemm_batch_c_array = new double *[local_batch_size];
    auto local_gemm_batch_m_array = new long[local_batch_size];
    auto local_gemm_batch_n_array = new long[local_batch_size];
    auto local_gemm_batch_k_array = new long[local_batch_size];

    for (long i = 0; i < local_batch_size; ++i) {
      MPI_RecvGemmData(
          i, local_batch_size,
          &local_gemm_batch_m_array[i],
          &local_gemm_batch_n_array[i],
          &local_gemm_batch_k_array[i],
          local_gemm_batch_a_array[i],
          local_gemm_batch_b_array[i],
          local_gemm_batch_c_array[i]);
    }


    for (long i = 0; i < local_batch_size; ++i) {
      cblas_dgemm(
          CblasRowMajor,
          CblasNoTrans, CblasNoTrans,
          local_gemm_batch_m_array[i],
          local_gemm_batch_n_array[i],
          local_gemm_batch_k_array[i],
          1.0,
          local_gemm_batch_a_array[i], local_gemm_batch_k_array[i],
          local_gemm_batch_b_array[i], local_gemm_batch_n_array[i],
          0.0,
          local_gemm_batch_c_array[i], local_gemm_batch_n_array[i]);
    }


    for (long i = 0; i < local_batch_size; ++i) {
      MPI_SendGemmRes(
          i,
          local_gemm_batch_c_array[i],
          local_gemm_batch_m_array[i], local_gemm_batch_n_array[i]);
      delete[] local_gemm_batch_a_array[i];
      delete[] local_gemm_batch_b_array[i];
      delete[] local_gemm_batch_c_array[i];
    }

    delete[] local_gemm_batch_m_array;
    delete[] local_gemm_batch_n_array;
    delete[] local_gemm_batch_k_array;
    delete[] local_gemm_batch_a_array;
    delete[] local_gemm_batch_b_array;
    delete[] local_gemm_batch_c_array;
  }


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
