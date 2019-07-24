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
const int kMpiGemmDataRecverCallMpiRecvFuncNum = 3;


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

  long *pgemm_infos = nullptr;
  MPI_Request *precv_reqs = nullptr;
  MPI_Request *psend_reqs = nullptr;
  MPI_Status *precv_stats = nullptr;
  MPI_Status *psend_stats = nullptr;
  double **local_gemm_batch_a_array = nullptr;
  double **local_gemm_batch_b_array = nullptr;
  double **local_gemm_batch_c_array = nullptr;
  long *local_gemm_batch_m_array = nullptr;
  long *local_gemm_batch_n_array = nullptr;
  long *local_gemm_batch_k_array = nullptr;

  while (CheckWorkStat() != kGemmWorkerStatStop) {
    MPI_Recv(
        &local_batch_size, 1, MPI_LONG, 0, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    local_gemm_batch_a_array = new double *[local_batch_size];
    local_gemm_batch_b_array = new double *[local_batch_size];
    local_gemm_batch_c_array = new double *[local_batch_size];
    local_gemm_batch_m_array = new long[local_batch_size];
    local_gemm_batch_n_array = new long[local_batch_size];
    local_gemm_batch_k_array = new long[local_batch_size];
    precv_reqs = new MPI_Request [local_batch_size *
                                  kMpiGemmDataRecverCallMpiRecvFuncNum];
    precv_stats = new MPI_Status [local_batch_size *
                                  kMpiGemmDataRecverCallMpiRecvFuncNum];
    psend_reqs = new MPI_Request [local_batch_size];
    psend_stats = new MPI_Status [local_batch_size];
    pgemm_infos = new long [local_batch_size *
                            kMpiGemmDataRecverCallMpiRecvFuncNum];

    // Non-blocking receive gemm data informations.
    for (long i = 0; i < local_batch_size; ++i) {
      MPI_Irecv(
          &pgemm_infos[i*kMpiGemmDataRecverCallMpiRecvFuncNum],
          3, MPI_LONG, 0, i, MPI_COMM_WORLD, &precv_reqs[i]); 
    }

    // Non-blocking receive gemm data a and b matrices.
    for (long i = 0; i < local_batch_size; ++i) {
      MPI_Wait(&precv_reqs[i], &precv_stats[i]);

      auto gemm_infos_offset = i * kMpiGemmDataRecverCallMpiRecvFuncNum;
      local_gemm_batch_m_array[i] = pgemm_infos[gemm_infos_offset + 0];
      local_gemm_batch_n_array[i] = pgemm_infos[gemm_infos_offset + 1];
      local_gemm_batch_k_array[i] = pgemm_infos[gemm_infos_offset + 2];
      auto a_size = local_gemm_batch_m_array[i] * local_gemm_batch_k_array[i];
      auto b_size = local_gemm_batch_k_array[i] * local_gemm_batch_n_array[i];
      local_gemm_batch_a_array[i] = new double [a_size];
      local_gemm_batch_b_array[i] = new double [b_size];

      MPI_Irecv(
          local_gemm_batch_a_array[i], a_size, MPI_DOUBLE,
          0, i+local_batch_size, MPI_COMM_WORLD,
          &precv_reqs[local_batch_size + 2*i]);
      MPI_Irecv(
          local_gemm_batch_b_array[i], b_size, MPI_DOUBLE,
          0, i+2*local_batch_size, MPI_COMM_WORLD,
          &precv_reqs[local_batch_size + 2*i + 1]);
    }

    // Gemm and non-blocking result send.
    for (long i = 0; i < local_batch_size; ++i) {
      MPI_Waitall(
          2,
          &precv_reqs[local_batch_size + 2*i],
          &precv_stats[local_batch_size + 2*i]);
      auto c_size = local_gemm_batch_m_array[i] * local_gemm_batch_n_array[i];
      local_gemm_batch_c_array[i] = new double [c_size];
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
      MPI_Isend(
          local_gemm_batch_c_array[i], c_size, MPI_DOUBLE,
          0, i, MPI_COMM_WORLD,
          &psend_reqs[i]);
    }

    // Wait send finish and delete temporary data.
    MPI_Waitall(local_batch_size, psend_reqs, psend_stats);
    delete [] precv_reqs;
    delete [] precv_stats;
    delete [] psend_reqs;
    delete [] psend_stats;
    delete [] pgemm_infos;
    for (long i = 0; i < local_batch_size; ++i) {
      delete [] local_gemm_batch_a_array[i];
      delete [] local_gemm_batch_b_array[i];
      delete [] local_gemm_batch_c_array[i];
    }
    delete [] local_gemm_batch_a_array;
    delete [] local_gemm_batch_b_array;
    delete [] local_gemm_batch_c_array;
    delete [] local_gemm_batch_m_array;
    delete [] local_gemm_batch_n_array;
    delete [] local_gemm_batch_k_array;
  }


  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
