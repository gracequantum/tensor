// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-28 16:19
* 
* Description: GraceQ/tensor project. Parallel GEMM batch worker.
*/
#include <string>
#include <fstream>
#include <iomanip>

#include "mkl.h"
#include "mpi.h"
#include "omp.h"


const char kGemmWorkerStatCont = 'c';
const char kGemmWorkerStatStop = 's';


inline void MPI_RecvGemmData(
    long *pm, long *pn, long *pk,
    double * &a, double * &b, double * &c) {
  long gemm_info[3];
  MPI_Recv(gemm_info, 3, MPI_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  *pm = gemm_info[0];
  *pn = gemm_info[1];
  *pk = gemm_info[2];
  auto a_size = (*pm) * (*pk);
  a = new double[a_size];
  MPI_Recv(a, a_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  auto b_size = (*pk) * (*pn);
  b = new double[b_size];
  MPI_Recv(b, b_size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  auto c_size = (*pm) * (*pn);
  c = new double[c_size];
}


inline void MPI_SendGemmRes(double *c, const long m, const long n) {
  MPI_Send(c, m*n, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
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

#ifdef GQTEN_TIMING_MODE
  auto timing_file = "gemm_batch_timing_p" + std::to_string(my_rank) + ".log";
  auto timing_notes = "gemm_batch_p" + std::to_string(my_rank);
  std::ofstream ofs(timing_file);
  double start_time;
#endif

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
          &local_gemm_batch_m_array[i],
          &local_gemm_batch_n_array[i],
          &local_gemm_batch_k_array[i],
          local_gemm_batch_a_array[i],
          local_gemm_batch_b_array[i],
          local_gemm_batch_c_array[i]);
    }

#ifdef GQTEN_TIMING_MODE
    start_time = omp_get_wtime();
#endif

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

#ifdef GQTEN_TIMING_MODE
    ofs << "[timing] "
        << timing_notes << "\t"
        << std::setw(9) << std::setprecision(6) << std::fixed
        << omp_get_wtime() - start_time << std::endl;
#endif


    for (long i = 0; i < local_batch_size; ++i) {
      MPI_SendGemmRes(
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

#ifdef GQTEN_TIMING_MODE
  ofs.close();
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
