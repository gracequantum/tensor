// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-28 09:34
* 
* Description: GraceQ/tensor project. Distributed numerical function for GQTensor, src file.
*/
#include "mpi_ten_numer_func.h"
#include "ten_numer_func.h"
#include "gqten/gqten.h"

#include <algorithm>

#include "mkl.h"
#include "mpi.h"
#include "omp.h"

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


namespace gqten {


GQTensor *GQTEN_MPI_Contract(
    const GQTensor &ta, const GQTensor &tb,
    const std::vector<std::vector<long>> &axes_set,
    MPI_Comm comm, const int workers) {
  auto ctrct_axes_a = axes_set[0];
  auto ctrct_axes_b = axes_set[1];

  // Blocks contraction batch.
  std::vector<QNBlock *> pnew_blks;
  if (ta.cblocks().size() > 0 && tb.cblocks().size() > 0) {

#ifdef GQTEN_TIMING_MODE
    Timer blks_ctrct_batch_timer("blks_ctrct_batch");
    blks_ctrct_batch_timer.Restart();
#endif

    pnew_blks = GQTEN_MPI_BlocksCtrctBatch(
        ctrct_axes_a, ctrct_axes_b,
        1.0, ta.cblocks(), tb.cblocks(),
        comm, workers);

#ifdef GQTEN_TIMING_MODE
    blks_ctrct_batch_timer.PrintElapsed();
#endif

  }

  // Initialize contracted tensor.
  auto res_t  = InitCtrctedTen(ta, tb, ctrct_axes_a, ctrct_axes_b);

#ifdef GQTEN_TIMING_MODE
  Timer blks_wrap_timer("blks_wrap");
  blks_wrap_timer.Restart();
#endif

  // Wrap blocks.
  WrapCtrctBlocks(pnew_blks, res_t);

#ifdef GQTEN_TIMING_MODE
  blks_wrap_timer.PrintElapsed();
#endif

  return res_t;
}


std::vector<QNBlock *> GQTEN_MPI_BlocksCtrctBatch(
    const std::vector<long> &ctrct_axes_a,
    const std::vector<long> &ctrct_axes_b,
    const double alpha,
    const std::vector<QNBlock *> &ta_blks,
    const std::vector<QNBlock *> &tb_blks,
    MPI_Comm comm, const int workers) {
  // Data prepare.
#ifdef GQTEN_TIMING_MODE
  Timer blk_match_timer("blk_match");
  blk_match_timer.Restart();
  Timer blk_match_ten_trans_timer("blk_match_ten_trans");
#endif

  auto ta_blks_num = ta_blks.size();
  auto tb_blks_num = tb_blks.size();
  assert(ta_blks_num > 0);
  assert(tb_blks_num > 0);
  // Check whether need transpose.
  std::vector<long> transed_axes_a, transed_axes_b;
  bool ta_need_trans = CtrctTransChecker(
                           ctrct_axes_a,
                           ta_blks[0]->ndim, 'a',
                           transed_axes_a);
  bool tb_need_trans = CtrctTransChecker(
                           ctrct_axes_b,
                           tb_blks[0]->ndim, 'b',
                           transed_axes_b);
  // Generate blocks part-hash tokens.
  auto ta_blks_part_hash_table = GenBlksPartHashTable(ta_blks, ctrct_axes_a);
  auto tb_blks_part_hash_table = GenBlksPartHashTable(tb_blks, ctrct_axes_b);
  // Count matched block pairs.
  long blk_pairs = 0;
  for (std::size_t i = 0; i < ta_blks_num; ++i) {
    for (std::size_t j = 0; j < tb_blks_num; ++j) {
      if (ta_blks_part_hash_table[i] == tb_blks_part_hash_table[j]) {
        ++blk_pairs;
      }
    }
  }

#ifdef GQTEN_CONTRACT_BLOCK_COUNTING
  std::cout << "[counting] "
            << "ta # of blks " << std::setw(10) << std::left << ta_blks_num
            << "tb # of blks " << std::setw(10) << std::left << tb_blks_num
            << "matched pair " << std::setw(10) << std::left << blk_pairs << std::endl;
#endif

  // No match, return empty vector.
  if (blk_pairs == 0) {
    return std::vector<QNBlock *>();
  }

  // Initialize data.
  // Data with size of number of blocks of tensor.
  auto ta_to_ctrct_blks = new const double *[ta_blks_num] ();
  auto tb_to_ctrct_blks = new const double *[tb_blks_num] ();
  auto ta_to_ctrct_blk_saved_dims = new long[ta_blks_num];
  auto tb_to_ctrct_blk_saved_dims = new long[tb_blks_num];
  auto ta_to_ctrct_blk_ctrct_dims = new long[ta_blks_num];
  auto tb_to_ctrct_blk_ctrct_dims = new long[tb_blks_num];
  // Data with size of number of block pairs.
  auto gemm_batch_a_array = new const double *[blk_pairs];
  auto gemm_batch_b_array = new const double *[blk_pairs];
  auto gemm_batch_c_array = new double *[blk_pairs];
  auto gemm_batch_m_array = new long[blk_pairs];
  auto gemm_batch_n_array = new long[blk_pairs];
  auto gemm_batch_k_array = new long[blk_pairs];
  std::vector<QNBlock *> pnew_blks(blk_pairs, nullptr);

  // Assign data.
  long blk_pair_cnt = 0;
  for (std::size_t i = 0; i < ta_blks_num; ++i) {
    for (std::size_t j = 0; j < tb_blks_num; ++j) {
      if (ta_blks_part_hash_table[i] == tb_blks_part_hash_table[j]) {
        // Generate new blocks.
        auto pnew_blk_qnscts = GetPNewBlkQNScts(
                                   ta_blks[i], tb_blks[j],
                                   ctrct_axes_a, ctrct_axes_b);
        pnew_blks[blk_pair_cnt] = new QNBlock(pnew_blk_qnscts);
        // For contracting to scalar case.
        if (pnew_blks[blk_pair_cnt]->cdata() == nullptr) {
          pnew_blks[blk_pair_cnt]->data() = new double[1];
        }

        // Deal with ta block.
        if (ta_to_ctrct_blks[i] == nullptr) {
          // Calculate dimensions information.
          CalcBlkCtrctDimsInfo(
              i, ta_blks[i], ctrct_axes_a,
              ta_to_ctrct_blk_saved_dims, ta_to_ctrct_blk_ctrct_dims);
          // Generate contraction data.
          if (ta_need_trans) {

#ifdef GQTEN_TIMING_MODE
            blk_match_ten_trans_timer.Restart();
#endif

            auto blk_data_transed_to_ctrct = TransposeData(
                ta_blks[i]->cdata(),
                ta_blks[i]->ndim,
                ta_blks[i]->size,
                ta_blks[i]->shape,
                transed_axes_a);

#ifdef GQTEN_TIMING_MODE
            blk_match_ten_trans_timer.PrintElapsed(8);
#endif

            ta_to_ctrct_blks[i] = blk_data_transed_to_ctrct;
          } else {
            ta_to_ctrct_blks[i] = ta_blks[i]->cdata();
          }
        }
        // Assign gemm_batch parameters.
        gemm_batch_a_array[blk_pair_cnt] = ta_to_ctrct_blks[i];
        gemm_batch_m_array[blk_pair_cnt] = ta_to_ctrct_blk_saved_dims[i];
        gemm_batch_k_array[blk_pair_cnt] = ta_to_ctrct_blk_ctrct_dims[i];

        // Deal with tb block.
        if (tb_to_ctrct_blks[j] == nullptr) {
          // Calculate dimensions information.
          CalcBlkCtrctDimsInfo(
              j, tb_blks[j], ctrct_axes_b,
              tb_to_ctrct_blk_saved_dims, tb_to_ctrct_blk_ctrct_dims);
          // Generate contraction data.
          if (tb_need_trans) {

#ifdef GQTEN_TIMING_MODE
            blk_match_ten_trans_timer.Restart();
#endif

            auto blk_data_transed_to_ctrct = TransposeData(
                tb_blks[j]->cdata(),
                tb_blks[j]->ndim,
                tb_blks[j]->size,
                tb_blks[j]->shape,
                transed_axes_b);
            tb_to_ctrct_blks[j] = blk_data_transed_to_ctrct;

#ifdef GQTEN_TIMING_MODE
            blk_match_ten_trans_timer.PrintElapsed(8);
#endif

          } else {
            tb_to_ctrct_blks[j] = tb_blks[j]->cdata();
          }
        }
        // Assign gemm_batch parameters.
        gemm_batch_b_array[blk_pair_cnt] = tb_to_ctrct_blks[j];
        gemm_batch_n_array[blk_pair_cnt] = tb_to_ctrct_blk_saved_dims[j];
        gemm_batch_c_array[blk_pair_cnt] = pnew_blks[blk_pair_cnt]->data();

#ifdef GQTEN_CONTRACT_BLOCK_COUNTING
        std::cout << "[counting] blk_m_dim " << std::setw(10) << std::left << gemm_batch_m_array[blk_pair_cnt]
                  << "blk_k_dim " << std::setw(10) << std::left << gemm_batch_k_array[blk_pair_cnt]
                  << "blk_n_dim " << std::setw(10) << std::left << gemm_batch_n_array[blk_pair_cnt] << std::endl;
#endif

        ++blk_pair_cnt;
      }
    }
  }

#ifdef GQTEN_TIMING_MODE
  blk_match_timer.PrintElapsed();
#endif

  // Call gemm batch function.
#ifdef GQTEN_TIMING_MODE
  Timer dgemm_batch_timer("gemm_batch");
  dgemm_batch_timer.Restart();
#endif

  GQTEN_MPI_GemmBatch(
      gemm_batch_m_array, gemm_batch_n_array, gemm_batch_k_array,
      gemm_batch_a_array, gemm_batch_b_array, gemm_batch_c_array,
      blk_pairs,
      comm, workers); 

#ifdef GQTEN_TIMING_MODE
  dgemm_batch_timer.PrintElapsed();
#endif

  // Free temporary variables.
  if (ta_need_trans) {
    for (std::size_t i = 0; i < ta_blks_num; ++i) {
      delete[] ta_to_ctrct_blks[i];     // Delete the data.
    }
  }
  if (tb_need_trans) {
    for (std::size_t i = 0; i < tb_blks_num; ++i) {
      delete[] tb_to_ctrct_blks[i];
    }
  }
  delete[] ta_to_ctrct_blks;            // Delete the pointers.
  delete[] tb_to_ctrct_blks;
  delete[] ta_to_ctrct_blk_saved_dims;
  delete[] tb_to_ctrct_blk_saved_dims;
  delete[] ta_to_ctrct_blk_ctrct_dims;
  delete[] tb_to_ctrct_blk_ctrct_dims;

  delete[] gemm_batch_a_array;
  delete[] gemm_batch_b_array;
  delete[] gemm_batch_c_array;
  delete[] gemm_batch_m_array;
  delete[] gemm_batch_n_array;
  delete[] gemm_batch_k_array;

  return pnew_blks;
}


void GQTEN_MPI_GemmBatch(     // Manager.
    const long *m_array, const long *n_array, const long *k_array,
    const double **a_array, const double **b_array, double **c_array,
    const long batch_size,
    MPI_Comm comm, const int workers) {

#ifdef GQTEN_TIMING_MODE
  Timer gemm_batch_task_allocation_timer("gemm_batch_task_allocation");
  gemm_batch_task_allocation_timer.Restart();
#endif

  auto tasks = TaskScheduler(workers, batch_size, m_array, k_array, n_array);
  auto local_batch_sizes = CalcLocalBatchSizes(tasks);

#ifdef GQTEN_TIMING_MODE
  gemm_batch_task_allocation_timer.PrintElapsed();
#endif

#pragma omp parallel for num_threads(workers) schedule(static,1)
  for (int i = 1; i <= workers; ++i) {
    MPI_SendGemmWorkerStat(kGemmWorkerStatCont, i, comm);
    long local_batch_size = local_batch_sizes[i];
    MPI_Send(&local_batch_size, 1, MPI_LONG, i, 0, comm);
    for (auto &task_idx : tasks[i]) {
      MPI_SendGemmData(
          m_array[task_idx], n_array[task_idx], k_array[task_idx],
          a_array[task_idx], b_array[task_idx],
          i, comm);
    }
  }

#ifdef GQTEN_TIMING_MODE
  Timer gemm_batch_p0_timer("gemm_batch_p0");
    gemm_batch_p0_timer.Restart();
#endif

  for (auto &task_idx : tasks[0]) {
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        m_array[task_idx], n_array[task_idx], k_array[task_idx],
        1.0,
        a_array[task_idx], k_array[task_idx],
        b_array[task_idx], n_array[task_idx],
        0.0,
        c_array[task_idx], n_array[task_idx]);
  }

#ifdef GQTEN_TIMING_MODE
  gemm_batch_p0_timer.PrintElapsed();
#endif

#pragma omp parallel for num_threads(workers) schedule(static,1)
  for (int i = 1; i <= workers; ++i) {
    for (auto &task_idx : tasks[i]) {
      MPI_RecvGemmRes(
          c_array[task_idx], m_array[task_idx], n_array[task_idx],
          i, comm);
    }
  }
}


bool sortbysecdesc(
    const std::pair<long, long> &a,
    const std::pair<long, long> &b) {
  return (a.second > b.second);
}


std::vector<std::vector<long>> TaskScheduler(
    const int workers, const long batch_size,
    const long *m_array, const long  *k_array, const long *n_array) {
  long cost;
  long cost_tot = 0;
  auto costs = std::vector<std::pair<long, long>>(batch_size);
  for (long i = 0; i < batch_size; ++i) {
    cost = m_array[i] * k_array[i] * n_array[i];
    costs[i] = std::make_pair(i, cost);
    cost_tot += cost;
  }
  std::sort(costs.begin(), costs.end(), sortbysecdesc);
  int labours = workers + 1;    // Labours include manager and workers.
  long cost_avg = cost_tot / labours;
  auto tasks = std::vector<std::vector<long>>(labours);
  long cost_temp = 0;
  int labour = 0;
  long tail_task;
  for (long i = 0; i < batch_size; ++i) {
    cost_temp += costs[i].second;
    if (cost_temp > cost_avg) {
      labour++;
      if (labour == labours) {
        tail_task = i;
        break;
      }
      tasks[labour].push_back(costs[i].first);
      cost_temp = costs[i].second;
    } else {
      tasks[labour].push_back(costs[i].first);
    }
  }
  for (long i = tail_task; i < batch_size; ++i) {
    tasks[labours-1].push_back(costs[i].first);
  }
  return tasks;
}
} /* gqten */ 
