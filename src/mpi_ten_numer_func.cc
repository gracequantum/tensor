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

#include "mkl.h"
#include "mpi.h"

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
  auto local_batch_sizes = CalcLocalBatchSizes(batch_size, workers);
  auto task_offsets = CalcTaskOffsets(local_batch_sizes, workers);

  for (int i = 0; i < workers; ++i) {
    MPI_SendGemmWorkerStat(kGemmWorkerStatCont, i+1, comm);
    long local_batch_size = local_batch_sizes[i];
    MPI_Send(&local_batch_size, 1, MPI_LONG, i+1, 0, comm);

    int local_task_base_offset = task_offsets[i];
    for (long j = 0; j < local_batch_size; ++j) {
      long local_task_offset = local_task_base_offset + j;
      MPI_SendGemmData(
          m_array[local_task_offset],
          n_array[local_task_offset],
          k_array[local_task_offset],
          a_array[local_task_offset], b_array[local_task_offset],
          i+1, comm);
    }

    for (long j = 0; j < local_batch_size; ++j) {
      long local_task_offset = local_task_base_offset + j;
      MPI_RecvGemmRes(
          c_array[local_task_offset],
          m_array[local_task_offset], n_array[local_task_offset],
          i+1, comm);
    }

  }
}
} /* gqten */ 
