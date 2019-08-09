// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-09 11:05
* 
* Description: GraceQ/tensor project. Implementation details about tensor contraction.
*/
#include "gqten/gqten.h"
#include "ten_ctrct.h"
#include "ten_trans.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <assert.h>

#include "mkl.h"

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


GQTensor *Contract(
    const GQTensor &ta, const GQTensor &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto ctrct_axes_a = axes_set[0];
  auto ctrct_axes_b = axes_set[1];

  // Blocks contraction batch.
  std::vector<QNBlock *> pnew_blks;
  if (ta.cblocks().size() > 0 && tb.cblocks().size() > 0) {

#ifdef GQTEN_TIMING_MODE
    Timer blks_ctrct_batch_timer("blks_ctrct_batch");
    blks_ctrct_batch_timer.Restart();
#endif

    pnew_blks = BlocksCtrctBatch(
        ctrct_axes_a, ctrct_axes_b,
        1.0, ta.cblocks(), tb.cblocks());

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


std::vector<QNBlock *> BlocksCtrctBatch(
    const std::vector<long> &ctrct_axes_a,
    const std::vector<long> &ctrct_axes_b,
    const double alpha,
    const std::vector<QNBlock *> &ta_blks,
    const std::vector<QNBlock *> &tb_blks) {
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
            << "matched pair " << std::setw(10) << std::left << blk_pairs
            << std::endl;
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
  auto gemm_batch_transa_array = new CBLAS_TRANSPOSE[blk_pairs];
  auto gemm_batch_transb_array = new CBLAS_TRANSPOSE[blk_pairs];
  for (long i = 0; i < blk_pairs; ++i) {
    gemm_batch_transa_array[i] = CblasNoTrans;
    gemm_batch_transb_array[i] = CblasNoTrans;
  }
  auto gemm_batch_a_array = new const double *[blk_pairs];
  auto gemm_batch_b_array = new const double *[blk_pairs];
  auto gemm_batch_c_array = new double *[blk_pairs];
  auto gemm_batch_m_array = new MKL_INT[blk_pairs];
  auto gemm_batch_n_array = new MKL_INT[blk_pairs];
  auto gemm_batch_k_array = new MKL_INT[blk_pairs];
  std::vector<QNBlock *> pnew_blks(blk_pairs, nullptr);
  auto gemm_batch_alpha_array = new double[blk_pairs];
  auto gemm_batch_beta_array = new double[blk_pairs] ();
  auto gemm_batch_grp_size_array = new MKL_INT[blk_pairs];
  for (long i = 0; i < blk_pairs; ++i) {
    gemm_batch_alpha_array[i] = alpha;
    gemm_batch_grp_size_array[i] = 1;
  }

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

            auto blk_data_transed_to_ctrct = DenseTensorTranspose(
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
        gemm_batch_m_array[blk_pair_cnt] = MKL_INT(
                                               ta_to_ctrct_blk_saved_dims[i]);
        gemm_batch_k_array[blk_pair_cnt] = MKL_INT(
                                               ta_to_ctrct_blk_ctrct_dims[i]);

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

            auto blk_data_transed_to_ctrct = DenseTensorTranspose(
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
        gemm_batch_n_array[blk_pair_cnt] = MKL_INT(
                                               tb_to_ctrct_blk_saved_dims[j]);
        gemm_batch_c_array[blk_pair_cnt] = pnew_blks[blk_pair_cnt]->data();

#ifdef GQTEN_CONTRACT_BLOCK_COUNTING
        std::cout << "[counting] blk_m_dim "
                  << std::setw(10) << std::left
                  << gemm_batch_m_array[blk_pair_cnt]
                  << "blk_k_dim "
                  << std::setw(10) << std::left
                  << gemm_batch_k_array[blk_pair_cnt]
                  << "blk_n_dim "
                  << std::setw(10) << std::left
                  << gemm_batch_n_array[blk_pair_cnt] << std::endl;
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

  GemmBatch(
      CblasRowMajor,
      gemm_batch_transa_array, gemm_batch_transb_array,
      gemm_batch_m_array,
      gemm_batch_n_array,
      gemm_batch_k_array,
      gemm_batch_alpha_array,
      gemm_batch_a_array, gemm_batch_k_array,
      gemm_batch_b_array, gemm_batch_n_array,
      gemm_batch_beta_array,
      gemm_batch_c_array, gemm_batch_n_array,
      blk_pairs,
      gemm_batch_grp_size_array); 

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
  delete[] gemm_batch_transa_array;
  delete[] gemm_batch_transb_array;
  delete[] gemm_batch_alpha_array;
  delete[] gemm_batch_beta_array;
  delete[] gemm_batch_grp_size_array;

  return pnew_blks;
}


GQTensor *InitCtrctedTen(
    const GQTensor &t1, const GQTensor &t2,
    const std::vector<long> &t1_ctrct_axes,
    const std::vector<long> &t2_ctrct_axes) {
  std::vector<Index> saved_idxs;
  const std::vector<Index> &t1_idxs  = t1.indexes;
  const std::vector<Index> &t2_idxs  = t2.indexes;
  for (size_t i = 0; i < t1_idxs.size(); ++i) {
    if (std::find(t1_ctrct_axes.begin(), t1_ctrct_axes.end(), i) ==
        t1_ctrct_axes.end()) {
      saved_idxs.push_back(t1.indexes[i]);
    }
  }
  for (size_t i = 0; i < t2_idxs.size(); ++i) {
    if (std::find(t2_ctrct_axes.begin(), t2_ctrct_axes.end(), i) ==
        t2_ctrct_axes.end()) {
      saved_idxs.push_back(t2.indexes[i]);
    }
  }
  auto pnew_ten = new GQTensor(saved_idxs);
  return pnew_ten;
}


void WrapCtrctBlocks(std::vector<QNBlock *> &pnew_blks, GQTensor *res_t) {
  auto nnew_blk = pnew_blks.size();   // nnew_blk: number of new blocks.
  if (res_t->indexes.size() == 0 && nnew_blk != 0) {  // Contract to scalar case.
    double scalar = 0;
    for (auto &pnew_blk : pnew_blks) {
      scalar += (pnew_blk->cdata()[0]);
      delete pnew_blk;
    }
    res_t->scalar = scalar;
  } else {                                            // Contract to tensor case.
    auto merged_blks = MergeCtrctBlks(pnew_blks);
    res_t->blocks() = merged_blks;
  }

#ifdef GQTEN_CONTRACT_BLOCK_COUNTING
  std::cout << "[counting] res # of blks "
            << res_t->cblocks().size() << std::endl;
#endif
}


std::vector<QNBlock *> MergeCtrctBlks(const std::vector<QNBlock *> &pblks) {
  std::vector<QNBlock *> merged_blks;

#ifdef GQTEN_TIMING_MODE
  Timer daxpy_timer("daxpy");
#endif

  for (auto &pnew_blk : pblks) {
    auto has_blk = false;  
    for (auto &pmerged_blk : merged_blks) {
      if (pmerged_blk->QNSectorSetHash() == pnew_blk->QNSectorSetHash()) {
        auto data_size = pnew_blk->size;
        assert(data_size == pmerged_blk->size);

#ifdef GQTEN_TIMING_MODE
  daxpy_timer.Restart();
#endif

        cblas_daxpy(
            data_size,
            1.0, pnew_blk->cdata(), 1,
            pmerged_blk->data(), 1);

#ifdef GQTEN_TIMING_MODE
  daxpy_timer.PrintElapsed(8);
#endif

        delete pnew_blk;
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      merged_blks.push_back(pnew_blk);
    }
  }
  return merged_blks;
}


void CalcBlkCtrctDimsInfo(
    const std::size_t blk_idx_in_ten, const QNBlock *pblk,
    const std::vector<long> &ctrct_axes,
    long *saved_dims, long *ctrct_dims) {
  long ctrct_dim = 1;
  long saved_dim = 1;
  for (long i = 0; i < pblk->ndim; ++i) {
    if (std::find(ctrct_axes.begin(), ctrct_axes.end(), i) !=
        ctrct_axes.end()) {
      ctrct_dim *= pblk->qnscts[i].dim;
    } else {
      saved_dim *= pblk->qnscts[i].dim;
    }
  } 
  saved_dims[blk_idx_in_ten] = saved_dim;
  ctrct_dims[blk_idx_in_ten] = ctrct_dim;
}


std::vector<const QNSector *> GetPNewBlkQNScts(
    const QNBlock *pta_blk, const QNBlock *ptb_blk,
    const std::vector<long> &ctrct_axes_a,
    const std::vector<long> &ctrct_axes_b) {
  std::vector<const QNSector *> pnew_blk_qnscts; 
  auto pta_blk_qnscts = pta_blk->qnscts.data();
  auto ptb_blk_qnscts = ptb_blk->qnscts.data();
  for (long i = 0; i < pta_blk->ndim; ++i) {
    if (std::find(ctrct_axes_a.begin(), ctrct_axes_a.end(), i) ==
        ctrct_axes_a.end()) {
      pnew_blk_qnscts.push_back(pta_blk_qnscts+i);
    }
  }
  for (long i = 0; i < ptb_blk->ndim; ++i) {
    if (std::find(ctrct_axes_b.begin(), ctrct_axes_b.end(), i) ==
        ctrct_axes_b.end()) {
      pnew_blk_qnscts.push_back(ptb_blk_qnscts+i);
    }
  }
  return pnew_blk_qnscts;
}


bool CtrctTransChecker(
    const std::vector<long> &ctrct_axes,
    const long ndim,
    const char position,
    std::vector<long> &transed_axes) {
  auto ctrct_ndim = ctrct_axes.size();
  std::vector<long> saved_axes(ndim-ctrct_ndim);
  std::size_t saved_axes_idx = 0;
  std::vector<long> ordered_axes(ndim);
  for (long i = 0; i < ndim; ++i) {
    if (std::find(ctrct_axes.begin(), ctrct_axes.end(), i) ==
        ctrct_axes.end()) {
      saved_axes[saved_axes_idx] = i;
      saved_axes_idx++;
    }
    ordered_axes[i] = i;
  }
  switch (position) {
    case 'a':
      transed_axes = saved_axes;
      transed_axes.insert(
          transed_axes.end(),
          ctrct_axes.begin(), ctrct_axes.end());
      if (transed_axes != ordered_axes) { return true; }
      break;
    case 'b':
      transed_axes = ctrct_axes;
      transed_axes.insert(
          transed_axes.end(),
          saved_axes.begin(), saved_axes.end());
      if (transed_axes != ordered_axes) { return true; }
      break;
    default:
      std::cout << "position must be 'a' or 'b', but" << position << std::endl;
      exit(1);
  }
  return false;
}


std::vector<std::size_t> GenBlksPartHashTable(
    const std::vector<QNBlock *> &blks, const std::vector<long> &ctrct_axes) {
  std::vector<std::size_t> part_hash_table(blks.size());
  for (std::size_t i = 0; i < blks.size(); i++) {
    part_hash_table[i] = blks[i]->PartHash(ctrct_axes);
  }
  return part_hash_table;
}
} /* gqten */ 
