// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-05 14:10
* 
* Description: GraceQ/tensor project. Numerical functions for GQTensor, src file.
*/
#include "ten_numer_func.h"
#include "gqten/gqten.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

#include "mkl.h"

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>

namespace gqten {


// Tensor contraction.
GQTensor *Contract(
    const GQTensor &ta, const GQTensor &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto ctrct_axes_a = axes_set[0];
  auto ctrct_axes_b = axes_set[1];

  // Blocks contraction batch.
  std::vector<QNBlock *> pnew_blks;
  if (ta.BlksConstRef().size() > 0 && tb.BlksConstRef().size() > 0) {

    Timer blks_ctrct_batch_timer("blks_ctrct_batch");
    blks_ctrct_batch_timer.Restart();
    pnew_blks = BlksCtrctBatch(
        ctrct_axes_a, ctrct_axes_b,
        1.0, ta.BlksConstRef(), tb.BlksConstRef());
    std::cout << std::fixed;
    blks_ctrct_batch_timer.PrintElapsed();

  }

  auto res_t  = InitCtrctedTen(ta, tb, ctrct_axes_a, ctrct_axes_b);

  // Wrap blocks.
  Timer blks_wrap_timer("blks_wrap");
  blks_wrap_timer.Restart();
  WrapCtrctBlks(pnew_blks, res_t);
  blks_wrap_timer.PrintElapsed();

  std::cout << res_t->BlksConstRef().size() << std::endl;

  return res_t;
}


GQTensor *InitCtrctedTen(
    const GQTensor &t1, const GQTensor &t2,
    const std::vector<long> &t1_ctrct_idxs,
    const std::vector<long> &t2_ctrct_idxs) {
  std::vector<Index> saved_idxs;
  const std::vector<Index> &t1_idxs  = t1.indexes;
  const std::vector<Index> &t2_idxs  = t2.indexes;
  for (size_t i = 0; i < t1_idxs.size(); ++i) {
    if (std::find(t1_ctrct_idxs.begin(), t1_ctrct_idxs.end(), i) == t1_ctrct_idxs.end()) {
      saved_idxs.push_back(t1.indexes[i]);
    }
  }
  for (size_t i = 0; i < t2_idxs.size(); ++i) {
    if (std::find(t2_ctrct_idxs.begin(), t2_ctrct_idxs.end(), i) == t2_ctrct_idxs.end()) {
      saved_idxs.push_back(t2.indexes[i]);
    }
  }
  auto pnew_ten = new GQTensor(saved_idxs);
  return pnew_ten;
}


std::vector<QNBlock *> BlksCtrctBatch(
    const std::vector<long> &ctrct_axes_a,
    const std::vector<long> &ctrct_axes_b,
    const double alpha,
    const std::vector<QNBlock *> &ta_blks,
    const std::vector<QNBlock *> &tb_blks) {
  // Data prepare.
  std::cout << std::fixed;
  Timer blk_match_timer("blk_match");
  blk_match_timer.Restart();
  auto ta_blks_num = ta_blks.size();
  auto tb_blks_num = tb_blks.size();
  assert(ta_blks_num > 0);
  assert(tb_blks_num > 0);
  // Check whether need transpose.
  std::vector<long> transed_axes_a, transed_axes_b;
  bool ta_need_trans = CtrctTransCheck(
                           ctrct_axes_a,
                           ta_blks[0]->ndim, 'a',
                           transed_axes_a);
  bool tb_need_trans = CtrctTransCheck(
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
  std::cout << ta_blks_num << " " << tb_blks_num << " " << blk_pairs << std::endl;
  // No match, return empty vector.
  if (blk_pairs == 0) {
    return std::vector<QNBlock *>();
  }

  // Initialize data.
  // Data with size of tensor blocks.
  auto ta_to_ctrct_blks = new const double *[ta_blks_num] ();
  auto tb_to_ctrct_blks = new const double *[tb_blks_num] ();
  auto ta_to_ctrct_blk_saved_dims = new long[ta_blks_num] ();
  auto tb_to_ctrct_blk_saved_dims = new long[tb_blks_num] ();
  auto ta_to_ctrct_blk_ctrct_dims = new long[ta_blks_num] ();
  auto tb_to_ctrct_blk_ctrct_dims = new long[tb_blks_num] ();
  // Data with size of block pairs.
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
        if (pnew_blks[blk_pair_cnt]->DataConstRef() == nullptr) {
          pnew_blks[blk_pair_cnt]->DataRef() = new double[1];
        }

        // Deal with ta block.
        if (ta_to_ctrct_blks[i] == nullptr) {
          // Calculate dimensions information.
          CalcCtrctBlkDimInfo(
              i, ta_blks[i], ctrct_axes_a,
              ta_to_ctrct_blk_saved_dims, ta_to_ctrct_blk_ctrct_dims);
          // Generate contraction data.
          if (ta_need_trans) {
            auto blk_data_transed_to_ctrct = TransposeData(
                ta_blks[i]->DataConstRef(),
                ta_blks[i]->ndim,
                ta_blks[i]->size,
                ta_blks[i]->shape,
                transed_axes_a);
            ta_to_ctrct_blks[i] = blk_data_transed_to_ctrct;
          } else {
            ta_to_ctrct_blks[i] = ta_blks[i]->DataConstRef();
          }
        }
        // Assign gemm_batch parameters.
        gemm_batch_a_array[blk_pair_cnt] = ta_to_ctrct_blks[i];
        gemm_batch_m_array[blk_pair_cnt] = MKL_INT(ta_to_ctrct_blk_saved_dims[i]);
        gemm_batch_k_array[blk_pair_cnt] = MKL_INT(ta_to_ctrct_blk_ctrct_dims[i]);

        // Deal with tb block.
        if (tb_to_ctrct_blks[j] == nullptr) {
          // Calculate dimensions information.
          CalcCtrctBlkDimInfo(
              j, tb_blks[j], ctrct_axes_b,
              tb_to_ctrct_blk_saved_dims, tb_to_ctrct_blk_ctrct_dims);
          // Generate contraction data.
          if (tb_need_trans) {
            auto blk_data_transed_to_ctrct = TransposeData(
                tb_blks[j]->DataConstRef(),
                tb_blks[j]->ndim,
                tb_blks[j]->size,
                tb_blks[j]->shape,
                transed_axes_b);
            tb_to_ctrct_blks[j] = blk_data_transed_to_ctrct;
          } else {
            tb_to_ctrct_blks[j] = tb_blks[j]->DataConstRef();
          }
        }
        // Assign gemm_batch parameters.
        gemm_batch_b_array[blk_pair_cnt] = tb_to_ctrct_blks[j];
        gemm_batch_n_array[blk_pair_cnt] = MKL_INT(tb_to_ctrct_blk_saved_dims[j]);
        gemm_batch_c_array[blk_pair_cnt] = pnew_blks[blk_pair_cnt]->DataRef();

        ++blk_pair_cnt;
      }
    }
  }
  blk_match_timer.PrintElapsed();

  // Call gemm_batch function.
  Timer dgemm_batch_timer("gemm_batch");
  dgemm_batch_timer.Restart();
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
  dgemm_batch_timer.PrintElapsed();

  // Free temporary variables.
  if (ta_need_trans) {
    for (std::size_t i = 0; i < ta_blks_num; ++i) {
      delete[] ta_to_ctrct_blks[i];
    }
  }
  if (tb_need_trans) {
    for (std::size_t i = 0; i < tb_blks_num; ++i) {
      delete[] tb_to_ctrct_blks[i];
    }
  }
  delete[] ta_to_ctrct_blks;
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


void WrapCtrctBlks(std::vector<QNBlock *> &pnew_blks, GQTensor *res_t) {
  auto nnew_blk = pnew_blks.size();   // nnew_blk: number of new blocks.
  if (res_t->indexes.size() == 0) {   // Contract to scalar case.
    if (nnew_blk == 0) {    // No matched block pair.
    } else {                          // Has matched block pair.
      double scalar = 0;
      for (auto &pnew_blk : pnew_blks) {
        scalar += (pnew_blk->DataConstRef()[0]);
        delete pnew_blk;
      }
      res_t->scalar = scalar;
    }
  } else {                            // Contract to tensor case.
    auto merged_blks = MergeCtrctBlks(pnew_blks);
    res_t->BlksRef() = merged_blks;
  }
}


std::vector<QNBlock *> MergeCtrctBlks(const std::vector<QNBlock *> &pblks) {
  std::vector<QNBlock *> merged_blks;
  for (auto &pnew_blk : pblks) {
    auto has_blk = false;  
    for (auto &pmerged_blk : merged_blks) {
      if (pnew_blk->QNSectorSetHash() == pmerged_blk->QNSectorSetHash()) {
        auto data_size = pnew_blk->size;
        assert(data_size == pmerged_blk->size);
        cblas_daxpy(
            data_size,
            1.0, pnew_blk->DataConstRef(), 1,
            pmerged_blk->DataRef(), 1);
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


void CalcCtrctBlkDimInfo(
    const std::size_t blk_idx, const QNBlock *pblk,
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
  saved_dims[blk_idx] = saved_dim;
  ctrct_dims[blk_idx] = ctrct_dim;
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


// Tensor linear combination.
// Do the operation: res += (coefs[0]*ts[0] + coefs[1]*ts[1] + ...).
void LinearCombine(
    const std::vector<double> &coefs,
    const std::vector<GQTensor *> &ts,
    GQTensor *res) {
  auto nt = ts.size();
  assert(coefs.size() == nt);
  for (std::size_t i = 0; i < nt; ++i) {
    LinearCombineOneTerm(coefs[i], ts[i], res);
  }
}


void LinearCombine(
    const std::size_t size,
    const double *coefs,
    const std::vector<GQTensor *> &ts,
    GQTensor *res) {
  for (std::size_t i = 0; i < size; i++) {
    LinearCombineOneTerm(coefs[i], ts[i], res);
  }
}



void LinearCombineOneTerm(const double coef, const GQTensor *t, GQTensor *res) {
  for (auto &blk : t->BlksConstRef()) {
    auto has_blk = false;
    for (auto &res_blk : res->BlksRef()) {
      if (res_blk->QNSectorSetHash() == blk->QNSectorSetHash()) {
        auto size = res_blk->size;
        assert(size == blk->size);
        auto blk_data = blk->DataConstRef();
        auto res_blk_data = res_blk->DataRef();
        cblas_daxpy(size, coef, blk_data, 1, res_blk_data, 1);
        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto new_blk = new QNBlock(*blk);
      auto size = new_blk->size;
      auto new_blk_data = new_blk->DataRef();
      for (std::size_t i = 0; i < size; ++i) {
        new_blk_data[i] *= coef;
      }
      res->BlksRef().push_back(new_blk);
    }
  }
}


// Tensor SVD.
SvdRes Svd(
    const GQTensor &t,
    const long &ldims, const long &rdims,
    const QN &ldiv, const QN &rdiv,
    const double &cutoff, const long &Dmin, const long &Dmax) {
  assert((ldims + rdims) == t.indexes.size());

  Timer svd_merge_blks_timer("svd_merge_blks");
  svd_merge_blks_timer.Restart();
  auto merged_blocks = SvdMergeBlocks(t, ldims, rdims);
  svd_merge_blks_timer.PrintElapsed();

  Timer svd_blks_svd_timer("svd_blks_svd");
  svd_blks_svd_timer.Restart();
  auto trunc_blk_svd_res = TruncatedBlockSvd(merged_blocks, cutoff, Dmin, Dmax);
  svd_blks_svd_timer.PrintElapsed();
  
  Timer svd_wrap_blks("svd_wrap_blks");
  svd_wrap_blks.Restart();
  auto res = SvdWrapBlocks(
                 trunc_blk_svd_res,
                 ldiv, rdiv,
                 t.indexes,
                 ldims, rdims);
  svd_wrap_blks.PrintElapsed();
  return res;
}


SvdRes Svd(
    const GQTensor &t,
    const long ldims, const long rdims,
    const QN &ldiv, const QN &rdiv) {
  auto t_shape = t.shape;
  long lsize = 1;
  long rsize = 1;
  for (std::size_t i = 0; i < t_shape.size(); ++i) {
    if (i < ldims) {
      lsize *= t_shape[i];
    } else {
      rsize *= t_shape[i];
    }
  }
  auto D = ((lsize >= rsize) ? lsize : rsize);
  return Svd(
      t,
      ldims, rdims,
      ldiv, rdiv,
      0, D, D);
}


PartDivsAndMergedBlk SvdMergeBlocks(
    const GQTensor &t, const long &ldims, const long &rdims) {
  PartDivsAndBipartiteBlkDatas tomerge_blkdatas;
  PartDivsEqual partdivs_equaler;
  for (auto &blk : t.BlksConstRef()) {
    auto lqnscts = SliceFromBegin(blk->qnscts, ldims);
    auto rqnscts = SliceFromEnd(blk->qnscts, rdims);
    auto lpartdiv = CalcDiv(lqnscts, SliceFromBegin(t.indexes, ldims));
    auto rpartdiv = CalcDiv(rqnscts, SliceFromEnd(t.indexes, rdims));
    auto partdivs = std::make_pair(lpartdiv, rpartdiv);
    auto blk_data = BipartiteBlkData(lqnscts, rqnscts, blk->DataConstRef());
    auto has_partdivs = false;
    for (auto &kv : tomerge_blkdatas) {
      if (partdivs_equaler(kv.first, partdivs)) {
        tomerge_blkdatas[kv.first].push_back(blk_data);
        has_partdivs = true;
        break;
      }
    }
    if (!has_partdivs) {
      tomerge_blkdatas[partdivs] = {blk_data};
    }
  }
  PartDivsAndMergedBlk merged_blocks;
  for (auto &kv : tomerge_blkdatas) {
    merged_blocks.emplace(kv.first, SvdMergeBlk(kv.second));
  }
  return merged_blocks;
}


MergedBlk SvdMergeBlk(const std::vector<BipartiteBlkData> &blkdatas) {
  QNSectorsSet lqnscts_set;
  QNSectorsSet rqnscts_set;
  long extra_ldim = 0;
  long extra_rdim = 0;
  double *mat = nullptr;
  for (auto &bipblk_data : blkdatas) {
    const std::vector<QNSector> &lqnscts = bipblk_data.lqnscts;
    const std::vector<QNSector> &rqnscts = bipblk_data.rqnscts;
    auto data = bipblk_data.data;
    auto intra_ldim = MulDims(lqnscts);
    auto intra_rdim = MulDims(rqnscts);
    auto loffset = OffsetInQNSectorsSet(lqnscts, lqnscts_set);
    auto roffset = OffsetInQNSectorsSet(rqnscts, rqnscts_set);
    if (loffset < extra_ldim && roffset < extra_rdim) {
      CpySubMat(
          mat, extra_ldim, extra_rdim,
          data, intra_ldim, intra_rdim,
          loffset, roffset);
    } else if (loffset == extra_ldim && roffset < extra_rdim) {
      auto temp_mat = new double [(extra_ldim+intra_ldim)*extra_rdim] ();
      std::memcpy(temp_mat, mat, (extra_ldim*extra_rdim)*sizeof(double));
      delete [] mat;
      mat = temp_mat;
      CpySubMat(
          mat, extra_ldim+intra_ldim, extra_rdim,
          data, intra_ldim, intra_rdim,
          loffset, roffset);
      extra_ldim += intra_ldim;
      lqnscts_set.push_back(lqnscts);
    } else if (loffset < extra_ldim && roffset == extra_rdim) {
      auto temp_mat = new double [extra_ldim*(extra_rdim+intra_rdim)] ();
      CpySubMat(
          temp_mat, extra_ldim, extra_rdim+intra_rdim,
          mat, extra_ldim, extra_rdim,
          0, 0);
      delete [] mat;
      mat = temp_mat;
      CpySubMat(
          mat, extra_ldim, extra_rdim+intra_rdim,
          data, intra_ldim, intra_rdim,
          loffset, roffset);
      extra_rdim += intra_rdim;
      rqnscts_set.push_back(rqnscts);
    } else if (loffset == extra_ldim && roffset == extra_rdim) {
      auto temp_mat = new double [(extra_ldim+intra_ldim)*
                                  (extra_rdim+intra_rdim)] ();
      if (mat != nullptr) {
        CpySubMat(
            temp_mat, extra_ldim+intra_ldim, extra_rdim+intra_rdim,
            mat, extra_ldim, extra_rdim,
            0, 0);
        delete [] mat;
      }
      mat = temp_mat;
      CpySubMat(
          mat, extra_ldim+intra_ldim, extra_rdim+intra_rdim,
          data, intra_ldim, intra_rdim,
          loffset, roffset);
      extra_ldim += intra_ldim;
      extra_rdim += intra_rdim;
      lqnscts_set.push_back(lqnscts);
      rqnscts_set.push_back(rqnscts);
    }
  }
  return MergedBlk(lqnscts_set, rqnscts_set, mat, extra_ldim, extra_rdim);
}


TruncBlkSvdData TruncatedBlockSvd(
    const PartDivsAndMergedBlk &merged_blocks,
    const double &cutoff,
    const long &dmin,
    const long &dmax) {
  PartDivsAndBlkSvdData svd_data;
  std::vector<double> singular_values;
  for (auto &kv : merged_blocks) {
    auto raw_svd_res = MatSvd(
                           kv.second.mat,
                           kv.second.mat_ldim, kv.second.mat_rdim);
    delete[] kv.second.mat;
    if (raw_svd_res.info == 0) {
      auto sdim = std::min(kv.second.mat_ldim, kv.second.mat_rdim);
      svd_data.emplace(
          kv.first,
          BlkSvdData(
              kv.second.lqnscts_set, kv.second.rqnscts_set,
              raw_svd_res.u, raw_svd_res.s, raw_svd_res.v,
              kv.second.mat_ldim, sdim, kv.second.mat_rdim));
      for (long i = 0; i < sdim; i++) {
        auto sv = raw_svd_res.s[i];
        if (sv != 0.0) { singular_values.push_back(sv); }
      }
    } else {
      std::cout << "LAPACKE_dgesdd error." << std::endl;
      exit(1);
    }
  }
  long total_dim = singular_values.size();
  if (total_dim <= dmin) {
    TruncBlkSvdData truncated_blk_svd_data;
    truncated_blk_svd_data.trunc_blks = svd_data;
    truncated_blk_svd_data.trunc_err = 0.0;
    truncated_blk_svd_data.kept_dim = total_dim;
    return truncated_blk_svd_data;
  }
  std::sort(singular_values.begin(), singular_values.end());
  auto sv_squares = SquareVec(singular_values);
  auto normalized_sv_squares = NormVec(sv_squares);
  long kept_dim = dmin;
  double trunc_err = std::accumulate(
                         normalized_sv_squares.begin(),
                         normalized_sv_squares.end()-kept_dim,
                         0.0);
  long next_kept_dim;
  double next_trunc_err;
  while (true) {
    next_kept_dim = kept_dim + 1;
    next_trunc_err = trunc_err -
                     normalized_sv_squares[total_dim-(kept_dim+1)];
    if (DoubleEq(next_trunc_err, 0)) { next_trunc_err = 0; }
    if (next_kept_dim > total_dim ||
        next_kept_dim > dmax ||
        next_trunc_err < cutoff) {
      break;
    } else {
      kept_dim = next_kept_dim;
      trunc_err = next_trunc_err;
    }
  }
  assert(kept_dim <= total_dim);
  auto kept_smallest_sv = singular_values[total_dim-kept_dim];
  PartDivsAndBlkSvdData truncated_blocks;
  SvdTruncteBlks(svd_data, kept_smallest_sv, truncated_blocks);
  TruncBlkSvdData truncated_blk_svd_data;
  truncated_blk_svd_data.trunc_blks = truncated_blocks;
  truncated_blk_svd_data.trunc_err = trunc_err;
  truncated_blk_svd_data.kept_dim = kept_dim;
  return truncated_blk_svd_data;
}


void SvdTruncteBlks(
    PartDivsAndBlkSvdData &svd_data,
    const double kept_smallest_sv,
    PartDivsAndBlkSvdData &truncated_svd_data) {
  for (auto &kv : svd_data) {
    double *trunced_u = nullptr;
    double *trunced_s = nullptr;
    double *trunced_v = nullptr;
    auto blk_kept_dim = 0;
    for (long i = 0; i < kv.second.sdim; ++i) {
      if (kv.second.s[i] < kept_smallest_sv) {
        break;
      } else {
        ++blk_kept_dim;
      }
    }
    if (blk_kept_dim == 0) {
      delete [] kv.second.u; kv.second.u = nullptr;
      delete [] kv.second.s; kv.second.s = nullptr;
      delete [] kv.second.v; kv.second.v = nullptr;
    } else if (blk_kept_dim < kv.second.sdim) {
      trunced_s = new double [blk_kept_dim];
      std::memcpy(trunced_s, kv.second.s, blk_kept_dim*sizeof(double));
      MatTrans(kv.second.uldim, kv.second.sdim, kv.second.u);
      trunced_u = MatGetRows(
                      kv.second.u, kv.second.sdim, kv.second.uldim,
                      0, blk_kept_dim);
      MatTrans(blk_kept_dim, kv.second.uldim, trunced_u);
      trunced_v = MatGetRows(
                      kv.second.v, kv.second.sdim, kv.second.vrdim,
                      0, blk_kept_dim);
      truncated_svd_data.emplace(
          kv.first,
          BlkSvdData(
              kv.second.lqnscts_set, kv.second.rqnscts_set,
              trunced_u, trunced_s, trunced_v,
              kv.second.uldim, blk_kept_dim, kv.second.vrdim));
      delete [] kv.second.u; kv.second.u = nullptr;
      delete [] kv.second.s; kv.second.s = nullptr;
      delete [] kv.second.v; kv.second.v = nullptr;
    } else {
      trunced_s = kv.second.s;
      trunced_u = kv.second.u;
      trunced_v = kv.second.v;
      truncated_svd_data.emplace(
          kv.first,
          BlkSvdData(
              kv.second.lqnscts_set, kv.second.rqnscts_set,
              trunced_u, trunced_s, trunced_v,
              kv.second.uldim, blk_kept_dim, kv.second.vrdim));
    }
  }
}


SvdRes SvdWrapBlocks(
    TruncBlkSvdData &truncated_blk_svd_data,
    const QN &ldiv, const QN &rdiv,
    const std::vector<Index> &indexes,
    const long &ldims, const long &rdims) {
  std::vector<QNBlock *> ublocks, sblocks, vblocks;
  std::vector<QNSector> sblk_qnscts;
  for (auto &kv : truncated_blk_svd_data.trunc_blks) {
    // Create s block.
    auto sblk_qn = ldiv - kv.first.first;
    auto sblk_qnsct = QNSector(sblk_qn, kv.second.sdim);
    sblk_qnscts.push_back(sblk_qnsct);
    auto sblock = new QNBlock({sblk_qnsct, sblk_qnsct});
    GenDiagMat(kv.second.s, kv.second.sdim, sblock->DataRef());
    delete [] kv.second.s; kv.second.s = nullptr;
    sblocks.push_back(sblock);
    // Create u block.
    long u_row_offset = 0;
    for (auto &lqnscts : kv.second.lqnscts_set) {
      auto u_row_dim = MulDims(lqnscts);
      auto ublock_qnscts = lqnscts; ublock_qnscts.push_back(sblk_qnsct);
      auto ublock = new QNBlock(ublock_qnscts);
      MatGetRows(
          kv.second.u, kv.second.uldim, kv.second.sdim,
          u_row_offset, u_row_dim,
          ublock->DataRef());
      ublocks.push_back(ublock);
      u_row_offset += u_row_dim;
    }
    delete [] kv.second.u; kv.second.u = nullptr;
    // Create v block.
    long v_col_offset = 0;
    auto transed_v = MatTrans(kv.second.v, kv.second.sdim, kv.second.vrdim);
    for (auto &rqnscts : kv.second.rqnscts_set) {
      auto v_col_dim = MulDims(rqnscts);
      auto vblock_qnscts = rqnscts;
      vblock_qnscts.insert(vblock_qnscts.begin(), sblk_qnsct);
      auto vblock = new QNBlock(vblock_qnscts);
      MatGetRows(
          transed_v, kv.second.uldim, kv.second.sdim,
          v_col_offset, v_col_dim,
          vblock->DataRef());
      MatTrans(v_col_dim, kv.second.sdim, vblock->DataRef());
      vblocks.push_back(vblock);
      v_col_offset += v_col_dim;
    }
    delete [] transed_v; transed_v = nullptr;
    delete [] kv.second.v; kv.second.v = nullptr;
  }
  auto s_index_in = Index(sblk_qnscts, IN);
  auto s_index_out = Index(sblk_qnscts, OUT);
  auto s = new GQTensor({s_index_in, s_index_out});
  s->BlksRef() = sblocks;
  auto u_indexes = SliceFromBegin(indexes, ldims);
  u_indexes.push_back(s_index_out);
  auto u = new GQTensor(u_indexes);
  u->BlksRef() = ublocks;
  auto v_indexes = SliceFromEnd(indexes, rdims);
  v_indexes.insert(v_indexes.begin(), s_index_in);
  auto v = new GQTensor(v_indexes);
  v->BlksRef() = vblocks;
  return SvdRes(
             u, s, v,
             truncated_blk_svd_data.trunc_err,
             truncated_blk_svd_data.kept_dim);
}


// Operations for matrix.
RawSvdRes MatSvd(double *mat, const long &mld, const long &mrd) {
  auto m = mld;
  auto n = mrd;
  auto lda = n;
  long ldu, ldvt;
  double *s;
  double *vt;
  if (m >= n) {
    ldu = n;
    ldvt = n;
    s = new double [n];
    vt = new double [ldvt*n];
  } else {
    ldu = m;
    ldvt = n;
    s = new double [m];
    vt = new double [ldvt*m];
  }
  double *u = new double [ldu*m];
  auto info = LAPACKE_dgesdd(
      LAPACK_ROW_MAJOR, 'S',
      m, n,
      mat, lda,
      s,
      u, ldu,
      vt, ldvt);
  RawSvdRes raw_svd_res;
  raw_svd_res.info = info;
  raw_svd_res.u = u;
  raw_svd_res.s = s;
  raw_svd_res.v = vt;
  return raw_svd_res;
}


double *MatTrans(
    const double *mat, const long &mat_ldim, const long &mat_rdim) {
  auto size = mat_ldim*mat_rdim;
  auto transed_mat = new double [size];
  std::memcpy(transed_mat, mat, size*sizeof(double));
  mkl_dimatcopy(
      'R', 'T',
      mat_ldim, mat_rdim,
      1.0,
      transed_mat,
      mat_rdim, mat_ldim);
  return transed_mat;
}


void MatTrans(const long &mat_ldim, const long &mat_rdim, double *mat) {
  mkl_dimatcopy(
      'R', 'T',
      mat_ldim, mat_rdim,
      1.0,
      mat,
      mat_rdim, mat_ldim);
}

void GenDiagMat(const double *diag_v, const long &diag_v_dim, double *full_mat) {
  for (long i = 0; i < diag_v_dim; ++i) {
    *(full_mat + (i*diag_v_dim + i)) = diag_v[i];
  }
}


double *MatGetRows(
    const double *mat, const long &rows, const long &cols,
    const long &from, const long &num_rows) {
  auto new_size = num_rows*cols;
  auto new_mat = new double [new_size]; 
  std::memcpy(new_mat, mat+(from*cols), new_size*sizeof(double));
  return new_mat;
}


void MatGetRows(
    const double *mat, const long &rows, const long &cols,
    const long &from, const long &num_rows,
    double *new_mat) {
  auto new_size = num_rows*cols;
  std::memcpy(new_mat, mat+(from*cols), new_size*sizeof(double));
}


// Tensor contraction helpers.
bool CtrctTransCheck(
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
