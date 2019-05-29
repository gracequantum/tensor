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

  std::vector<QNBlock *> pnew_blks;
  if (ta.BlksConstRef().size() > 0 && tb.BlksConstRef().size() > 0) {
    pnew_blks = BlksCtrctBatch(
        ctrct_axes_a, ctrct_axes_b,
        1.0, ta.BlksConstRef(), tb.BlksConstRef());
  }

  auto res_t  = InitCtrctedTen(ta, tb, ctrct_axes_a, ctrct_axes_b);
  WrapCtrctBlks(pnew_blks, res_t);
  return res_t;
}


GQTensor *SeriesContract(
    const std::vector<GQTensor *> &ts,
    const std::vector<std::pair<std::vector<long>,
                                std::vector<long>>> &ctrct_axes_series) {
  auto nt = ts.size();
  auto nctrct = ctrct_axes_series.size();
  assert(nt >= 2);
  assert(nt == nctrct + 1);
  std::vector<QNBlock *> res_blks = ts[0]->BlksRef();
  GQTensor *res_t = ts[0];
  for (std::size_t i = 0; i < ctrct_axes_series.size(); ++i) {
    SeriesBlksCtrct(
        i, nctrct,
        res_blks, res_t,
        ts[i+1], ctrct_axes_series[i]);
    if (res_blks.size() == 0 && i != nctrct-1) {
      return res_t;
    }
  }
  WrapCtrctBlks(res_blks, res_t);
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
  std::vector<long> transed_axes_a, transed_axes_b;
  assert(ta_blks.size() > 0);
  assert(tb_blks.size() > 0);
  bool ta_need_trans = CtrctTransCheck(ctrct_axes_a, ta_blks[0]->ndim, '1', transed_axes_a);
  bool tb_need_trans = CtrctTransCheck(ctrct_axes_b, tb_blks[0]->ndim, '2', transed_axes_b);
  auto ta_blks_part_hash_table = GenBlksPartHashTable(ta_blks, ctrct_axes_a);
  auto tb_blks_part_hash_table = GenBlksPartHashTable(tb_blks, ctrct_axes_b);
  auto ta_blks_num = ta_blks.size();
  auto tb_blks_num = tb_blks.size();
  std::vector<const double *> ta_to_ctrct_blk_datas(ta_blks_num, nullptr); 
  std::vector<const double *> tb_to_ctrct_blk_datas(tb_blks_num, nullptr);
  std::vector<long> ta_to_ctrct_blk_saved_dims(ta_blks_num, NULL);
  std::vector<long> tb_to_ctrct_blk_saved_dims(tb_blks_num, NULL);
  std::vector<long> ta_to_ctrct_blk_ctrct_dims(ta_blks_num, NULL);
  std::vector<long> tb_to_ctrct_blk_ctrct_dims(tb_blks_num, NULL);
  std::vector<QNBlock *> pnew_blks;
  // For MKL ?gemm_batch parameters.
  std::vector<const double *> gemm_batch_a_array_vec;
  std::vector<const double *> gemm_batch_b_array_vec;
  std::vector<double *> gemm_batch_c_array_vec;
  std::vector<MKL_INT> gemm_batch_m_array_vec;
  std::vector<MKL_INT> gemm_batch_n_array_vec;
  std::vector<MKL_INT> gemm_batch_k_array_vec;
  long gemm_batch_grp_cnt = 0;
  for (std::size_t i = 0; i < ta_blks_num; ++i) {
    for (std::size_t j = 0; j < tb_blks_num; ++j) {
      if (ta_blks_part_hash_table[i] == tb_blks_part_hash_table[j]) {
        auto pnew_blk_qnscts = GetPNewBlkQNScts(
                                   ta_blks[i], tb_blks[j],
                                   ctrct_axes_a, ctrct_axes_b);
        pnew_blks.push_back(new QNBlock(pnew_blk_qnscts));
        if (pnew_blks[gemm_batch_grp_cnt]->DataConstRef() == nullptr) {
          pnew_blks[gemm_batch_grp_cnt]->DataRef() = new double[1];
        } 
        if (ta_to_ctrct_blk_datas[i] == nullptr) {
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
            ta_to_ctrct_blk_datas[i] = blk_data_transed_to_ctrct;
          } else {
            ta_to_ctrct_blk_datas[i] = ta_blks[i]->DataConstRef();
          }
        } 
        // Assign gemm_batch parameters.
        gemm_batch_a_array_vec.push_back(ta_to_ctrct_blk_datas[i]);
        gemm_batch_m_array_vec.push_back(ta_to_ctrct_blk_saved_dims[i]);
        gemm_batch_k_array_vec.push_back(ta_to_ctrct_blk_ctrct_dims[i]);

        if (tb_to_ctrct_blk_datas[j] == nullptr) {
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
            tb_to_ctrct_blk_datas[j] = blk_data_transed_to_ctrct;
          } else {
            tb_to_ctrct_blk_datas[j] = tb_blks[j]->DataConstRef();
          }
        }
        // Assign gemm_batch parameters.
        gemm_batch_b_array_vec.push_back(tb_to_ctrct_blk_datas[j]);
        gemm_batch_n_array_vec.push_back(tb_to_ctrct_blk_saved_dims[j]);
        gemm_batch_c_array_vec.push_back(
            pnew_blks[gemm_batch_grp_cnt]->DataRef());
        ++gemm_batch_grp_cnt;
      }
    }
  }
  std::vector<CBLAS_TRANSPOSE> transa_array_vec(
                                   gemm_batch_grp_cnt, CblasNoTrans);
  std::vector<CBLAS_TRANSPOSE> transb_array_vec(
                                   gemm_batch_grp_cnt, CblasNoTrans);
  std::vector<double> alpha_array_vec(gemm_batch_grp_cnt, alpha);
  std::vector<double> beta_array_vec(gemm_batch_grp_cnt, 0.0);
  std::vector<MKL_INT> group_size_vec(gemm_batch_grp_cnt, 1);

  // Call MKL ?gemm_batch function.
  if (gemm_batch_grp_cnt != 0) {
    cblas_dgemm_batch(
        CblasRowMajor,
        transa_array_vec.data(), transb_array_vec.data(),
        gemm_batch_m_array_vec.data(),
        gemm_batch_n_array_vec.data(),
        gemm_batch_k_array_vec.data(),
        alpha_array_vec.data(),
        gemm_batch_a_array_vec.data(), gemm_batch_k_array_vec.data(),
        gemm_batch_b_array_vec.data(), gemm_batch_n_array_vec.data(),
        beta_array_vec.data(),
        gemm_batch_c_array_vec.data(), gemm_batch_n_array_vec.data(),
        gemm_batch_grp_cnt,
        group_size_vec.data());
  }
  // Free temporary block data.
  if (ta_need_trans) {
    for (auto &blk_data : ta_to_ctrct_blk_datas) { delete [] blk_data; }
  }
  if (tb_need_trans) {
    for (auto &blk_data : tb_to_ctrct_blk_datas) { delete [] blk_data; }
  }
  return pnew_blks;
}


void SeriesBlksCtrct(
    const std::size_t i, const std::size_t nctrct,
    std::vector<QNBlock *> &pres_blks, GQTensor * &rpres_t,
    const GQTensor *pnew_t,
    const std::pair<std::vector<long>, std::vector<long>> &ctrct_axes) {
  std::vector<QNBlock *> pnew_blks;
  if (pres_blks.size() > 0 && pnew_t->BlksConstRef().size() > 0) {
    pnew_blks = BlksCtrctBatch(
        ctrct_axes.first, ctrct_axes.second,
        1.0,
        pres_blks, pnew_t->BlksConstRef());
    auto pnew_res_t = InitCtrctedTen(
        *rpres_t, *pnew_t,
        ctrct_axes.first, ctrct_axes.second);

    if (i != 0) {
      FreeBlks(pres_blks);
      delete rpres_t;
    }

    if (i != nctrct-1) {
      pres_blks = MergeCtrctBlks(pnew_blks);
    } else {
      pres_blks = pnew_blks;
    }
    rpres_t = pnew_res_t;
  } else {
    auto pnew_res_t = InitCtrctedTen(
        *rpres_t, *pnew_t,
        ctrct_axes.first, ctrct_axes.second);
    if (i != 0) {
      FreeBlks(pres_blks);
      delete rpres_t;
    }
    pres_blks = pnew_blks;
    rpres_t = pnew_res_t;
  }
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
        ArrayElemAttach(
            pmerged_blk->DataRef(), data_size,
            pnew_blk->DataConstRef());
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
    std::vector<long> &saved_dims, std::vector<long> &ctrct_dims) {
  long ctrct_dim = 1;
  long saved_dim = 1;
  for (long i = 0; i < pblk->ndim; ++i) {
    if (std::find(ctrct_axes.begin(), ctrct_axes.end(), i) != ctrct_axes.end()) {
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


// Tensor SVD.
SvdRes Svd(
    const GQTensor &t,
    const long &ldims, const long &rdims,
    const QN &ldiv, const QN &rdiv,
    const double &cutoff, const long &Dmin, const long &Dmax) {
  assert((ldims + rdims) == t.indexes.size());
  auto merged_blocks = MergeBlocks(t, ldims, rdims);
  auto trunc_blk_svd_res = TruncatedBlockSvd(merged_blocks, cutoff, Dmin, Dmax);
  return WrapBlock(trunc_blk_svd_res, ldiv, rdiv, t.indexes, ldims, rdims);
}


PartDivsAndMergedBlk MergeBlocks(
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
    merged_blocks.emplace(kv.first, MergeBlock(kv.second));
  }
  return merged_blocks;
}


MergedBlk MergeBlock(const std::vector<BipartiteBlkData> &blkdatas) {
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
    auto raw_svd_data = MatSvd(kv.second.mat, kv.second.mat_ldim, kv.second.mat_rdim);
    delete[] kv.second.mat;
    if (raw_svd_data.info == 0) {
      auto sdim = std::min(kv.second.mat_ldim, kv.second.mat_rdim);
      svd_data.emplace(
          kv.first,
          BlkSvdData(
              kv.second.lqnscts_set, kv.second.rqnscts_set,
              raw_svd_data.u, raw_svd_data.s, raw_svd_data.v,
              kv.second.mat_ldim, sdim, kv.second.mat_rdim));
      for (long i = 0; i < sdim; i++) {
        auto sv = raw_svd_data.s[i];
        if (sv != 0.0) { singular_values.push_back(sv); }
      }
    } else {
      std::cout << "LAPACKE_dgesdd error." << std::endl;
      exit(1);
    }
  }
  long total_dim = singular_values.size();
  double trunc_err = 0.0;
  long kept_dim = total_dim;
  std::sort(singular_values.begin(), singular_values.end());
  auto sv_squares = SquareVec(singular_values);
  auto normalized_sv_squares = NormVec(sv_squares);
  if (total_dim > dmax) {
    trunc_err = std::accumulate(
                    normalized_sv_squares.begin(),
                    normalized_sv_squares.end()-dmax,
                    0.0);
    normalized_sv_squares = SliceFromEnd(normalized_sv_squares, dmax);
    kept_dim = dmax;
  }
  if (kept_dim > dmin) {
    for (auto &nsvs : normalized_sv_squares) {
      if (trunc_err + nsvs > cutoff) {
        break;
      } else {
        trunc_err += nsvs;
        kept_dim -= 1;
        if (kept_dim == dmin) {
          break;
        }
      }
    }
  }
  auto kept_smallest_sv = *(singular_values.end() + (-kept_dim));
  PartDivsAndBlkSvdData truncated_blocks;
  for (auto &kv : svd_data) {
    double *trunced_u = nullptr;
    double *trunced_s = nullptr;
    double *trunced_v = nullptr;
    long kept_sv_num = 0;
    MatTrans(kv.second.uldim, kv.second.sdim, kv.second.u);
    for (long i = 0; i < kv.second.sdim; ++i) {
      if (kv.second.s[i] >= kept_smallest_sv) {
        MatAppendRow(
            trunced_u,
            kept_sv_num,
            kv.second.uldim,
            MatGetConstRow(kv.second.u, i, kv.second.uldim));
        ArrayAppend(trunced_s, kept_sv_num, kv.second.s[i]);
        MatAppendRow(
            trunced_v,
            kept_sv_num,
            kv.second.vrdim,
            MatGetConstRow(kv.second.v, i, kv.second.vrdim));
        kept_sv_num += 1;
      }
    }
    delete [] kv.second.u; kv.second.u = nullptr;
    delete [] kv.second.s; kv.second.s = nullptr;
    delete [] kv.second.v; kv.second.v = nullptr;
    if (kept_sv_num != 0) {
      MatTrans(kept_sv_num, kv.second.uldim, trunced_u);
      truncated_blocks.emplace(
          kv.first,
          BlkSvdData(
              kv.second.lqnscts_set, kv.second.rqnscts_set,
              trunced_u, trunced_s, trunced_v,
              kv.second.uldim, kept_sv_num, kv.second.vrdim));
    }
  }
  TruncBlkSvdData truncated_blk_svd_data;
  truncated_blk_svd_data.trunc_blks = truncated_blocks;
  truncated_blk_svd_data.trunc_err = trunc_err;
  truncated_blk_svd_data.kept_dim = kept_dim;
  return truncated_blk_svd_data;
}


SvdRes WrapBlock(
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
    // Create u and v block.
    long u_row_offset = 0;
    long v_col_offset = 0;
    auto transed_v = MatTrans(kv.second.v, kv.second.sdim, kv.second.vrdim);
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
void MatMul(
    const double *m1, const long &ldim1, const long &rdim1,
    const double *m2, const long &ldim2, const long &rdim2,
    double *res) {
  assert(rdim1 == ldim2);
  double alpha = 1.0, beta = 0.0;
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      ldim1, rdim2, ldim2,
      alpha,
      m1, ldim2,
      m2, rdim2,
      beta,
      res, rdim2);
}


RawSvdData MatSvd(double *mat, const long &mld, const long &mrd) {
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
  RawSvdData raw_svd_data;
  raw_svd_data.info = info;
  raw_svd_data.u = u;
  raw_svd_data.s = s;
  raw_svd_data.v = vt;
  return raw_svd_data;
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

void MatAppendRow(
    double * &mat, const long &rows, const long &cols, const double *new_row) {
  auto old_size = rows * cols;
  auto new_size = (rows+1) * cols;
  auto new_mat = new double [new_size];
  if (old_size == 0) {
    std::memcpy(new_mat+old_size, new_row, cols*sizeof(double));
    mat = new_mat;
  } else {
    std::memcpy(new_mat, mat, old_size*sizeof(double));
    std::memcpy(new_mat+old_size, new_row, cols*sizeof(double));
    delete [] mat;
    mat = new_mat;
  }
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


void ArrayAppend(double * &v, const long &size, const double &elem) {
  if (size == 0) {
    v = new double [1];
    v[0] = elem;
  } else {
    auto new_v = new double [size + 1];
    std::memcpy(new_v, v, size*sizeof(double));
    delete [] v;
    new_v[size] = elem;
    v = new_v;
  }
}


void  ArrayElemAttach(
    double * to_v, const long &size, const double * from_v) {
  for (long i = 0; i < size; i++) { to_v[i] += from_v[i]; }
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
    case '1':
      transed_axes = saved_axes;
      transed_axes.insert(
          transed_axes.end(),
          ctrct_axes.begin(), ctrct_axes.end());
      if (transed_axes != ordered_axes) { return true; }
      break;
    case '2':
      transed_axes = ctrct_axes;
      transed_axes.insert(
          transed_axes.end(),
          saved_axes.begin(), saved_axes.end());
      if (transed_axes != ordered_axes) { return true; }
      break;
    default:
      std::cout << "position must be '1' or '2', but" << position << std::endl;
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
