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
    const GQTensor &t1, const GQTensor &t2,
    const std::vector<std::vector<long>> &axes_set) {
  auto t1_ctrct_idxs = axes_set[0];
  auto t2_ctrct_idxs = axes_set[1];

  // Block contraction data prepare.
  std::vector<long> t1_transed_axes, t2_transed_axes;
  bool t1_need_trans = CtrctTransCheck(t1_ctrct_idxs, t1, '1', t1_transed_axes);
  bool t2_need_trans = CtrctTransCheck(t2_ctrct_idxs, t2, '2', t2_transed_axes);
  auto t1_blks_part_hash_table = GenPartHashTable(t1, t1_ctrct_idxs);
  auto t2_blks_part_hash_table = GenPartHashTable(t2, t2_ctrct_idxs);
  auto t1_blks_num = t1.BlksConstRef().size();
  auto t2_blks_num = t2.BlksConstRef().size();
  std::vector<QNBlock *> t1_to_ctrct_blks(t1_blks_num);
  std::vector<QNBlock *> t2_to_ctrct_blks(t2_blks_num);
  std::vector<std::pair<std::size_t, std::size_t>> ctrct_blk_pair_idxs;
  for (std::size_t i = 0; i < t1_blks_num; ++i) {
    for (std::size_t j = 0; j < t2_blks_num; ++j) {
      if (t1_blks_part_hash_table[i] == t2_blks_part_hash_table[j]) {
        ctrct_blk_pair_idxs.push_back(std::make_pair(i, j));
        if (t1_to_ctrct_blks[i] == nullptr) {
          if (t1_need_trans) {
            auto blk_transed_to_ctrct = new QNBlock(*t1.BlksConstRef()[i]);
            blk_transed_to_ctrct->Transpose(t1_transed_axes);
            t1_to_ctrct_blks[i] = blk_transed_to_ctrct;
          } else {
            t1_to_ctrct_blks[i] = t1.BlksConstRef()[i]; 
          }
        }
        if (t2_to_ctrct_blks[j] == nullptr) {
          if (t2_need_trans) {
            auto blk_transed_to_ctrct = new QNBlock(*t2.BlksConstRef()[j]);
            blk_transed_to_ctrct->Transpose(t2_transed_axes);
            t2_to_ctrct_blks[j] = blk_transed_to_ctrct;
          } else {
            t2_to_ctrct_blks[j] = t2.BlksConstRef()[j];
          }
        }
      }
    }
  }

  // Contract blocks.
  std::vector<QNBlock *> ctrcted_blocks;
  std::vector<double> ctrcted_scalars;
  auto t1_ctrct_ndim = t1_ctrct_idxs.size();
  auto t2_ctrct_ndim = t2_ctrct_idxs.size();
  for (auto &ctrct_blk_pair : ctrct_blk_pair_idxs) {
    auto pnew_b = ContractBlockNoTrans(
                      *t1_to_ctrct_blks[ctrct_blk_pair.first],
                      *t2_to_ctrct_blks[ctrct_blk_pair.second],
                      t1_ctrct_ndim, t2_ctrct_ndim);
    if (pnew_b->QNSectorSetHash() == 0) {
      ctrcted_scalars.push_back(pnew_b->DataConstRef()[0]);
      delete pnew_b;
    } else {
      auto has_blk = false;
      for (auto &pctrcted_blk : ctrcted_blocks) {
        if (pnew_b->QNSectorSetHash() == pctrcted_blk->QNSectorSetHash()) {
          auto data_size = pnew_b->size;
          assert(data_size == pctrcted_blk->size);
          ArrayElemAttach(
              pctrcted_blk->DataRef(), data_size,
              pnew_b->DataConstRef());
          delete pnew_b;
          has_blk = true;
          break;
        }
      }
      if (!has_blk) {
        ctrcted_blocks.push_back(pnew_b);
      }
    }
  }
  if (t1_need_trans) {
    for (auto &to_ctrct_blk : t1_to_ctrct_blks) { delete to_ctrct_blk; }
  }
  if (t2_need_trans) {
    for (auto &to_ctrct_blk : t2_to_ctrct_blks) { delete to_ctrct_blk; }
  }

  // Wrap results.
  auto pctrcted_ten = InitCtrctedTen(t1, t2, t1_ctrct_idxs, t2_ctrct_idxs);
  if (pctrcted_ten->indexes == kNullIndexes) {
    pctrcted_ten->scalar = VecSumOver(ctrcted_scalars);
  } else {
    pctrcted_ten->BlksRef() = ctrcted_blocks;
  }
  return pctrcted_ten;
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


QNBlock *ContractBlockNoTrans(
    const QNBlock &b1, const QNBlock &b2,
    const std::size_t t1_ctrct_ndim, const std::size_t t2_ctrct_ndim) {
  std::size_t b1_saved_size = 1;
  std::size_t b1_ctrct_size = 1;
  auto b1_saved_ndim = b1.ndim - t1_ctrct_ndim;
  for (long i = 0; i < b1.ndim; i++) {
    if (i < b1_saved_ndim) {
      b1_saved_size *= b1.qnscts[i].dim;
    } else {
      b1_ctrct_size *= b1.qnscts[i].dim;
    }
  }
  std::size_t b2_saved_size = 1;
  std::size_t b2_ctrct_size = 1;
  for (long i = 0; i < b2.ndim; ++i) {
    if (i < t2_ctrct_ndim) {
      b2_ctrct_size *= b2.qnscts[i].dim;
    } else {
      b2_saved_size *= b2.qnscts[i].dim;
    }
  }
  std::vector<QNSector> saved_qnscts(
      b1.qnscts.cbegin(), b1.qnscts.cend()-t1_ctrct_ndim);
  saved_qnscts.insert(saved_qnscts.end(),
      b2.qnscts.cbegin()+t2_ctrct_ndim, b2.qnscts.cend());
  auto new_blk = new QNBlock(saved_qnscts);
  if (saved_qnscts.size() == 0) { new_blk->DataRef() = new double[1]; }
  MatMul(
      b1.DataConstRef(),
      b1_saved_size, b1_ctrct_size,
      b2.DataConstRef(),
      b2_ctrct_size, b2_saved_size,
      new_blk->DataRef());
  return new_blk;
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
  //auto res = new double [ldim1*rdim2];
  double alpha = 1.0, beta = 0.0;
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      ldim1, rdim2, ldim2,
      alpha,
      m1, ldim2,
      m2, rdim2,
      beta,
      res, rdim2);
  //return res;
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


double VecSumOver(const std::vector<double> &v) {
  double sum = 0.0;
  for (auto &elem : v) { sum += elem; }
  return sum;
}


// Tensor contraction helpers.
bool CtrctTransCheck(
    const std::vector<long> &ctrct_axes,
    const GQTensor &t,
    char position,
    std::vector<long> &transed_axes) {
  auto ndim = t.indexes.size();
  auto ctrct_ndim = ctrct_axes.size();
  std::vector<long> saved_axes(ndim-ctrct_ndim);
  std::size_t saved_axes_idx = 0;
  std::vector<long> ordered_axes(ndim);
  for (std::size_t i = 0; i < ndim; ++i) {
    if (std::find(ctrct_axes.begin(), ctrct_axes.end(), i) ==
        ctrct_axes.end()) {
      saved_axes[saved_axes_idx] = i;
      saved_axes_idx++;
    }
    ordered_axes[i] = 0;
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


std::vector<std::size_t>GenPartHashTable(
    const GQTensor &t, const std::vector<long>ctrct_axes) {
  std::vector<std::size_t> part_hash_table(t.BlksConstRef().size());
  for (std::size_t i = 0; i < t.BlksConstRef().size(); i++) {
    part_hash_table[i] = t.BlksConstRef()[i]->PartHash(ctrct_axes);
  }
  return part_hash_table;
}
} /* gqten */ 
