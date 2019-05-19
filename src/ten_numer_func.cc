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
#include "hptt.h"

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
  std::vector<QNBlock *> ctrcted_blocks;
  std::vector<double> ctrcted_scalars;
  for (auto &b1 : t1.BlksConstRef()) {
    for (auto &b2 : t2.BlksConstRef()) {
      if (b1->PartHash(t1_ctrct_idxs) == b2->PartHash(t2_ctrct_idxs)) {
        auto pnew_b = ContractBlock(b1, b2, t1_ctrct_idxs, t2_ctrct_idxs);
        if (pnew_b->qnscts == kNullQNSectors) {
          ctrcted_scalars.push_back(pnew_b->DataConstRef()[0]);
          delete pnew_b;
        } else {
          auto has_blk = false;
          for (auto &pctrcted_blk : ctrcted_blocks) {
            if (pnew_b->qnscts == pctrcted_blk->qnscts) {
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
    } 
  }
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


QNBlock *ContractBlock(
    const QNBlock * b1,
    const QNBlock * b2,
    const std::vector<long> &t1_ctrct_idxs,
    const std::vector<long> &t2_ctrct_idxs) {
  auto b1_ctrct_info = BlkCtrctPreparer(*b1, t1_ctrct_idxs, "first");
  auto b2_ctrct_info = BlkCtrctPreparer(*b2, t2_ctrct_idxs, "second");
  auto ctrcted_data = MatMul(
                          b1_ctrct_info.data,
                          b1_ctrct_info.savedim,
                          b1_ctrct_info.ctrctdim,
                          b2_ctrct_info.data,
                          b2_ctrct_info.ctrctdim,
                          b2_ctrct_info.savedim);
  auto saved_qnscts = b1_ctrct_info.saved_qnscts;
  saved_qnscts.insert(
      saved_qnscts.end(),
      b2_ctrct_info.saved_qnscts.begin(), b2_ctrct_info.saved_qnscts.end());
  auto new_blk = new QNBlock(saved_qnscts);
  new_blk->DataRef() = ctrcted_data;
  return new_blk;
}


BlkCtrctInfo BlkCtrctPreparer(
    const QNBlock &b,
    const std::vector<long> &ctrct_idxs,
    const std::string &which) {
  std::vector<QNSector> saved_qnscts;
  std::vector<long> saved_idxs;
  long savedim = 1;
  long ctrctdim = 1;
  for (long i = 0; i < b.ndim; ++i) {
    if (std::find(ctrct_idxs.begin(), ctrct_idxs.end(), i) == ctrct_idxs.end()) {
      saved_idxs.push_back(i);
      saved_qnscts.push_back(b.qnscts[i]);
      savedim *= b.qnscts[i].dim;
    } else {
      ctrctdim *= b.qnscts[i].dim;
    }
  }
  std::vector<long> new_idxs;
  if (which == "first") {
    new_idxs = saved_idxs;
    new_idxs.insert(new_idxs.end(), ctrct_idxs.begin(), ctrct_idxs.end());
  } else if (which == "second") {
    new_idxs = ctrct_idxs;
    new_idxs.insert(new_idxs.end(), saved_idxs.begin(), saved_idxs.end());
  }
  auto sorted_new_idxs = new_idxs;
  std::sort(sorted_new_idxs.begin(), sorted_new_idxs.end());
  const double *data;
  if (new_idxs != sorted_new_idxs) {
    data = TransposeData(
               b.DataConstRef(),
               b.ndim,
               b.size,
               b.shape,
               new_idxs);
  } else {
    data = b.DataConstRef();
  }
  return BlkCtrctInfo(data, savedim, ctrctdim, saved_qnscts);
}


double *TransposeData(
    const double *old_data,
    const long &old_ndim,
    const long &old_size,
    const std::vector<long> &old_shape,
    const std::vector<long> &transed_axes) {
  int dim = old_ndim;
  int perm[dim];  for (int i = 0; i < dim; ++i) { perm[i] = transed_axes[i]; }
  int sizeA[dim]; for (int i = 0; i < dim; ++i) { sizeA[i] = old_shape[i]; }
  int outerSizeB[dim];
  for (int i = 0; i < dim; ++i) { outerSizeB[i] = old_shape[perm[i]]; }
  auto transed_data = new double[old_size];
  dTensorTranspose(perm, dim,
      1.0, old_data, sizeA, sizeA,
      0.0, transed_data, outerSizeB,
      20, 1);
  return transed_data;
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
    auto blk_data = BipartiteBlkData(lqnscts, rqnscts, blk->DataRef());
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
    for (long i = 0; i < kv.second.sdim; ++i) {
      auto ut = MatTrans(kv.second.u, kv.second.uldim, kv.second.sdim);
      if (kv.second.s[i] >= kept_smallest_sv) {
        MatAppendRow(
            trunced_u,
            kept_sv_num,
            kv.second.uldim,
            MatGetConstRow(ut, i, kv.second.uldim));
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
      trunced_u = MatTrans(trunced_u, kept_sv_num, kv.second.uldim);
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
    sblock->DataRef() = GenDiagMat(kv.second.s, kv.second.sdim);
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
      ublock->DataRef() = MatGetRows(
                              kv.second.u, kv.second.uldim, kv.second.sdim,
                              u_row_offset, u_row_dim);
      ublocks.push_back(ublock);
      u_row_offset += u_row_dim;
    }
    delete [] kv.second.u; kv.second.u = nullptr;
    for (auto &rqnscts : kv.second.rqnscts_set) {
      auto v_col_dim = MulDims(rqnscts);
      auto vblock_qnscts = rqnscts;
      vblock_qnscts.insert(vblock_qnscts.begin(), sblk_qnsct);
      auto vblock = new QNBlock(vblock_qnscts);
      vblock->DataRef() = MatTrans(   /* TODO: two copy here, no efficiency.*/
                              MatGetRows(
                                  transed_v, kv.second.uldim, kv.second.sdim,
                                  v_col_offset, v_col_dim),
                              v_col_dim, kv.second.sdim);
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
double *MatMul(
    const double *m1, const long &ldim1, const long &rdim1,
    const double *m2, const long &ldim2, const long &rdim2) {
  assert(rdim1 == ldim2);
  auto res = new double [ldim1*rdim2];
  double alpha = 1.0, beta = 0.0;
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      ldim1, rdim2, ldim2,
      alpha,
      m1, ldim2,
      m2, rdim2,
      beta,
      res, rdim2);
  return res;
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


double *GenDiagMat(const double *diag_v, const long &diag_v_dim) {
  auto full_mat = new double [diag_v_dim*diag_v_dim] ();
  for (long i = 0; i < diag_v_dim; ++i) {
    *(full_mat + (i*diag_v_dim + i)) = diag_v[i];
  }
  return full_mat;
}


double *MatGetRows(
    const double *mat, const long &rows, const long &cols,
    const long &from, const long &num_rows) {
  auto new_size = num_rows*cols;
  auto new_mat = new double [new_size]; 
  std::memcpy(new_mat, mat+(from*cols), new_size*sizeof(double));
  return new_mat;
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
} /* gqten */ 
