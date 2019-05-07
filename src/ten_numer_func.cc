/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-05 14:10
* 
* Description: GraceQ/tensor project. Numerical functions for GQTensor, src file.
*/
#include "ten_numer_func.h"
#include "gqten/gqten.h"

#include <vector>
#include <algorithm>

#include "Accelerate/Accelerate.h"

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>

namespace gqten {


GQTensor Contract(
    const GQTensor &t1, const GQTensor &t2,
    const std::vector<std::vector<long>> &axes_set) {
  auto t1_ctrct_idxs = axes_set[0];
  auto t2_ctrct_idxs = axes_set[1];
  std::vector<QNBlock *> ctrcted_blocks;
  std::vector<double>  ctrcted_scalars;
  for (auto &b1 : t1.BlksConstRef()) {
    for (auto &b2 : t2.BlksConstRef()) {
      if (b1->PartHash(t1_ctrct_idxs) == b2->PartHash(t2_ctrct_idxs)) {
        auto new_b = ContractBlock(b1, b2, t1_ctrct_idxs, t2_ctrct_idxs);
        if (new_b->qnscts == QNSectorSet()) {
          ctrcted_scalars.push_back(new_b->DataConstRef()[0]);
          /* TODO: Possible memory leak for new_b ??? */
        } else {
          auto has_blk = false;
          for (auto &ctrcted_blk : ctrcted_blocks) {
            if (new_b->qnscts == ctrcted_blk->qnscts) {
              assert(new_b->size == ctrcted_blk->size);
              auto ctrcted_blk_data = ctrcted_blk->DataRef();
              auto new_b_data = new_b->DataConstRef();
              for (long i = 0; i < ctrcted_blk->size; ++i) {
                ctrcted_blk_data[i] += new_b_data[i];
              }
              has_blk = true;
              break;
            }
          }
          if (!has_blk) {
            ctrcted_blocks.push_back(new_b);
          }
        }
      }
    }
  }
  auto ctrcted_ten  = InitCtrctedTen(t1, t2, t1_ctrct_idxs, t2_ctrct_idxs);
  if (ctrcted_ten.indexes == std::vector<Index>()) {
    for (auto &s : ctrcted_scalars) {
      ctrcted_ten.scalar += s;
    }
  } else {
    ctrcted_ten.BlksRef() = ctrcted_blocks;
  }
  return ctrcted_ten;
}


GQTensor InitCtrctedTen(
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
  return GQTensor(saved_idxs);
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
  std::vector<long> transed_shape(old_ndim);
  for (long i = 0; i < old_ndim; ++i) {
    transed_shape[i] = old_shape[transed_axes[i]];
  }
  auto old_data_offsets = CalcDataOffsets(old_shape);
  auto transed_data_offsets = CalcDataOffsets(transed_shape);
  auto transed_data = new double [old_size];
  for (auto &old_coors : GenAllCoors(old_shape)) {
    transed_data[CalcOffset(TransCoors(old_coors, transed_axes), old_ndim, transed_data_offsets)] = 
        old_data[CalcOffset(old_coors, old_ndim, old_data_offsets)];
  }
  return transed_data;
}


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
} /* gqten */ 
