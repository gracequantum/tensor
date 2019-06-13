// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-05 14:11
* 
* Description: GraceQ/tensor project. Numerical functions for GQTensor, head file.
*/
#ifndef GQTEN_TEN_NUMER_FUNC_H
#define GQTEN_TEN_NUMER_FUNC_H


#include "gqten/gqten.h"

#include <vector>
#include <unordered_map>
#include <numeric>
#include <cstring>
#include <cmath>


#include "mkl.h"


namespace gqten {


// Tensor contraction.
GQTensor *InitCtrctedTen(
    const GQTensor &, const GQTensor &,
    const std::vector<long> &, const std::vector<long> &);

std::vector<QNBlock *> BlksCtrctBatch(
    const std::vector<long> &, const std::vector<long> &,
    const double,
    const std::vector<QNBlock *> &, const std::vector<QNBlock *> &);

void WrapCtrctBlks(std::vector<QNBlock *> &, GQTensor *);

std::vector<QNBlock *> MergeCtrctBlks(const std::vector<QNBlock *> &);

inline void GemmBatch(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE* transa_array, const CBLAS_TRANSPOSE* transb_array,
    const MKL_INT* m_array, const MKL_INT* n_array, const MKL_INT* k_array,
    const double* alpha_array,
    const double **a_array, const MKL_INT* lda_array,
    const double **b_array, const MKL_INT* ldb_array,
    const double* beta_array,
    double **c_array, const MKL_INT* ldc_array,
    const MKL_INT group_count,
    const MKL_INT* group_size) {
  cblas_dgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      alpha_array,
      a_array, lda_array,
      b_array, ldb_array,
      beta_array,
      c_array, ldc_array,
      group_count,
      group_size);
}

void CalcCtrctBlkDimInfo(
    const std::size_t, const QNBlock *, const std::vector<long> &,
    long *, long *);

std::vector<const QNSector *> GetPNewBlkQNScts(
    const QNBlock *, const QNBlock *,
    const std::vector<long> &, const std::vector<long> &);

inline void FreeBlks(std::vector<QNBlock *> &blks) {
  for (auto &blk : blks) { delete blk; }
}

// Tensor linear combination.
void LinearCombineOneTerm(const double, const GQTensor *, GQTensor *);

// Tensor SVD.
struct BipartiteBlkData {
  BipartiteBlkData(
      const std::vector<QNSector> &lqnscts,
      const std::vector<QNSector> &rqnscts,
      const double *data) :
      lqnscts(lqnscts), rqnscts(rqnscts), data(data) {}
  BipartiteBlkData(const BipartiteBlkData &blk_data) :
      lqnscts(blk_data.lqnscts), rqnscts(blk_data.rqnscts), data(blk_data.data) {}
  BipartiteBlkData operator=(const BipartiteBlkData &blk_data) {
    return BipartiteBlkData(blk_data); 
  }
  std::vector<QNSector> lqnscts;
  std::vector<QNSector> rqnscts;
  const double *data;
};

typedef std::vector<std::vector<QNSector>> QNSectorsSet;

struct MergedBlk {
  MergedBlk(
      const QNSectorsSet &lqnscts_set,
      const QNSectorsSet &rqnscts_set,
      double *mat,
      const long &mat_ldim,
      const long &mat_rdim) :
      lqnscts_set(lqnscts_set), rqnscts_set(rqnscts_set),
      mat(mat), mat_ldim(mat_ldim), mat_rdim(mat_rdim) {}
  MergedBlk(const MergedBlk &mb) :
      lqnscts_set(mb.lqnscts_set), rqnscts_set(mb.rqnscts_set),
      mat(mb.mat), mat_ldim(mb.mat_ldim), mat_rdim(mb.mat_rdim) {}
  MergedBlk operator=(const MergedBlk &mb) { return MergedBlk(mb); }
  QNSectorsSet lqnscts_set;
  QNSectorsSet rqnscts_set;
  double *mat;
  long mat_ldim;
  long mat_rdim;
};

typedef std::pair<QN, QN> PartDivs;

struct PartDivsHash {
  size_t operator()(const PartDivs& part_divs) const {
    return part_divs.first.Hash() ^ (part_divs.second.Hash() << 1); 
  }
};

struct PartDivsEqual {
  bool operator()(const PartDivs &lhs, const PartDivs &rhs) const {
    return lhs.first.Hash() == rhs.first.Hash() &&
           lhs.second.Hash() == rhs.second.Hash(); 
  }
};

typedef 
    std::unordered_map<PartDivs, MergedBlk, PartDivsHash, PartDivsEqual>
    PartDivsAndMergedBlk;

typedef
    std::unordered_map<PartDivs, std::vector<BipartiteBlkData>,
                       PartDivsHash, PartDivsEqual>
    PartDivsAndBipartiteBlkDatas;

PartDivsAndMergedBlk SvdMergeBlocks(
    const GQTensor &, const long &, const long &); 

MergedBlk SvdMergeBlk(const std::vector<BipartiteBlkData> &);

// For SVD block data process.
struct RawSvdRes {
  int info;
  double *u;
  double *s;
  double *v;
};

struct BlkSvdData {
  BlkSvdData(
      const QNSectorsSet &lqnscts_set,
      const QNSectorsSet &rqnscts_set,
      double * &u,
      double * &s,
      double * &v,
      const long &uldim,
      const long &sdim,
      const long &vrdim) :
      lqnscts_set(lqnscts_set), rqnscts_set(rqnscts_set),
      u(u), s(s), v(v), uldim(uldim), sdim(sdim), vrdim(vrdim) {}
  BlkSvdData(const BlkSvdData &bsd) :
      lqnscts_set(bsd.lqnscts_set), rqnscts_set(bsd.rqnscts_set),
      u(bsd.u), s(bsd.s), v(bsd.v),
      uldim(bsd.uldim), sdim(bsd.sdim), vrdim(bsd.vrdim) {}
  BlkSvdData operator=(const BlkSvdData &blk_svd_data) {
    return BlkSvdData(blk_svd_data);
  }

  QNSectorsSet lqnscts_set;
  QNSectorsSet rqnscts_set;
  double *u;
  double *s;
  double *v;
  long uldim;
  long sdim;
  long vrdim;
};

typedef
    std::unordered_map<PartDivs, BlkSvdData, PartDivsHash, PartDivsEqual>
    PartDivsAndBlkSvdData;

struct TruncBlkSvdData {
  PartDivsAndBlkSvdData trunc_blks;
  double trunc_err;
  long kept_dim;
};

TruncBlkSvdData TruncatedBlockSvd(
    const PartDivsAndMergedBlk &,
    const double &,
    const long &,
    const long &);

void SvdTruncteBlks(
    PartDivsAndBlkSvdData &, const double,
    PartDivsAndBlkSvdData &);

// For block wrap.
SvdRes SvdWrapBlocks(
    TruncBlkSvdData &,
    const QN &, const QN &,
    const std::vector<Index> &,
    const long &, const long &);


// Operations for matrix.
RawSvdRes MatSvd(double *, const long &, const long &);

double *MatTrans(const double *, const long &, const long &); // off-place

void MatTrans(const long &, const long &, double *);          // in-place

void GenDiagMat(const double *, const long &, double *);

double *MatGetRows(       // off-place
    const double *, const long &, const long &, const long &, const long &);

void MatGetRows(          // in-place
    const double *, const long &, const long &, const long &, const long &,
    double *);


// Helpers.
template<typename T>
std::vector<T> SliceFromBegin(const std::vector<T> &v, size_t to) {
  auto first = v.cbegin();
  return std::vector<T>(first, first+to);
}


template<typename T>
std::vector<T> SliceFromEnd(const std::vector<T> &v, size_t to) {
  auto last = v.cend();
  return std::vector<T>(last-to, last);
}


inline long MulDims(const std::vector<QNSector> &qnscts) {
  if (qnscts.size() == 0) { return 0; }
  long res = 1;
  for (auto &qnsct : qnscts) { res *= qnsct.dim; }
  return res;
}


inline long OffsetInQNSectorsSet(
    const std::vector<QNSector> &qnscts, const QNSectorsSet &qnscts_set) {
  long offset = 0;
  for (auto &qnss : qnscts_set) {
    if (qnss == qnscts) {
      break;
    } else {
      offset += MulDims(qnss);
    }
  }
  return offset;
}


inline void CpySubMat(
    double *mat, const long &mat_ldim, const long &mat_rdim,
    const double *sub, const long &sub_ldim, const long &sub_rdim,
    const long &linter_offset, const long &rinter_offset) {
  for (long i = 0; i < sub_ldim; ++i) {
    long inter_offset = (i + linter_offset)*mat_rdim + rinter_offset;
    long intra_offset = i*sub_rdim;
    std::memcpy(mat+inter_offset, sub+intra_offset, sub_rdim*sizeof(double));
  }
}


inline std::vector<double> SquareVec(const std::vector<double> &v) {
  std::vector<double> res(v.size());
  for (size_t i = 0; i < v.size(); ++i) { res[i] = std::pow(v[i], 2.0); }
  return res;
}


inline std::vector<double> NormVec(const std::vector<double> &v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  std::vector<double> res(v.size());
  for (size_t i = 0; i < v.size(); ++i) { res[i] = v[i] / sum; }
  return res;
}

// Tensor contraction helpers.
bool CtrctTransCheck(
    const std::vector<long> &, const long, const char, std::vector<long> &);

std::vector<std::size_t> GenBlksPartHashTable(
    const std::vector<QNBlock *> &, const std::vector<long> &);
} /* gqten */ 
#endif /* ifndef GQTEN_TEN_NUMER_FUNC_H */
