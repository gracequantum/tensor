// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-20 15:21
* 
* Description: GraceQ/tensor project. Implementation details for tensor SVD.
*/
#include <unordered_map>
#include <numeric>

#include "gqten/gqten.h"
#include "gqten/detail/utils_inl.h"
#include "gqten/detail/ten_linalg_wrapper.h"


namespace gqten {


// Forward declarations.

typedef std::pair<QN, QN> PartDivs;

typedef std::vector<std::vector<QNSector>> QNSectorsSet;

template <typename TenElemType>
struct MergedBlk {
  MergedBlk(
      const QNSectorsSet &lqnscts_set,
      const QNSectorsSet &rqnscts_set,
      TenElemType *mat,
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
  TenElemType *mat;
  long mat_ldim;
  long mat_rdim;
};

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

template <typename TenElemType>
using PartDivsAndMergedBlk =
    std::unordered_map<PartDivs, MergedBlk<TenElemType>,
                       PartDivsHash, PartDivsEqual>;

template <typename TenElemType>
struct BipartiteBlkData {
  BipartiteBlkData(
      const std::vector<QNSector> &lqnscts,
      const std::vector<QNSector> &rqnscts,
      const TenElemType *data) :
      lqnscts(lqnscts), rqnscts(rqnscts), data(data) {}
  BipartiteBlkData(const BipartiteBlkData &blk_data) :
      lqnscts(blk_data.lqnscts),
      rqnscts(blk_data.rqnscts),
      data(blk_data.data) {}
  BipartiteBlkData operator=(const BipartiteBlkData &blk_data) {
    return BipartiteBlkData(blk_data); 
  }
  std::vector<QNSector> lqnscts;
  std::vector<QNSector> rqnscts;
  const TenElemType *data;
};

template <typename TenElemType>
using PartDivsAndBipartiteBlkDatas =
    std::unordered_map<PartDivs, std::vector<BipartiteBlkData<TenElemType>>,
                       PartDivsHash, PartDivsEqual>;

template <typename TenElemType>
PartDivsAndMergedBlk<TenElemType> SvdMergeBlocks(
    const GQTensor<TenElemType> &, const long &, const long &);

template <typename TenElemType>
MergedBlk<TenElemType> SvdMergeBlk(
    const std::vector<BipartiteBlkData<TenElemType>> &);


template <typename TenElemType>
struct BlkSvdData {
  BlkSvdData(
      const QNSectorsSet &lqnscts_set,
      const QNSectorsSet &rqnscts_set,
      TenElemType * &u,
      TenElemType * &s,
      TenElemType * &v,
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
  TenElemType *u;
  TenElemType *s;
  TenElemType *v;
  long uldim;
  long sdim;
  long vrdim;
};

template <typename TenElemType>
using PartDivsAndBlkSvdData =
    std::unordered_map<PartDivs, BlkSvdData<TenElemType>,
                       PartDivsHash, PartDivsEqual>;

template <typename TenElemType>
struct TruncBlkSvdData {
  PartDivsAndBlkSvdData<TenElemType> trunc_blks;
  double trunc_err;
  long kept_dim;
};

template <typename TenElemType>
TruncBlkSvdData<TenElemType> TruncatedBlockSvd(
    const PartDivsAndMergedBlk<TenElemType> &,
    const double &,
    const long &,
    const long &);

template <typename TenElemType>
void SvdTruncteBlks(
    PartDivsAndBlkSvdData<TenElemType> &, const double,
    PartDivsAndBlkSvdData<TenElemType> &);

template <typename TenElemType>
void SvdWrapBlocks(
    TruncBlkSvdData<TenElemType> &,
    const QN &, const QN &,
    const std::vector<Index> &,
    const long &, const long &,
    GQTensor<TenElemType> *,
    GQTensor<TenElemType> *,
    GQTensor<TenElemType> *,
    double *, long *);


long MulDims(const std::vector<QNSector> &);

long OffsetInQNSectorsSet(const std::vector<QNSector> &, const QNSectorsSet &);

void CpySubMat(
    double *, const long &, const long &,
    const double *, const long &, const long &,
    const long &, const long &);


// Tensor SVD.
template <typename TenElemType>
void Svd(
    const GQTensor<TenElemType> *pt,
    const long ldims, const long rdims,
    const QN &ldiv, const QN &rdiv,
    const double cutoff, const long Dmin, const long Dmax,
    GQTensor<TenElemType> *pu,
    GQTensor<TenElemType> *ps,
    GQTensor<TenElemType> *pvt,
    double *ptrunc_err, long *pD) {
  assert((ldims + rdims) == pt->indexes.size());

#ifdef GQTEN_TIMING_MODE
  Timer svd_merge_blks_timer("svd_merge_blks");
  svd_merge_blks_timer.Restart();
#endif

  auto merged_blocks = SvdMergeBlocks(*pt, ldims, rdims);

#ifdef GQTEN_TIMING_MODE
  svd_merge_blks_timer.PrintElapsed();
#endif

#ifdef GQTEN_TIMING_MODE
  Timer svd_blks_svd_timer("svd_blks_svd");
  svd_blks_svd_timer.Restart();
#endif

  auto trunc_blk_svd_res = TruncatedBlockSvd(merged_blocks, cutoff, Dmin, Dmax);

#ifdef GQTEN_TIMING_MODE
  svd_blks_svd_timer.PrintElapsed();
#endif
  
#ifdef GQTEN_TIMING_MODE
  Timer svd_wrap_blks("svd_wrap_blks");
  svd_wrap_blks.Restart();
#endif

  SvdWrapBlocks(
      trunc_blk_svd_res,
      ldiv, rdiv, pt->indexes, ldims, rdims,
      pu, ps, pvt, ptrunc_err, pD);

#ifdef GQTEN_TIMING_MODE
  svd_wrap_blks.PrintElapsed();
#endif
}


template <typename TenElemType>
PartDivsAndMergedBlk<TenElemType> SvdMergeBlocks(
    const GQTensor<TenElemType> &t, const long &ldims, const long &rdims) {
  PartDivsAndBipartiteBlkDatas<TenElemType> tomerge_blkdatas;
  PartDivsEqual partdivs_equaler;
  for (auto &blk : t.cblocks()) {
    auto lqnscts = SliceFromBegin(blk->qnscts, ldims);
    auto rqnscts = SliceFromEnd(blk->qnscts, rdims);
    auto lpartdiv = CalcDiv(lqnscts, SliceFromBegin(t.indexes, ldims));
    auto rpartdiv = CalcDiv(rqnscts, SliceFromEnd(t.indexes, rdims));
    auto partdivs = std::make_pair(lpartdiv, rpartdiv);
    auto blk_data = BipartiteBlkData<TenElemType>(lqnscts, rqnscts, blk->cdata());
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
  PartDivsAndMergedBlk<TenElemType> merged_blocks;
  for (auto &kv : tomerge_blkdatas) {
    merged_blocks.emplace(kv.first, SvdMergeBlk(kv.second));
  }
  return merged_blocks;
}


template <typename TenElemType>
MergedBlk<TenElemType> SvdMergeBlk(
    const std::vector<BipartiteBlkData<TenElemType>> &blkdatas) {
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
  return MergedBlk<TenElemType>(
      lqnscts_set, rqnscts_set, mat, extra_ldim, extra_rdim);
}


template <typename TenElemType>
TruncBlkSvdData<TenElemType> TruncatedBlockSvd(
    const PartDivsAndMergedBlk<TenElemType> &merged_blocks,
    const double &cutoff,
    const long &dmin,
    const long &dmax) {
  PartDivsAndBlkSvdData<TenElemType> svd_data;
  std::vector<double> singular_values;

#ifdef GQTEN_TIMING_MODE
  Timer svd_gesdd_timer("svd_gesdd");
  svd_gesdd_timer.Restart();
#endif

  for (auto &kv : merged_blocks) {
    auto raw_svd_res = MatSvd(
                           kv.second.mat,
                           kv.second.mat_ldim, kv.second.mat_rdim);
    delete[] kv.second.mat;
    if (raw_svd_res.info == 0) {
      auto sdim = std::min(kv.second.mat_ldim, kv.second.mat_rdim);
      svd_data.emplace(
          kv.first,
          BlkSvdData<TenElemType>(
              kv.second.lqnscts_set, kv.second.rqnscts_set,
              raw_svd_res.u, raw_svd_res.s, raw_svd_res.v,
              kv.second.mat_ldim, sdim, kv.second.mat_rdim));
      for (long i = 0; i < sdim; i++) {
        auto sv = raw_svd_res.s[i];
        if (sv != 0.0) {
          singular_values.push_back(sv);
        }
      }
    } else {
      std::cout << "LAPACKE_dgesdd error." << std::endl;
      exit(1);
    }
  }

#ifdef GQTEN_TIMING_MODE
  svd_gesdd_timer.PrintElapsed();
#endif

  long total_dim = singular_values.size();
  assert(total_dim > 0);
  if (total_dim <= dmin) {
    TruncBlkSvdData<TenElemType> truncated_blk_svd_data;
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
    if (next_kept_dim > total_dim) { break; }
    next_trunc_err = trunc_err -
                     normalized_sv_squares[total_dim-(kept_dim+1)];
    if (DoubleEq(next_trunc_err, 0)) { next_trunc_err = 0; }
    if (next_kept_dim > dmax || next_trunc_err < cutoff) {
      break;
    } else {
      kept_dim = next_kept_dim;
      trunc_err = next_trunc_err;
    }
  }
  assert(kept_dim <= total_dim);
  auto kept_smallest_sv = singular_values[total_dim-kept_dim];
  PartDivsAndBlkSvdData<TenElemType> truncated_blocks;
  SvdTruncteBlks(svd_data, kept_smallest_sv, truncated_blocks);
  TruncBlkSvdData<TenElemType> truncated_blk_svd_data;
  truncated_blk_svd_data.trunc_blks = truncated_blocks;
  truncated_blk_svd_data.trunc_err = trunc_err;
  truncated_blk_svd_data.kept_dim = kept_dim;
  return truncated_blk_svd_data;
}


template <typename TenElemType>
void SvdTruncteBlks(
    PartDivsAndBlkSvdData<TenElemType> &svd_data,
    const double kept_smallest_sv,
    PartDivsAndBlkSvdData<TenElemType> &truncated_svd_data) {
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
      trunced_u = MatGetCols(
                        kv.second.u, kv.second.uldim, kv.second.sdim,
                        0, blk_kept_dim);
      trunced_v = MatGetRows(
                      kv.second.v, kv.second.sdim, kv.second.vrdim,
                      0, blk_kept_dim);
      truncated_svd_data.emplace(
          kv.first,
          BlkSvdData<TenElemType>(
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
          BlkSvdData<TenElemType>(
              kv.second.lqnscts_set, kv.second.rqnscts_set,
              trunced_u, trunced_s, trunced_v,
              kv.second.uldim, blk_kept_dim, kv.second.vrdim));
    }
  }
}


template <typename TenElemType>
void SvdWrapBlocks(
    TruncBlkSvdData<TenElemType> &truncated_blk_svd_data,
    const QN &ldiv, const QN &rdiv,
    const std::vector<Index> &indexes,
    const long &ldims, const long &rdims,
    GQTensor<TenElemType> *pu,
    GQTensor<TenElemType> *ps,
    GQTensor<TenElemType> *pvt,
    double *ptrunc_err, long *pD) {
  std::vector<QNBlock<TenElemType> *> ublocks, sblocks, vblocks;
  std::vector<QNSector> sblk_qnscts;
  for (auto &kv : truncated_blk_svd_data.trunc_blks) {
    // Create s block.
    auto sblk_qn = ldiv - kv.first.first;
    auto sblk_qnsct = QNSector(sblk_qn, kv.second.sdim);
    sblk_qnscts.push_back(sblk_qnsct);
    auto sblock = new QNBlock<TenElemType>({sblk_qnsct, sblk_qnsct});
    GenDiagMat(kv.second.s, kv.second.sdim, sblock->data());
    delete [] kv.second.s; kv.second.s = nullptr;
    sblocks.push_back(sblock);
    // Create u block.
    long u_row_offset = 0;
    for (auto &lqnscts : kv.second.lqnscts_set) {
      auto u_row_dim = MulDims(lqnscts);
      auto ublock_qnscts = lqnscts; ublock_qnscts.push_back(sblk_qnsct);
      auto ublock = new QNBlock<TenElemType>(ublock_qnscts);
      MatGetRows(
          kv.second.u, kv.second.uldim, kv.second.sdim,
          u_row_offset, u_row_dim,
          ublock->data());
      ublocks.push_back(ublock);
      u_row_offset += u_row_dim;
    }
    delete [] kv.second.u; kv.second.u = nullptr;
    // Create v block.
    long v_col_offset = 0;
    for (auto &rqnscts : kv.second.rqnscts_set) {
      auto v_col_dim = MulDims(rqnscts);
      auto vblock_qnscts = rqnscts;
      vblock_qnscts.insert(vblock_qnscts.begin(), sblk_qnsct);
      auto vblock = new QNBlock<TenElemType>(vblock_qnscts);
      MatGetCols(
          kv.second.v, kv.second.sdim, kv.second.vrdim,
          v_col_offset, v_col_dim,
          vblock->data());
      vblocks.push_back(vblock);
      v_col_offset += v_col_dim;
    }
    delete [] kv.second.v; kv.second.v = nullptr;
  }

  auto s_index_in = Index(sblk_qnscts, IN);
  auto s_index_out = Index(sblk_qnscts, OUT);
  assert(ps != nullptr);
  GQTenFree(ps);
  *ps = GQTensor<TenElemType>({s_index_in, s_index_out});
  ps->blocks() = sblocks;
  assert(pu != nullptr);
  auto u_indexes = SliceFromBegin(indexes, ldims);
  u_indexes.push_back(s_index_out);
  GQTenFree(pu);
  *pu = GQTensor<TenElemType>(u_indexes);
  pu->blocks() = ublocks;
  assert(pvt != nullptr);
  GQTenFree(pvt);
  auto v_indexes = SliceFromEnd(indexes, rdims);
  v_indexes.insert(v_indexes.begin(), s_index_in);
  *pvt = GQTensor<TenElemType>(v_indexes);
  pvt->blocks() = vblocks;
  *ptrunc_err = truncated_blk_svd_data.trunc_err;
  *pD = truncated_blk_svd_data.kept_dim;
}


// Inline functions and templates.
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
} /* gqten */ 
