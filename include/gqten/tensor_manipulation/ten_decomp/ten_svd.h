// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-30 20:13
*
* Description: GraceQ/tensor project. SVD for a symmetric GQTensor.
*/

/**
@file ten_svd.h
@brief SVD for a symmetric GQTensor.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_SVD_H
#define GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_SVD_H


#include "gqten/framework/bases/executor.h"                           // Executor
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_decomp_basic.h"    // GenIdxTenDecompDataBlkMats, IdxDataBlkMatMap

#include <utility>    // pair

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


// Some helpers


// (unnormalized sv, data_blk_mat, mat_dim_label, sv2)
using TruncedSVInfo = std::tuple<GQTEN_Double, size_t, size_t, GQTEN_Double>;


struct SDataBlkInfo {
  size_t blk_coor;
  size_t blk_dim;
  std::vector<GQTEN_Double> svs;
  std::vector<size_t> mat_dims;

  SDataBlkInfo(void) = default;

  SDataBlkInfo(
      const size_t blk_coor,
      const size_t blk_dim,
      const std::vector<GQTEN_Double> &svs,
      const std::vector<size_t> &mat_dims
  ) : blk_coor(blk_coor),
      blk_dim(blk_dim),
      svs(svs),
      mat_dims(mat_dims) {}
};


struct UVtDataBlkInfo {
  CoorsT blk_coors;
  size_t data_blk_mat_idx;
  size_t offset;
  size_t m;
  size_t n;

  UVtDataBlkInfo(
      const CoorsT &blk_coors,
      const size_t data_blk_mat_idx,
      const size_t offset,
      const size_t m,
      const size_t n
  ) : blk_coors(blk_coors),
      data_blk_mat_idx(data_blk_mat_idx),
      offset(offset),
      m(m), n(n) {}
};

using UVtDataBlkInfoVec = std::vector<UVtDataBlkInfo>;
using UVtDataBlkInfoVecPair = std::pair<UVtDataBlkInfoVec, UVtDataBlkInfoVec>;


/**
Tensor SVD executor.

@tparam TenElemT Element type of tensors.
@tparam QNT Quantum number type of tensors.
*/
template <typename TenElemT, typename QNT>
class TensorSVDExecutor : public Executor {
public:
  TensorSVDExecutor(
      const GQTensor<TenElemT, QNT> *,
      const size_t,
      const QNT &,
      const GQTEN_Double, const size_t, const size_t,
      GQTensor<TenElemT, QNT> *,
      GQTensor<GQTEN_Double, QNT> *,
      GQTensor<TenElemT, QNT> *,
      GQTEN_Double *, size_t *
  );

  ~TensorSVDExecutor(void) = default;

  void Execute(void) override;

private:
  const GQTensor<TenElemT, QNT> *pt_;
  const size_t ldims_;
  const QNT &lqndiv_;
  const GQTEN_Double trunc_err_;
  const size_t Dmin_;
  const size_t Dmax_;
  GQTensor<TenElemT, QNT> *pu_;
  GQTensor<GQTEN_Double, QNT> *ps_;
  GQTensor<TenElemT, QNT> *pvt_;
  GQTEN_Double *pactual_trunc_err_;
  size_t *pD_;
  IdxDataBlkMatMap<QNT> idx_ten_decomp_data_blk_mat_map_;
  std::vector<TruncedSVInfo> CalcTruncedSVInfo_(
    const std::map<size_t, DataBlkMatSvdRes<TenElemT>> &
  );
  void ConstructSVDResTens_(
    const std::vector<TruncedSVInfo> &,
    const std::map<size_t, DataBlkMatSvdRes<TenElemT>> &
  );
  UVtDataBlkInfoVecPair CreateSVDResTens_(
      const std::map<size_t, SDataBlkInfo> &
  );
  void FillSVDResTens_(
      const std::map<size_t, DataBlkMatSvdRes<TenElemT>> &,
      const std::map<size_t, SDataBlkInfo> &,
      const UVtDataBlkInfoVecPair &
  );
};


/**
Initialize a tensor SVD executor.

@param pt A pointer to to-be SVD decomposed tensor \f$ T \f$. The rank of \f$ T
       \f$ should be larger then 1.
@param ldims Number of indexes on the left hand side of the decomposition.
@param lqndiv Quantum number divergence of the result \f$ U \f$ tensor.
@param trunc_err The target truncation error.
@param Dmin The target minimal kept dimensions for the truncated SVD decomposition.
@param Dmax The target maximal kept dimensions for the truncated SVD decomposition.
@param pu A pointer to result \f$ U \f$ tensor.
@param ps A pointer to result \f$ S \f$ tensor.
@param pvt A pointer to result \f$ V^{\dagger} \f$ tensor.
@param pactual_trunc_err A pointer to actual truncation error after the truncation.
@param pD A pointer to actual kept dimensions after the truncation.
*/
template <typename TenElemT, typename QNT>
TensorSVDExecutor<TenElemT, QNT>::TensorSVDExecutor(
    const GQTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    const GQTEN_Double trunc_err, const size_t Dmin, const size_t Dmax,
    GQTensor<TenElemT, QNT> *pu,
    GQTensor<GQTEN_Double, QNT> *ps,
    GQTensor<TenElemT, QNT> *pvt,
    GQTEN_Double *pactual_trunc_err, size_t *pD
) : pt_(pt),
    ldims_(ldims),
    lqndiv_(lqndiv),
    trunc_err_(trunc_err), Dmin_(Dmin), Dmax_(Dmax),
    pu_(pu), ps_(ps), pvt_(pvt),
    pactual_trunc_err_(pactual_trunc_err), pD_(pD) {
  assert(pt_->Rank() >= 2);
  assert(pu_->IsDefault());
  assert(ps_->IsDefault());
  assert(pvt_->IsDefault());
  assert(trunc_err_ >= 0);
  assert(Dmin_ <= Dmax_);

  idx_ten_decomp_data_blk_mat_map_ = GenIdxTenDecompDataBlkMats(
                                         *pt_,
                                         ldims_,
                                         lqndiv_
                                     );

  SetStatus(ExecutorStatus::INITED);
}


/**
Execute tensor SVD calculation.
*/
template <typename TenElemT, typename QNT>
void TensorSVDExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

  auto idx_raw_data_svd_res = pt_->GetBlkSparDataTen().DataBlkDecompSVD(
                                  idx_ten_decomp_data_blk_mat_map_
                              );
  auto kept_sv_info = CalcTruncedSVInfo_(idx_raw_data_svd_res);
  ConstructSVDResTens_(kept_sv_info, idx_raw_data_svd_res);
  DeleteDataBlkMatSvdResMap(idx_raw_data_svd_res);

  SetStatus(ExecutorStatus::FINISH);
}


/**
Function version for tensor SVD.

@tparam TenElemT The element type of the tensors.
@tparam QNT The quantum number type of the tensors.

@param pt A pointer to to-be SVD decomposed tensor \f$ T \f$. The rank of \f$ T
       \f$ should be larger then 1.
@param ldims Number of indexes on the left hand side of the decomposition.
@param lqndiv Quantum number divergence of the result \f$ U \f$ tensor.
@param trunc_err The target truncation error.
@param Dmin The target minimal kept dimensions for the truncated SVD decomposition.
@param Dmax The target maximal kept dimensions for the truncated SVD decomposition.
@param pu A pointer to result \f$ U \f$ tensor.
@param ps A pointer to result \f$ S \f$ tensor.
@param pvt A pointer to result \f$ V^{\dagger} \f$ tensor.
@param pactual_trunc_err A pointer to actual truncation error after the truncation.
@param pD A pointer to actual kept dimensions after the truncation.
*/
template <typename TenElemT, typename QNT>
void SVD(
    const GQTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    const GQTEN_Double trunc_err, const size_t Dmin, const size_t Dmax,
    GQTensor<TenElemT, QNT> *pu,
    GQTensor<GQTEN_Double, QNT> *ps,
    GQTensor<TenElemT, QNT> *pvt,
    GQTEN_Double *pactual_trunc_err, size_t *pD
) {
  TensorSVDExecutor<TenElemT, QNT> ten_svd_executor(
      pt,
      ldims,
      lqndiv,
      trunc_err, Dmin, Dmax,
      pu, ps, pvt, pactual_trunc_err, pD
  );
  ten_svd_executor.Execute();
}


GQTEN_Double SumSV2(
    const std::vector<TruncedSVInfo> &trunced_sv_info,
    const size_t beg,
    const size_t end
) {
  GQTEN_Double sv2_sum = 0.0;
  for (size_t i = beg; i < end; ++i) {
    sv2_sum += std::get<3>(trunced_sv_info[i]);
  }
  return sv2_sum;
}


/**
Get truncated singular value information.
*/
template <typename TenElemT, typename QNT>
std::vector<TruncedSVInfo> TensorSVDExecutor<TenElemT, QNT>::CalcTruncedSVInfo_(
    const std::map<size_t, DataBlkMatSvdRes<TenElemT>> &idx_svd_res_map
) {
  std::vector<TruncedSVInfo> trunced_sv_info;
  trunced_sv_info.reserve(Dmin_);
  for (auto &idx_svd_res : idx_svd_res_map) {
    auto idx = idx_svd_res.first;
    auto svd_res = idx_svd_res.second;
    auto m = svd_res.m;
    auto n = svd_res.n;
    auto k = svd_res.k;
    auto s = svd_res.s;
    for (size_t i = 0; i < k; ++i) {
      GQTEN_Double &sv = s[i];
      if (sv != 0.0) {
        trunced_sv_info.emplace_back(std::make_tuple(sv, idx, i, sv * sv));
      }
    }
  }
  size_t total_sv_size = trunced_sv_info.size();

  // No truncate
  if (total_sv_size <= Dmin_) {
    *pactual_trunc_err_ = 0.0;
    *pD_ = total_sv_size;
    return trunced_sv_info;
  }

  // Truncate
  std::sort(
      trunced_sv_info.begin(),
      trunced_sv_info.end(),
      [] (
          const TruncedSVInfo &sv_info_a,
          const TruncedSVInfo &sv_info_b
      ) -> bool {
        return std::get<0>(sv_info_a) > std::get<0>(sv_info_b);
      }
  );
  GQTEN_Double total_kept_sv2_sum = SumSV2(trunced_sv_info, 0, trunced_sv_info.size());
  GQTEN_Double target_kept_sv2_sum = total_kept_sv2_sum * (1 - trunc_err_);
  size_t kept_dim = Dmin_;
  GQTEN_Double kept_sv2_sum = SumSV2(trunced_sv_info, 0, kept_dim);
  size_t next_kept_dim;
  GQTEN_Double next_kept_sv2_sum;
  while (true) {
    if (kept_sv2_sum > target_kept_sv2_sum) { break; }
    next_kept_dim = kept_dim + 1;
    if (next_kept_dim > total_sv_size) { break; }
    next_kept_sv2_sum = kept_sv2_sum +
                        std::get<3>(trunced_sv_info[next_kept_dim - 1]);
    if (next_kept_dim > Dmax_) {
      break;
    } else {
      kept_dim = next_kept_dim;
      kept_sv2_sum = next_kept_sv2_sum;
    }
  }
  assert(kept_dim <= total_sv_size);
  GQTEN_Double actual_trunc_err = 1 - (kept_sv2_sum / total_kept_sv2_sum);
  *pactual_trunc_err_ = actual_trunc_err > 0 ? actual_trunc_err : 0;
  *pD_ = kept_dim;
  trunced_sv_info.resize(kept_dim);
  // Sort back
  std::sort(
      trunced_sv_info.begin(),
      trunced_sv_info.end(),
      [] (
          const TruncedSVInfo &sv_info_a,
          const TruncedSVInfo &sv_info_b
      ) -> bool {
        if (std::get<1>(sv_info_a) < std::get<1>(sv_info_b)) {
          return true;
        } else if (std::get<1>(sv_info_a) == std::get<1>(sv_info_b)) {
          return std::get<2>(sv_info_a) < std::get<2>(sv_info_b);
        } else {
          return false;
        }
      }
  );
  return trunced_sv_info;
}


std::map<size_t, SDataBlkInfo> GenIdxSDataBlkInfoMap(
    const std::vector<TruncedSVInfo> &trunced_sv_info
) {
  std::map<size_t, SDataBlkInfo> idx_s_data_blk_info_map;
  size_t blk_coor = 0;
  std::vector<GQTEN_Double> svs;
  std::vector<size_t> mat_dims;
  size_t blk_dim = 0;
  auto total_sv_size = trunced_sv_info.size();
  for (size_t i = 0; i < total_sv_size; ++i) {
    svs.push_back(std::get<0>(trunced_sv_info[i]));
    mat_dims.push_back(std::get<2>(trunced_sv_info[i]));
    blk_dim += 1;
    auto idx = std::get<1>(trunced_sv_info[i]);
    if (
        (i == total_sv_size - 1) ||
        (idx != std::get<1>(trunced_sv_info[i+1]))
    ) {
      idx_s_data_blk_info_map[
          idx
      ] = SDataBlkInfo(blk_coor, blk_dim, svs, mat_dims);
      blk_coor += 1;
      blk_dim = 0;
      svs.clear();
      mat_dims.clear();
    }
  }
  return idx_s_data_blk_info_map;
}


inline CoorsT GetSVDUCoors(
    const CoorsT &t_coors, const size_t ldims, const size_t mid_coor
) {
  CoorsT u_coors(t_coors.begin(), t_coors.begin() + ldims);
  u_coors.push_back(mid_coor);
  return u_coors;
}


inline CoorsT GetSVDVtCoors(
    const CoorsT &t_coors, const size_t ldims, const size_t mid_coor
) {
  CoorsT vt_coors {mid_coor};
  vt_coors.insert(vt_coors.end(), t_coors.begin() + ldims, t_coors.end());
  return vt_coors;
}


/**
Construct SVD result tensors.
*/
template <typename TenElemT, typename QNT>
void TensorSVDExecutor<TenElemT, QNT>::ConstructSVDResTens_(
    const std::vector<TruncedSVInfo> &trunced_sv_info,
    const std::map<size_t, DataBlkMatSvdRes<TenElemT>> &idx_raw_data_svd_res
) {
  auto idx_s_data_blk_info_map = GenIdxSDataBlkInfoMap(trunced_sv_info);
  auto u_vt_data_blks_info = CreateSVDResTens_(idx_s_data_blk_info_map);
  FillSVDResTens_(
      idx_raw_data_svd_res,
      idx_s_data_blk_info_map,
      u_vt_data_blks_info
  );
}


template <typename QNT>
QNSectorVec<QNT> GenMidQNSects(
    const IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map,
    const std::map<size_t, SDataBlkInfo> idx_s_data_blk_info_map
) {
  QNSectorVec<QNT> mid_qnscts;
  mid_qnscts.reserve(idx_s_data_blk_info_map.size());
  for (auto &idx_s_data_blk_info : idx_s_data_blk_info_map) {
    auto idx = idx_s_data_blk_info.first;
    mid_qnscts.push_back(
        QNSector<QNT>(
            idx_data_blk_mat_map.at(idx).mid_qn,
            idx_s_data_blk_info.second.blk_dim
        )
    );
  }
  return mid_qnscts;
}


template <typename TenElemT, typename QNT>
UVtDataBlkInfoVecPair TensorSVDExecutor<TenElemT, QNT>::CreateSVDResTens_(
    const std::map<size_t, SDataBlkInfo> &idx_s_data_blk_info_map
) {
  // Initialize u, s, vt tensors
  auto mid_qnscts = GenMidQNSects(
                        idx_ten_decomp_data_blk_mat_map_,
                        idx_s_data_blk_info_map
                    );
  auto mid_index_out = Index<QNT>(mid_qnscts, GQTenIndexDirType::OUT);
  auto mid_index_in = InverseIndex(mid_index_out);
  auto t_indexes = pt_->GetIndexes();
  IndexVec<QNT> u_indexes(t_indexes.begin(), t_indexes.begin() + ldims_);
  u_indexes.push_back(mid_index_out);
  (*pu_) = GQTensor<TenElemT, QNT>(std::move(u_indexes));
  IndexVec<QNT> vt_indexes {mid_index_in};
  vt_indexes.insert(
      vt_indexes.end(),
      t_indexes.begin() + ldims_, t_indexes.end()
  );
  (*pvt_) = GQTensor<TenElemT, QNT>(std::move(vt_indexes));
  IndexVec<QNT> s_indexes {mid_index_in, mid_index_out};
  (*ps_) = GQTensor<GQTEN_Double, QNT>(std::move(s_indexes));

  // Insert empty data blocks
  UVtDataBlkInfoVec u_data_blks_info, vt_data_blks_info;
  for (auto &idx_s_data_blk_info : idx_s_data_blk_info_map) {
    auto data_blk_mat_idx = idx_s_data_blk_info.first;
    auto s_data_blk_info = idx_s_data_blk_info.second;
    auto blk_coor = s_data_blk_info.blk_coor;
    auto blk_dim = s_data_blk_info.blk_dim;
    auto mat_dims = s_data_blk_info.mat_dims;
    auto data_blk_mat = idx_ten_decomp_data_blk_mat_map_.at(data_blk_mat_idx);

    ps_->GetBlkSparDataTen().DataBlkInsert({blk_coor, blk_coor}, false);

    for (auto &row_sct : data_blk_mat.row_scts) {
      CoorsT u_data_blk_coors(std::get<0>(row_sct));
      u_data_blk_coors.push_back(blk_coor);
      pu_->GetBlkSparDataTen().DataBlkInsert(u_data_blk_coors, false);
      u_data_blks_info.push_back(
          UVtDataBlkInfo(
              u_data_blk_coors,
              data_blk_mat_idx, std::get<1>(row_sct),
              std::get<2>(row_sct), blk_dim
          )
      );
    }

    for (auto &col_sct : data_blk_mat.col_scts) {
      CoorsT vt_data_blk_coors {blk_coor};
      auto rpart_blk_coors = std::get<0>(col_sct);
      vt_data_blk_coors.insert(
          vt_data_blk_coors.end(),
          rpart_blk_coors.begin(), rpart_blk_coors.end()
      );
      pvt_->GetBlkSparDataTen().DataBlkInsert(vt_data_blk_coors, false);
      vt_data_blks_info.push_back(
          UVtDataBlkInfo(
              vt_data_blk_coors,
              data_blk_mat_idx, std::get<1>(col_sct),
              blk_dim, std::get<2>(col_sct)
          )
      );
    }
  }
  return std::make_pair(u_data_blks_info, vt_data_blks_info);
}


template <typename TenElemT, typename QNT>
void TensorSVDExecutor<TenElemT, QNT>::FillSVDResTens_(
    const std::map<size_t, DataBlkMatSvdRes<TenElemT>> &idx_svd_res_map,
    const std::map<size_t, SDataBlkInfo> &idx_s_data_blk_info_map,
    const UVtDataBlkInfoVecPair &u_vt_data_blks_info
) {
  // Fill s tensor
  ps_->GetBlkSparDataTen().Allocate(true);    // Initialize memory to 0 here
  size_t s_coor = 0;
  for (auto &idx_s_data_blk_info : idx_s_data_blk_info_map) {
    auto s_data_blk_info = idx_s_data_blk_info.second;
    for (size_t i = 0; i < s_data_blk_info.blk_dim; ++i) {
      ps_->SetElem({s_coor, s_coor}, s_data_blk_info.svs[i]);
      s_coor += 1;
    }
  }
  assert(s_coor == ps_->GetShape()[0]);

  // Fill u tensor
  pu_->GetBlkSparDataTen().Allocate();
  auto u_data_blks_info = u_vt_data_blks_info.first;
  for (auto u_data_blk_info : u_data_blks_info) {
    auto svd_res = idx_svd_res_map.at(u_data_blk_info.data_blk_mat_idx);
    pu_->GetBlkSparDataTen().DataBlkCopySVDUdata(
        u_data_blk_info.blk_coors,
        u_data_blk_info.m,
        u_data_blk_info.n,
        u_data_blk_info.offset,
        svd_res.u, svd_res.m, svd_res.k,
        idx_s_data_blk_info_map.at(u_data_blk_info.data_blk_mat_idx).mat_dims
    );
  }

  pvt_->GetBlkSparDataTen().Allocate();
  auto vt_data_blks_info = u_vt_data_blks_info.second;
  for (auto &vt_data_blk_info : vt_data_blks_info) {
    auto svd_res = idx_svd_res_map.at(vt_data_blk_info.data_blk_mat_idx);
    pvt_->GetBlkSparDataTen().DataBlkCopySVDVtData(
        vt_data_blk_info.blk_coors,
        vt_data_blk_info.m,
        vt_data_blk_info.n,
        vt_data_blk_info.offset,
        svd_res.vt, svd_res.k, svd_res.n,
        idx_s_data_blk_info_map.at(vt_data_blk_info.data_blk_mat_idx).mat_dims
    );
  }
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_SVD_H */
