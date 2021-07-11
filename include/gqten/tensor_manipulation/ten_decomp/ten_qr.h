// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-07-06 13:26
*
* Description: GraceQ/tensor project. QR for a symmetric GQTensor.
*/

/**
@file ten_qr.h
@brief QR for a symmetric GQTensor.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_QR_H
#define GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_QR_H


#include "gqten/framework/bases/executor.h"     // Executor
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_decomp_basic.h"    // GenIdxTenDecompDataBlkMats

#include <algorithm>    // min

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


struct QRDataBlkInfo {
  CoorsT blk_coors;
  size_t data_blk_mat_idx;
  size_t offset;
  size_t m;
  size_t n;

  QRDataBlkInfo(
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

using QRDataBlkInfoVec = std::vector<QRDataBlkInfo>;
using QRDataBlkInfoVecPair = std::pair<QRDataBlkInfoVec, QRDataBlkInfoVec>;


/**
Tensor QR executor.
*/
template <typename TenElemT, typename QNT>
class TensorQRExecutor : public Executor {
public:
  TensorQRExecutor(
      const GQTensor<TenElemT, QNT> *,
      const size_t,
      const QNT &,
      GQTensor<TenElemT, QNT> *,
      GQTensor<TenElemT, QNT> *
  );

  ~TensorQRExecutor(void) = default;

  void Execute(void) override;

private:
  const GQTensor<TenElemT, QNT> *pt_;
  const size_t ldims_;
  const QNT &lqndiv_;
  GQTensor<TenElemT, QNT> *pq_;
  GQTensor<TenElemT, QNT> *pr_;
  IdxDataBlkMatMap<QNT> idx_ten_decomp_data_blk_mat_map_;
  void ConstructQRResTens_(const std::map<size_t, DataBlkMatQrRes<TenElemT>> &);
  QRDataBlkInfoVecPair CreatQRResTens_(void);
  void FillQRResTens_(
      const std::map<size_t, DataBlkMatQrRes<TenElemT>> &,
      const QRDataBlkInfoVecPair &
  );
};


/**
Initialize a tensor QR executor.

@tparam TenElemT The element type of the tensors.
@tparam QNT The quantum number type of the tensors.

@param pt A pointer to to-be QR decomposed tensor \f$ T \f$. The rank of \f$ T
       \f$ should be larger than 1.
@param ldims Number of indeses on the left hand side of the decomposition.
@param lqndiv Quantum number divergence of the result \f$ Q \f$ tensor.
@param pq A pointer to result \f$ Q \f$ tensor.
@param pr A pointer to result \f$ R \f$ tensor.
*/
template <typename TenElemT, typename QNT>
TensorQRExecutor<TenElemT, QNT>::TensorQRExecutor(
    const GQTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    GQTensor<TenElemT, QNT> *pq,
    GQTensor<TenElemT, QNT> *pr
) : pt_(pt), ldims_(ldims), lqndiv_(lqndiv), pq_(pq), pr_(pr) {
  assert(pt_->Rank() >= 2);
  assert(ldims_ < pt_->Rank());
  assert(pq_->IsDefault());
  assert(pr_->IsDefault());

  idx_ten_decomp_data_blk_mat_map_ = GenIdxTenDecompDataBlkMats(
                                         *pt_,
                                         ldims_,
                                         lqndiv_
                                     );

  SetStatus(ExecutorStatus::INITED);
}


/**
Execute tensor QR decomposition calculation.
*/
template <typename TenElemT, typename QNT>
void TensorQRExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

  auto idx_raw_data_qr_res = pt_->GetBlkSparDataTen().DataBlkDecompQR(
                                 idx_ten_decomp_data_blk_mat_map_
                             );
  ConstructQRResTens_(idx_raw_data_qr_res);
  DeleteDataBlkMatQrResMap(idx_raw_data_qr_res);

  SetStatus(ExecutorStatus::FINISH);
}


/**
Function version for tensor QR.

@tparam TenElemT The element type of the tensors.
@tparam QNT The quantum number type of the tensors.

@param pt A pointer to to-be QR decomposed tensor \f$ T \f$. The rank of \f$ T
       \f$ should be larger than 1.
@param ldims Number of indeses on the left hand side of the decomposition.
@param lqndiv Quantum number divergence of the result \f$ Q \f$ tensor.
@param pq A pointer to result \f$ Q \f$ tensor.
@param pr A pointer to result \f$ R \f$ tensor.
*/
template <typename TenElemT, typename QNT>
void QR(
    const GQTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    GQTensor<TenElemT, QNT> *pq,
    GQTensor<TenElemT, QNT> *pr
) {
  TensorQRExecutor<TenElemT, QNT> ten_qr_executor(pt, ldims, lqndiv, pq, pr);
  ten_qr_executor.Execute();
}


/**
Construct QR result tensors.
*/
template <typename TenElemT, typename QNT>
void TensorQRExecutor<TenElemT, QNT>::ConstructQRResTens_(
    const std::map<size_t, DataBlkMatQrRes<TenElemT>> &idx_raw_data_qr_res
) {
  auto q_r_data_blks_info = CreatQRResTens_();
  FillQRResTens_(idx_raw_data_qr_res, q_r_data_blks_info);
}


template <typename QNT>
QNSectorVec<QNT> GenMidQNSects(
    const IdxDataBlkMatMap<QNT> &idx_data_blk_mat_map
) {
  QNSectorVec<QNT> mid_qnscts;
  mid_qnscts.reserve(idx_data_blk_mat_map.size());
  for (auto &idx_data_blk_mat : idx_data_blk_mat_map) {
    auto k_dim = std::min(
                     idx_data_blk_mat.second.rows,
                     idx_data_blk_mat.second.cols
                 );
    mid_qnscts.push_back(
        QNSector<QNT>(
            idx_data_blk_mat.second.mid_qn,
            k_dim
        )
    );
  }
  return mid_qnscts;
}


template <typename TenElemT, typename QNT>
QRDataBlkInfoVecPair TensorQRExecutor<TenElemT, QNT>::CreatQRResTens_(void) {
  // Initialize q, r tensors
  auto mid_qnscts = GenMidQNSects(idx_ten_decomp_data_blk_mat_map_);
  auto mid_index_out = Index<QNT>(mid_qnscts, GQTenIndexDirType::OUT);
  auto mid_index_in = InverseIndex(mid_index_out);
  auto t_indexes = pt_->GetIndexes();
  IndexVec<QNT> q_indexes(t_indexes.begin(), t_indexes.begin() + ldims_);
  q_indexes.push_back(mid_index_out);
  (*pq_) = GQTensor<TenElemT, QNT>(std::move(q_indexes));
  IndexVec<QNT> r_indexes{mid_index_in};
  r_indexes.insert(
      r_indexes.end(),
      t_indexes.begin() + ldims_, t_indexes.end()
  );
  (*pr_) = GQTensor<TenElemT, QNT>(std::move(r_indexes));

  // Insert empty data blocks
  QRDataBlkInfoVec q_data_blks_info, r_data_blks_info;
  size_t mid_blk_coor = 0;
  for (
      auto &[data_blk_mat_idx, data_blk_mat] : idx_ten_decomp_data_blk_mat_map_
  ) {
    auto k = std::min(data_blk_mat.rows, data_blk_mat.cols);
    for (auto &row_sct : data_blk_mat.row_scts) {
      CoorsT q_data_blk_coors(std::get<0>(row_sct));
      q_data_blk_coors.push_back(mid_blk_coor);
      pq_->GetBlkSparDataTen().DataBlkInsert(q_data_blk_coors, false);
      q_data_blks_info.push_back(
          QRDataBlkInfo(
              q_data_blk_coors,
              data_blk_mat_idx, std::get<1>(row_sct),
              std::get<2>(row_sct), k
          )
      );
    }

    for (auto &col_sct : data_blk_mat.col_scts) {
      CoorsT r_data_blk_coors{mid_blk_coor};
      auto rpart_blk_coors = std::get<0>(col_sct);
      r_data_blk_coors.insert(
          r_data_blk_coors.end(),
          rpart_blk_coors.begin(), rpart_blk_coors.end()
      );
      pr_->GetBlkSparDataTen().DataBlkInsert(r_data_blk_coors, false);
      r_data_blks_info.push_back(
          QRDataBlkInfo(
              r_data_blk_coors,
              data_blk_mat_idx, std::get<1>(col_sct),
              k, std::get<2>(col_sct)
          )
      );
    }

    mid_blk_coor++;
  }
  return std::make_pair(q_data_blks_info, r_data_blks_info);
}


template <typename TenElemT, typename QNT>
void TensorQRExecutor<TenElemT, QNT>::FillQRResTens_(
    const std::map<size_t, DataBlkMatQrRes<TenElemT>> &idx_qr_res_map,
    const QRDataBlkInfoVecPair &q_r_data_blks_info
) {
  // Fill Q tensor
  pq_->GetBlkSparDataTen().Allocate();
  auto q_data_blks_info = q_r_data_blks_info.first;
  for (auto &q_data_blk_info : q_data_blks_info) {
    auto qr_res = idx_qr_res_map.at(q_data_blk_info.data_blk_mat_idx);
    pq_->GetBlkSparDataTen().DataBlkCopyQRQdata(
        q_data_blk_info.blk_coors,
        q_data_blk_info.m,
        q_data_blk_info.n,
        q_data_blk_info.offset,
        qr_res.q, qr_res.m, qr_res.k
    );
  }

  // Fill R tensor
  pr_->GetBlkSparDataTen().Allocate();
  auto r_data_blks_info = q_r_data_blks_info.second;
  for (auto &r_data_blk_info : r_data_blks_info) {
    auto qr_res = idx_qr_res_map.at(r_data_blk_info.data_blk_mat_idx);
    pr_->GetBlkSparDataTen().DataBlkCopyQRRdata(
        r_data_blk_info.blk_coors,
        r_data_blk_info.m,
        r_data_blk_info.n,
        r_data_blk_info.offset,
        qr_res.r, qr_res.k, qr_res.n
    );
  }
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_DECOMP_TEN_QR_H */
