// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-10 18:23
*
* Description: GraceQ/tensor project. Perform linear combination of tensors.
*/

/**
@file ten_linear_combine.h
@brief Perform linear combination of tensors.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_LINEAR_COMBINE_H
#define GQTEN_TENSOR_MANIPULATION_TEN_LINEAR_COMBINE_H


#include "gqten/framework/bases/executor.h"                         // Executor
#include "gqten/gqtensor_all.h"

#include <unordered_set>    // unordered_set
#include <map>              // map

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


template <typename CoefT>
using TenLinCmbDataCopyTasks = std::vector<RawDataCopyAndScaleTask<CoefT>>;

template <typename CoefT>
using TenIdxTenLinCmbDataCopyTasksMap = std::map<
                                            size_t,
                                            TenLinCmbDataCopyTasks<CoefT>
                                        >;


/**
Tensors linear combination executor. \f$ T = a \cdot A + b \cdot B + \cdots + \beta T\f$.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@note For linear combination, rank 0 tensor (scalar) is not supported.
*/
template <typename TenElemT, typename QNT>
class TensorLinearCombinationExecutor : public Executor {
public:
  using TenT = GQTensor<TenElemT, QNT>;

  TensorLinearCombinationExecutor(
      const std::vector<TenElemT> &,
      const std::vector<TenT *> &,
      const TenElemT,
      TenT *
  );

  void Execute(void) override;

private:
  std::vector<TenElemT> coefs_;
  std::vector<TenT *> tens_;
  TenElemT beta_;
  TenT *pres_;

  TenT actual_res_;
  TenIdxTenLinCmbDataCopyTasksMap<
      TenElemT
  > ten_idx_ten_lin_cmb_data_copy_tasks_map_;
};


/**
Initialize a tensor linear combination executor.

@param coefs Coefficients of each to-be linear combined tensor.
@param tens To-be linear combined tensors. They mush have same indexes.
@param beta \f$ \beta \f$.
@param pres The pointer to result.
*/
template <typename TenElemT, typename QNT>
TensorLinearCombinationExecutor<TenElemT, QNT>::TensorLinearCombinationExecutor(
    const std::vector<TenElemT> &coefs,
    const std::vector<TenT *> &tens,
    const TenElemT beta,
    TenT *pres
) : coefs_(coefs), tens_(tens), beta_(beta), pres_(pres) {
  assert(coefs_.size() != 0);
  assert(coefs_.size() == tens_.size());
#ifndef NDEBUG
  auto indexes = tens_[0]->GetIndexes();
  for (size_t i = 1; i < tens_.size(); ++i) {
    assert(tens_[i]->GetIndexes() == indexes);
  }
  if (!pres_->IsDefault()) {
    assert(pres_->GetIndexes() == indexes);
  }
#endif

  actual_res_ = TenT(tens_[0]->GetIndexes());

  // Deal with input result tensor and its coefficient beta
  if (beta_ != 0.0) {
    assert(!pres_->IsDefault());
    coefs_.push_back(beta_);
    tens_.push_back(pres_);
  }
  std::unordered_set<size_t> res_data_blk_idx_set;
  for (size_t i = 0; i < coefs_.size(); ++i) {
    if (coefs_[i] != 0.0) {
      ten_idx_ten_lin_cmb_data_copy_tasks_map_[i] =
          GenTenLinearCombineDataCopyTasks(
              coefs_[i],
              tens_[i],
              actual_res_,
              res_data_blk_idx_set
          );
    }
  }

  SetStatus(ExecutorStatus::INITED);
}


template <typename TenElemT, typename QNT>
void TensorLinearCombinationExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

  actual_res_.GetBlkSparDataTen().Allocate();
  for (auto &ten_idx_tasks : ten_idx_ten_lin_cmb_data_copy_tasks_map_) {
    auto ten_idx = ten_idx_tasks.first;
    auto ten_bsdt_raw_data = tens_[
                                 ten_idx
                             ]->GetBlkSparDataTen().GetActualRawDataPtr();
    auto tasks = ten_idx_tasks.second;
    for (auto &task : tasks) {
      actual_res_.GetBlkSparDataTen().DataBlkCopyAndScale(task, ten_bsdt_raw_data);
    }
  }
  (*pres_) = std::move(actual_res_);

  SetStatus(ExecutorStatus::FINISH);
}


template <typename TenElemT, typename QNT>
TenLinCmbDataCopyTasks<TenElemT> GenTenLinearCombineDataCopyTasks(
    const TenElemT coef,
    const GQTensor<TenElemT, QNT> *pten,
    GQTensor<TenElemT, QNT> &res,
    std::unordered_set<size_t> &res_data_blk_idx_set
) {
  TenLinCmbDataCopyTasks<TenElemT> tasks;
  auto idx_data_blk_map = pten->GetBlkSparDataTen().GetBlkIdxDataBlkMap();
  auto data_blk_num = idx_data_blk_map.size();
  tasks.reserve(data_blk_num);
  bool copy_and_add;
  for (auto &idx_data_blk : idx_data_blk_map) {
    auto idx = idx_data_blk.first;
    auto data_blk = idx_data_blk.second;
    if (res_data_blk_idx_set.find(idx) != res_data_blk_idx_set.end()) {
      copy_and_add = true;
    } else {
      copy_and_add = false;
      res.GetBlkSparDataTen().DataBlkInsert(data_blk.blk_coors, false);
      res_data_blk_idx_set.insert(idx);
    }
    tasks.push_back(
        RawDataCopyAndScaleTask<TenElemT>(
            data_blk.data_offset,
            data_blk.size,
            data_blk.blk_coors,
            coef,
            copy_and_add
        )
    );
  }
  return tasks;
}


/**
Function version of tensors linear combination. \f$ T = a \cdot A + b \cdot B + \cdots + \beta T\f$.

@param coefs Coefficients of each to-be linear combined tensor.
@param tens To-be linear combined tensors. They mush have same indexes.
@param beta \f$ \beta \f$.
@param pres The pointer to result.
*/
template <typename TenElemT, typename QNT>
void LinearCombine(
    const std::vector<TenElemT> &coefs,
    const std::vector<GQTensor<TenElemT, QNT> *> &tens,
    const TenElemT beta,
    GQTensor<TenElemT, QNT> *pres
) {
  auto ten_lin_cmb_exector = TensorLinearCombinationExecutor<TenElemT, QNT>(
                                 coefs, tens,
                                 beta, pres
                             );
  ten_lin_cmb_exector.Execute();
}


// Other function versions
inline std::vector<GQTEN_Complex> ToCplxVec(
    const std::vector<GQTEN_Double> &real_v
) {
  std::vector<GQTEN_Complex> cplx_v;
  cplx_v.reserve(real_v.size());
  for (auto &e : real_v) {
    cplx_v.emplace_back(e);
  }
  return cplx_v;
}


template <typename QNT>
void LinearCombine(
    const std::vector<GQTEN_Double> &coefs,
    const std::vector<GQTensor<GQTEN_Complex, QNT> *> &tens,
    const GQTEN_Complex beta,
    GQTensor<GQTEN_Complex, QNT> *pres
) {
  LinearCombine(ToCplxVec(coefs), tens, beta, pres);
}


template <typename QNT>
void LinearCombine(
    const size_t size,
    const GQTEN_Double *pcoefs,
    const std::vector<GQTensor<GQTEN_Double, QNT> *> &tens,
    const GQTEN_Double beta,
    GQTensor<GQTEN_Double, QNT> *pres
) {
  std::vector<GQTEN_Double> coefs;
  coefs.resize(size);
  std::copy_n(pcoefs, size, coefs.begin());
  std::vector<GQTensor<GQTEN_Double, QNT> *> actual_tens;
  actual_tens.resize(size);
  std::copy_n(tens.begin(), size, actual_tens.begin());
  LinearCombine(coefs, actual_tens, beta, pres);
}


template <typename QNT>
void LinearCombine(
    const size_t size,
    const GQTEN_Double *pcoefs,
    const std::vector<GQTensor<GQTEN_Complex, QNT> *> &tens,
    const GQTEN_Complex beta,
    GQTensor<GQTEN_Complex, QNT> *pres
) {
  std::vector<GQTEN_Complex> coefs;
  coefs.resize(size);
  std::copy_n(pcoefs, size, coefs.begin());
  std::vector<GQTensor<GQTEN_Complex, QNT> *> actual_tens;
  actual_tens.resize(size);
  std::copy_n(tens.begin(), size, actual_tens.begin());
  LinearCombine(coefs, actual_tens, beta, pres);
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_LINEAR_COMBINE_H */
