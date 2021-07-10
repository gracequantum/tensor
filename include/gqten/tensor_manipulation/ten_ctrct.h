// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-27 10:02
*
* Description: GraceQ/tensor project. Contract two tensors.
*/

/**
@file ten_ctrct.h
@brief Contract two tensors.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_CTRCT_H
#define GQTEN_TENSOR_MANIPULATION_TEN_CTRCT_H


#include "gqten/framework/bases/executor.h"                 // Executor
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/basic_operations.h"     // ToComplex

#include <vector>     // vector

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


// Forward declarations
template <typename TenElemT, typename QNT>
void TenCtrctInitResTen(
    const GQTensor<TenElemT, QNT> *,
    const GQTensor<TenElemT, QNT> *,
    const std::vector<std::vector<size_t>> &,
    GQTensor<TenElemT, QNT> *
);


/**
Tensor contraction executor.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.
*/
template <typename TenElemT, typename QNT>
class TensorContractionExecutor : public Executor {
public:
  TensorContractionExecutor(
      const GQTensor<TenElemT, QNT> *,
      const GQTensor<TenElemT, QNT> *,
      const std::vector<std::vector<size_t>> &,
      GQTensor<TenElemT, QNT> *
  );

  void Execute(void) override;

private:
  const GQTensor<TenElemT, QNT> *pa_;
  const GQTensor<TenElemT, QNT> *pb_;
  GQTensor<TenElemT, QNT> *pc_;
  const std::vector<std::vector<size_t>> &axes_set_;
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks_;
};


/**
Initialize a tensor contraction executor.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template <typename TenElemT, typename QNT>
TensorContractionExecutor<TenElemT, QNT>::TensorContractionExecutor(
    const GQTensor<TenElemT, QNT> *pa,
    const GQTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<TenElemT, QNT> *pc
) : pa_(pa), pb_(pb), axes_set_(axes_set), pc_(pc) {
  assert(pc_->IsDefault());    // Only empty tensor can take the result
  // Check indexes matching
#ifndef NDEBUG
  auto indexesa = pa->GetIndexes();
  auto indexesb = pb->GetIndexes();
  for(size_t i = 0; i < axes_set[0].size(); ++i){
    assert(indexesa[axes_set[0][i]] == InverseIndex(indexesb[axes_set[1][i]]));
  }
#endif
  TenCtrctInitResTen(pa_, pb_, axes_set_, pc_);
  raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
                              pa_->GetBlkSparDataTen(),
                              pb_->GetBlkSparDataTen(),
                              axes_set_
                          );

  SetStatus(ExecutorStatus::INITED);
}


/**
Allocate memory and perform raw data contraction calculation.
*/
template <typename TenElemT, typename QNT>
void TensorContractionExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

  pc_->GetBlkSparDataTen().CtrctTwoBSDTAndAssignIn(
      pa_->GetBlkSparDataTen(),
      pb_->GetBlkSparDataTen(),
      raw_data_ctrct_tasks_
  );

  SetStatus(ExecutorStatus::FINISH);
}


/**
Function version for tensor contraction.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template <typename TenElemT, typename QNT>
void Contract(
    const GQTensor<TenElemT, QNT> *pa,
    const GQTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<TenElemT, QNT> *pc
) {
  TensorContractionExecutor<TenElemT, QNT> ten_ctrct_executor(
      pa,
      pb,
      axes_set,
      pc
  );
  ten_ctrct_executor.Execute();
}


template <typename QNT>
void Contract(
    const GQTensor<GQTEN_Double, QNT> *pa,
    const GQTensor<GQTEN_Complex, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<GQTEN_Complex, QNT> *pc
) {
  auto cplx_a = ToComplex(*pa);
  Contract(&cplx_a, pb, axes_set, pc);
}


template <typename QNT>
void Contract(
    const GQTensor<GQTEN_Complex, QNT> *pa,
    const GQTensor<GQTEN_Double, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<GQTEN_Complex, QNT> *pc
) {
  auto cplx_b = ToComplex(*pb);
  Contract(pa, &cplx_b, axes_set, pc);
}


/**
Initialize tensor contraction result tensor.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template <typename TenElemT, typename QNT>
void TenCtrctInitResTen(
    const GQTensor<TenElemT, QNT> *pa,
    const GQTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<TenElemT, QNT> *pc
) {
  auto a_ctrct_axes = axes_set[0];
  auto b_ctrct_axes = axes_set[1];
  auto a_rank = pa->Rank();
  auto b_rank = pb->Rank();
  auto c_rank = a_rank + b_rank - 2*(a_ctrct_axes.size());
  IndexVec<QNT> c_idxs;
  c_idxs.reserve(c_rank);
  auto a_idxs = pa->GetIndexes();
  auto b_idxs = pb->GetIndexes();
  for (size_t i = 0; i < a_rank; ++i) {
    if (
        std::find(a_ctrct_axes.begin(), a_ctrct_axes.end(), i) ==
        a_ctrct_axes.end()
    ) {
      c_idxs.push_back(a_idxs[i]);
    }
  }
  for (size_t i = 0; i < b_rank; ++i) {
    if (
        std::find(b_ctrct_axes.begin(), b_ctrct_axes.end(), i) ==
        b_ctrct_axes.end()
    ) {
      c_idxs.push_back(b_idxs[i]);
    }
  }
  (*pc) = GQTensor<TenElemT, QNT>(std::move(c_idxs));
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_CTRCT_H */
