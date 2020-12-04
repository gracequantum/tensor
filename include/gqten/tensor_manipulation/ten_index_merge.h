// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-30 14:44
*
* Description: GraceQ/tensor project. Merge Indexes of a tensor.
*/

/**
@file ten_index_merge.h
@brief Merge Indexes of a tensor.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_INDEX_MERGE_H
#define GQTEN_TENSOR_MANIPULATION_TEN_INDEX_MERGE_H


#include "gqten/framework/bases/executor.h"                         // Executor
#include "gqten/gqtensor_all.h"


namespace gqten {


using AxesGroupNewIdxDir = std::pair<std::vector<size_t>, GQTenIndexDirType>;


/**
Tensor index merge executor.

@tparam QNT The quantum number type of the tensors.
@tparam TenElemT The type of tensor elements.
*/
template <typename QNT, typename TenElemT>
class TensorIndexMergeExector : public Executor {
public:
  TensorIndexMergeExector(
      GQTensor<TenElemT, QNT> *,
      const std::vector<AxesGroupNewIdxDir> &
  );

  std::vector<GQTensor<TenElemT, QNT>> Execute(bool) override;
};


template <typename QNT, typename TenElemT>
std::vector<GQTensor<TenElemT, QNT>> IndexMerge(
    GQTensor<TenElemT, QNT> *pt,
    const std::vector<AxesGroupNewIdxDir> &axes_group_new_idx_idr,
    const bool create_restorer = false
) {
  TensorIndexMergeExector<QNT, TenElemT> ten_index_merge_executor(
      pt,
      axes_group_new_idx_idr
  );
  return ten_index_merge_executor.Execute(create_restorer);
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_INDEX_MERGE_H */
