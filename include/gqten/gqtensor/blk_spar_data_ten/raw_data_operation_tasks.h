// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-23 09:29
*
* Description: GraceQ/tensor project. Raw data operation tasks in block sparse
* data tensor.
*/

/**
@file raw_data_operation_tasks.h
@brief Raw data operation tasks in block sparse data tensor.
*/
#ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATION_TASKS_H
#define GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATION_TASKS_H


#include "gqten/framework/value_t.h"    // ShapeT, CoorsT

#include <vector>       // vector
#include <algorithm>    // sort


namespace gqten {


/**
Task for data transpose.
*/
struct RawDataTransposeTask {
  size_t ten_rank = 0;

  std::vector<size_t> transed_order;

  size_t original_blk_idx = 0;
  ShapeT original_shape;
  size_t original_data_offset = 0;

  size_t transed_blk_idx = 0;
  ShapeT transed_shape;
  size_t transed_data_offset = 0;

  RawDataTransposeTask(
      const size_t ten_rank,
      const std::vector<size_t> &transed_order,
      const size_t original_blk_idx,
      const ShapeT &original_shape,
      const size_t original_data_offset,
      const size_t transed_blk_idx,
      const ShapeT &transed_shape
  ) : ten_rank(ten_rank),
      transed_order(transed_order),
      original_blk_idx(original_blk_idx),
      original_shape(original_shape),
      original_data_offset(original_data_offset),
      transed_blk_idx(transed_blk_idx),
      transed_shape(transed_shape) {}

  static void SortTasksByOriginalBlkIdx(
      std::vector<RawDataTransposeTask> & tasks
  ) {
    std::sort(
        tasks.begin(),
        tasks.end(),
        [] (
            const RawDataTransposeTask &task_a,
            const RawDataTransposeTask &task_b
        ) -> bool {
          return task_a.original_blk_idx < task_b.original_blk_idx;
        }
    );
  }

  static void SortTasksByTranspoedBlkIdx(
      std::vector<RawDataTransposeTask> & tasks
  ) {
    std::sort(
        tasks.begin(),
        tasks.end(),
        [] (
            const RawDataTransposeTask &task_a,
            const RawDataTransposeTask &task_b
        ) -> bool {
          return task_a.transed_blk_idx < task_b.transed_blk_idx;
        }
    );
  }
};


/**
Task for data copy.
*/
struct RawDataCopyTask {
  CoorsT src_blk_coors;
  size_t src_data_offset;
  size_t src_data_size;

  size_t dest_data_offset;

  bool copy_and_add;

  RawDataCopyTask(
      const CoorsT &src_blk_coors,
      const size_t src_data_offset,
      const size_t src_data_size,
      const bool copy_and_add = false) :
      src_blk_coors(src_blk_coors),
      src_data_offset(src_data_offset),
      src_data_size(src_data_size),
      copy_and_add(copy_and_add) {}
};
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_BLK_SPAR_DATA_TEN_RAW_DATA_OPERATION_TASKS_H */
