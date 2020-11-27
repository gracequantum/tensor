// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-20 09:27
* 
* Description: GraceQ/tensor project. Implementation details for tensor linear combination.
*/
#include <assert.h>

#include "gqten/fwd_dcl.h"
#include "gqten/manipulation/ten_linalg_wrapper.h"

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


template <typename TenElemType>
void LinearCombineOneTerm(
    const TenElemType,
    const GQTensor<TenElemType> *,
    GQTensor<TenElemType> *);


template <typename TenElemType>
void LinearCombine(
    const std::vector<TenElemType> &coefs,
    const std::vector<GQTensor<TenElemType> *> &ts,
    GQTensor<TenElemType> *res) {
  auto nt = ts.size();
  assert(coefs.size() == nt);
  for (std::size_t i = 0; i < nt; ++i) {
    LinearCombineOneTerm(coefs[i], ts[i], res);
  }
}


template <typename TenElemType>
void LinearCombine(
    const std::size_t size,
    const TenElemType *coefs,
    const std::vector<GQTensor<TenElemType> *> &ts,
    GQTensor<TenElemType> *res) {
  for (std::size_t i = 0; i < size; i++) {
    LinearCombineOneTerm(coefs[i], ts[i], res);
  }
}


template <typename TenElemType>
void LinearCombineOneTerm(
    const TenElemType coef,
    const GQTensor<TenElemType> *t,
    GQTensor<TenElemType> *res) {

#ifdef GQTEN_TIMING_MODE
  Timer axpy_timer("axpy");
#endif

  for (auto &blk : t->cblocks()) {
    auto has_blk = false;
    for (auto &res_blk : res->blocks()) {
      if (res_blk->QNSectorSetHash() == blk->QNSectorSetHash()) {
        auto size = res_blk->size;
        assert(size == blk->size);
        auto blk_data = blk->cdata();
        auto res_blk_data = res_blk->data();

#ifdef GQTEN_TIMING_MODE
        axpy_timer.Restart();
#endif

        CblasAxpy(size, coef, blk_data, 1, res_blk_data, 1);

#ifdef GQTEN_TIMING_MODE
        axpy_timer.PrintElapsed(8);
#endif

        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto new_blk = new QNBlock<TenElemType>(*blk);
      auto size = new_blk->size;
      auto new_blk_data = new_blk->data();
      for (long i = 0; i < size; ++i) {
        new_blk_data[i] *= coef;
      }
      res->blocks().push_back(new_blk);
    }
  }
}
} /* gqten */ 
