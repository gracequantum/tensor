// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-09 11:39
* 
* Description: GraceQ/tensor project. Implementation details about tensor linear combination.
*/
#include "gqten/gqten.h"
#include "ten_lincmb.h"

#include <assert.h>

#include "mkl.h"

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


void LinearCombine(
    const std::vector<double> &coefs,
    const std::vector<GQTensor *> &ts,
    GQTensor *res) {
  auto nt = ts.size();
  assert(coefs.size() == nt);
  for (std::size_t i = 0; i < nt; ++i) {
    LinearCombineOneTerm(coefs[i], ts[i], res);
  }
}


void LinearCombine(
    const std::size_t size,
    const double *coefs,
    const std::vector<GQTensor *> &ts,
    GQTensor *res) {
  for (std::size_t i = 0; i < size; i++) {
    LinearCombineOneTerm(coefs[i], ts[i], res);
  }
}


void LinearCombineOneTerm(const double coef, const GQTensor *t, GQTensor *res) {

#ifdef GQTEN_TIMING_MODE
  Timer daxpy_timer("daxpy");
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
        daxpy_timer.Restart();
#endif

        cblas_daxpy(size, coef, blk_data, 1, res_blk_data, 1);

#ifdef GQTEN_TIMING_MODE
        daxpy_timer.PrintElapsed(8);
#endif

        has_blk = true;
        break;
      }
    }
    if (!has_blk) {
      auto new_blk = new QNBlock(*blk);
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
