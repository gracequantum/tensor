// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-03 21:34
* 
* Description: GraceQ/tensor project. Unittests for tensor SVD.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_svd.h"   // SVD
#include "gqten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "gqten/tensor_manipulation/basic_operations.h"     // Dag
#include "gqten/utility/utils_inl.h"
#include "gqten/framework/hp_numeric/lapack.h"

#include "gtest/gtest.h"
#include "../testing_utility.h"

#include "mkl.h"    // Included after other header file. Because GraceQ needs redefine MKL_Complex16 to gqten::GQTEN_Complex


using namespace gqten;
using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

struct TestSvd : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});

  size_t d_s = 3;
  QNSctT qnsct0_s =  QNSctT(qn0,  d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_4d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_4d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});
};


inline size_t IntDot(const size_t &size, const size_t *x, const size_t *y) {
  size_t res = 0;
  for (size_t i = 0; i < size; ++i) { res += x[i] * y[i]; }
  return res;
}


inline double ToDouble(const double d) {
  return d;
}


inline double ToDouble(const GQTEN_Complex z) {
  return z.real();
}


inline void SVDTensRestore(
    const DGQTensor *pu,
    const DGQTensor *ps,
    const DGQTensor *pvt,
    const size_t ldims,
    DGQTensor *pres) {
  DGQTensor t_restored_tmp;
  Contract(pu, ps, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}


inline void SVDTensRestore(
    const ZGQTensor *pu,
    const DGQTensor *ps,
    const ZGQTensor *pvt,
    const size_t ldims,
    ZGQTensor *pres) {
  ZGQTensor t_restored_tmp;
  auto zs = ToComplex(*ps);
  Contract(pu, &zs, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}


template <typename TenElemT, typename QNT>
void RunTestSvdCase(
    GQTensor<TenElemT, QNT> &t,
    const size_t &ldims,
    const size_t &rdims,
    const double &cutoff,
    const size_t &dmin,
    const size_t &dmax,
    const QNT *random_div = nullptr) {
  if (random_div != nullptr) {
    srand(0);
    t.Random(*random_div);
  }
  GQTensor<TenElemT, QNT> u, vt;
  GQTensor<GQTEN_Double, QNT> s;
  double trunc_err;
  size_t D;
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  SVD(
      &t,
      ldims,
      qn0,
      cutoff, dmin, dmax,
      &u, &s,&vt, &trunc_err, &D
  );

  auto ndim = ldims + rdims;
  size_t rows = 1, cols = 1;
  for (size_t i = 0; i < ndim; ++i) {
    if (i < ldims) {
      rows *= t.GetIndexes()[i].dim();
    } else {
      cols *= t.GetIndexes()[i].dim();
    }
  }
  auto dense_mat = new TenElemT [rows*cols];
  auto offsets = CalcMultiDimDataOffsets(t.GetShape());
  for (auto &coors : GenAllCoors(t.GetShape())) {
    dense_mat[IntDot(ndim, coors.data(), offsets.data())] = t.GetElem(coors);
  }
  TenElemT *dense_u;
  TenElemT *dense_vt;
  GQTEN_Double *dense_s;
  hp_numeric::MatSVD(dense_mat, rows, cols, dense_u, dense_s, dense_vt);
  size_t dense_sdim;
  if (rows > cols) {
    dense_sdim = cols;
  } else {
    dense_sdim = rows;
  }

  std::vector<double> dense_svs;
  for (size_t i = 0; i < dense_sdim; ++i) {
    if (dense_s[i] > 1.0E-13) {
      dense_svs.push_back(dense_s[i]);
    }
  }
  std::sort(dense_svs.begin(), dense_svs.end());
  auto endit = dense_svs.cend();
  auto begit =  endit - dmax;
  if (dmax > dense_svs.size()) { begit = dense_svs.cbegin(); }
  auto saved_dense_svs = std::vector<double>(begit, endit);
  std::vector<double> qn_svs;
  for (size_t i = 0; i < s.GetShape()[0]; i++) {
    qn_svs.push_back(ToDouble(s.GetElem({i, i})));
  }
  std::sort(qn_svs.begin(), qn_svs.end());
  EXPECT_EQ(qn_svs.size(), saved_dense_svs.size());
  for (size_t i = 0; i < qn_svs.size(); ++i) {
    EXPECT_NEAR(qn_svs[i], saved_dense_svs[i], kEpsilon);
  }

  double total_square_sum = 0.0;
  for (auto &sv : dense_svs) {
    total_square_sum += sv * sv;
  }
  double saved_square_sum = 0.0;
  for (auto &ssv : saved_dense_svs) {
    saved_square_sum += ssv * ssv;
  }
  auto dense_trunc_err = 1 - saved_square_sum / total_square_sum;
  EXPECT_NEAR(trunc_err, dense_trunc_err, kEpsilon);

  if (trunc_err < 1.0E-10) {
    GQTensor<TenElemT, QNT> t_restored;
    SVDTensRestore(&u, &s, &vt, ldims, &t_restored);
    for (auto &coors : GenAllCoors(t.GetShape())) {
      GtestExpectNear(t_restored.GetElem(coors), t.GetElem(coors), kEpsilon);
    }
  } else {
    GQTensor<TenElemT, QNT> t_restored;
    SVDTensRestore(&u, &s, &vt, ldims, &t_restored);
    auto t_diff = t + (-t_restored);
    auto t_diff_norm = t_diff.Normalize();
    auto t_norm = t.Normalize();
    auto norm_ratio = (t_diff_norm / t_norm);
    GtestExpectNear(norm_ratio * norm_ratio, trunc_err, 1E-02);
  }

  delete [] dense_mat;
  free(dense_s);
  free(dense_u);
  free(dense_vt);
}


TEST_F(TestSvd, 2DCase) {
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s - 1,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s + 1,
      &qn0);

  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s - 1,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s + 1,
      &qnp1);

  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm2);
}


TEST_F(TestSvd, 3DCase) {
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s - 1,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s + 1,
      &qn0);

  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s + 1,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s - 1,
      &qnp1);

  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s + 1,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s - 1,
      &qn0);

  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qnp1);

  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qnp1);
}


TEST_F(TestSvd, 4DCase) {
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) + 1,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) - 1,
      &qn0);

  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) + 1,
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) - 1,
      &qnp1);

  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qnp1);

  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qnp1);
}
