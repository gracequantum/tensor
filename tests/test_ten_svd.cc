// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-06 14:11
* 
* Description: GraceQ/tensor project. Unittests for tensor SVD functions.
*/
#include <iostream>
#include <algorithm>
#include <type_traits>    // is_same

#include "gtest/gtest.h"

#include "testing_utils.h"
#include "gqten/gqten.h"
#include "gqten/detail/ten_linalg_wrapper.h"
#include "utils.h"

#include "mkl.h"    // Included after other header file. Because GraceQ needs redefine MKL_Complex16 to gqten::GQTEN_Complex .


using namespace gqten;


struct TestSvd : public testing::Test {
  std::string qn_nm = "qn";
  QN qn0 =  QN({QNNameVal(qn_nm,  0)});
  QN qnp1 = QN({QNNameVal(qn_nm,  1)});
  QN qnp2 = QN({QNNameVal(qn_nm,  2)});
  QN qnm1 = QN({QNNameVal(qn_nm, -1)});
  QN qnm2 = QN({QNNameVal(qn_nm, -2)});
  int d_s = 3;
  QNSector qnsct0_s =  QNSector(qn0,  d_s);
  QNSector qnsctp1_s = QNSector(qnp1, d_s);
  QNSector qnsctm1_s = QNSector(qnm1, d_s);
  Index idx_in_s =  Index({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  Index idx_out_s = Index({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_4d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_4d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});
};


inline long IntDot(const long &size, const long *x, const long *y) {
  long res = 0;
  for (long i = 0; i < size; ++i) { res += x[i] * y[i]; }
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
    const long ldims,
    DGQTensor *pres) {
  DGQTensor t_restored_tmp;
  Contract(pu, ps, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}


inline void SVDTensRestore(
    const ZGQTensor *pu,
    const DGQTensor *ps,
    const ZGQTensor *pvt,
    const long ldims,
    ZGQTensor *pres) {
  ZGQTensor t_restored_tmp;
  auto zs = ToComplex(*ps);
  Contract(pu, &zs, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}


template <typename TenElemType>
void RunTestSvdCase(
    GQTensor<TenElemType> &t,
    const long &ldims,
    const long &rdims,
    const double &cutoff,
    const long &dmin,
    const long &dmax,
    const QN *random_div = nullptr) {
  if (random_div != nullptr) {
    srand(0);
    t.Random(*random_div);
  }
  GQTensor<TenElemType> u, vt;
  GQTensor<GQTEN_Double> s;
  double trunc_err;
  long D;
  std::string qn_nm = "qn";
  QN qn0 = QN({QNNameVal(qn_nm,  0)});
  Svd(
      &t,
      ldims, rdims,
      qn0, qn0,
      cutoff, dmin, dmax,
      &u, &s, &vt,
      &trunc_err, &D);

  auto ndim = ldims + rdims;
  long rows = 1, cols = 1;
  for (long i = 0; i < ndim; ++i) {
    if (i < ldims) {
      rows *= t.indexes[i].CalcDim();
    } else {
      cols *= t.indexes[i].CalcDim();
    }
  }
  auto dense_mat = new TenElemType [rows*cols];
  auto offsets = CalcMultiDimDataOffsets(t.shape);
  for (auto &coors : GenAllCoors(t.shape)) {
    dense_mat[IntDot(ndim, coors.data(), offsets.data())] = t.Elem(coors);
  }
  auto dense_mat_svd_res = MatSvd(dense_mat, rows, cols);
  auto dense_s = dense_mat_svd_res.s;
  auto dense_u = dense_mat_svd_res.u;
  auto dense_vt = dense_mat_svd_res.v;
  long dense_sdim;
  if (rows > cols) {
    dense_sdim = cols;
  } else {
    dense_sdim = rows;
  }

  std::vector<double> dense_svs;
  for (long i = 0; i < dense_sdim; ++i) {
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
  for (long i = 0; i < s.shape[0]; i++) {
    qn_svs.push_back(ToDouble(s.Elem({i, i})));
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
    GQTensor<TenElemType> t_restored;
    SVDTensRestore(&u, &s, &vt, ldims, &t_restored);
    for (auto &coors : GenAllCoors(t.shape)) {
      GtestExpectNear(t_restored.Elem(coors), t.Elem(coors), kEpsilon);
    }
  }

  delete [] dense_mat;
  delete [] dense_s;
  delete [] dense_u;
  delete [] dense_vt;
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
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*2,
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
      0, 1, (d_s*3)*(d_s*3),
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3),
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
