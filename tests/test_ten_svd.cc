// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-06 14:11
* 
* Description: GraceQ/tensor project. Unittests for tensor SVD functions.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "utils.h"

#include <iostream>
#include <algorithm>

#include "mkl.h"


using namespace gqten;


const double kEpsilon = 1.0E-12;


// Test SVD decomposition.
QN qn0 = QN({QNNameVal("Sz", 0)});
QN qnn2 = QN({QNNameVal("Sz", -2)});
QN qn2 = QN({QNNameVal("Sz", 2)});
QN qnn1 = QN({QNNameVal("Sz", -1)});
QN qn1 = QN({QNNameVal("Sz", 1)});


inline long IntDot(const long &size, const long *x, const long *y) {
  long res = 0;
  for (long i = 0; i < size; ++i) { res += x[i] * y[i]; }
  return res;
}


void RunTestSvdCase(
    GQTensor &t,
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
  auto svd_res = Svd(
      t,
      ldims,
      rdims,
      qn0, qn0,
      cutoff, dmin, dmax);

  auto ndim = ldims + rdims;
  long rows = 1, cols = 1;
  for (long i = 0; i < ndim; ++i) {
    if (i < ldims) {
      rows *= t.indexes[i].CalcDim();
    } else {
      cols *= t.indexes[i].CalcDim();
    }
  }
  auto dense_mat = new double [rows*cols];
  auto offsets = CalcMultiDimDataOffsets(t.shape);
  for (auto &coors : GenAllCoors(t.shape)) {
    dense_mat[IntDot(ndim, coors.data(), offsets.data())] = t.Elem(coors);
  }

  auto m = rows;
  auto n = cols;
  auto lda = n;
  long ldu, ldvt;
  double *dense_s;
  double *dense_vt;
  long dense_sdim;
  if (m >= n) {
    dense_sdim = n;
    ldu = n;
    ldvt = n;
    dense_s = new double [n];
    dense_vt = new double [ldvt*n];
  } else {
    dense_sdim = m;
    ldu = m;
    ldvt = n;
    dense_s = new double [m];
    dense_vt = new double [ldvt*m];
  }
  double *dense_u = new double [ldu*m];
  LAPACKE_dgesdd(
      LAPACK_ROW_MAJOR, 'S',
      m, n,
      dense_mat, lda,
      dense_s,
      dense_u, ldu,
      dense_vt, ldvt);

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
  for (long i = 0; i < svd_res.s->shape[0]; i++) {
    qn_svs.push_back(svd_res.s->Elem({i, i}));
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
  EXPECT_NEAR(svd_res.trunc_err, dense_trunc_err, kEpsilon);

  if (svd_res.trunc_err < 1.0E-10) {
    auto t_restored = Contract(*svd_res.u, *svd_res.s, {{ldims}, {0}});
    t_restored = Contract(*t_restored, *svd_res.v, {{ldims}, {0}});
    for (auto &coors : GenAllCoors(t.shape)) {
      EXPECT_NEAR(t_restored->Elem(coors), t.Elem(coors), kEpsilon);
    }
  }

  delete [] dense_mat;
  delete [] dense_s;
  delete [] dense_u;
  delete [] dense_vt;
}


struct TestSvd : public testing::Test {
  long d = 5;
  Index lidx_in = Index({
                     QNSector(QN({QNNameVal("Sz", -1)}), d),
                     QNSector(QN({QNNameVal("Sz",  0)}), d),
                     QNSector(QN({QNNameVal("Sz",  1)}), d)}, IN);
  Index lidx_out = InverseIndex(lidx_in);
};


TEST_F(TestSvd, 2DCase) {
  // Small
  auto sidx_in = Index({
      QNSector(QN({QNNameVal("Sz", -2)}), 1),
      QNSector(QN({QNNameVal("Sz",  0)}), 2),
      QNSector(QN({QNNameVal("Sz",  2)}), 1)}, IN);
  auto sidx_out = InverseIndex(sidx_in);
  auto st2 = GQTensor({sidx_in, sidx_out});
  st2({1, 1}) = 1;
  st2({2, 2}) = 1;

  RunTestSvdCase(
      st2,
      1, 1,
      0, 1, 2);

  RunTestSvdCase(
      st2,
      1, 1,
      0, 1, 4,
      &qn0);
  
  RunTestSvdCase(
      st2,
      1, 1,
      0, 1, 2,
      &qnn2);

  RunTestSvdCase(
      st2,
      1, 1,
      0, 1, 2,
      &qn2);

  // Large
  auto lt2 = GQTensor({lidx_in, lidx_out});

  RunTestSvdCase(
      lt2,
      1, 1,
      0, 1, d*3,
      &qn0);

  RunTestSvdCase(
      lt2,
      1, 1,
      0, 1, d,
      &qn0);

  RunTestSvdCase(
      lt2,
      1, 1,
      0, 1, d*3,
      &qnn1);
}


TEST_F(TestSvd, 3DCase) {
  // Small case
  auto smalld = 1;
  auto sidx_in = Index({
                     QNSector(QN({QNNameVal("Sz", -1)}), smalld),
                     QNSector(QN({QNNameVal("Sz",  0)}), smalld),
                     QNSector(QN({QNNameVal("Sz",  1)}), smalld)}, IN);
  auto sidx_out = InverseIndex(sidx_in);
  auto st3 = GQTensor({sidx_in, sidx_out, sidx_out});
  RunTestSvdCase(
      st3,
      1, 2,
      0, 1, smalld*3,
      &qn0);

  RunTestSvdCase(
      st3,
      1, 2,
      0, 1, smalld*3,
      &qn1);

  RunTestSvdCase(
      st3,
      2, 1,
      0, 1, smalld*3,
      &qn0);

  // Large case
  auto lt3 = GQTensor({lidx_in, lidx_out, lidx_out});
  RunTestSvdCase(
      lt3,
      1, 2,
      0, 1, d*3,
      &qn0);

  RunTestSvdCase(
      lt3,
      1, 2,
      0, 1, d*2,
      &qn0);

  RunTestSvdCase(
      lt3,
      2, 1,
      0, 1, d*3,
      &qn0);

  RunTestSvdCase(
      lt3,
      1, 2,
      0, 1, d*3,
      &qn1);
}


TEST_F(TestSvd, 4DCase) {
  auto t4 = GQTensor({lidx_in, lidx_out, lidx_out, lidx_out});
  RunTestSvdCase(
      t4,
      2, 2,
      0, 1, (d*3)*(d*3),
      &qn0);

  RunTestSvdCase(
      t4,
      2, 2,
      0, 1, d*3,
      &qn0);

  RunTestSvdCase(
      t4,
      1, 3,
      0, 1, d*3,
      &qn0);

  RunTestSvdCase(
      t4,
      2, 2,
      0, 1, (d*3)*(d*3),
      &qn1);

  RunTestSvdCase(
      t4,
      2, 2,
      0, 1, (d*3)*(d*3),
      &qn2);
}
