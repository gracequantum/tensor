/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-06 14:11
* 
* Description: GraceQ/tensor project. Unittests for tensor numerical functions.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <cmath>
#include <iostream>
#include <algorithm>

#include "mkl.h"


using namespace gqten;


const double kEpsilon = 1.0E-12;


inline double drand(void) { return double(rand()) / RAND_MAX; }


struct TestContraction : public testing::Test {
  long d = 3;
  Index idx = Index({
      QNSector(QN({QNNameVal("Sz", -1)}), d),
      QNSector(QN({QNNameVal("Sz",  0)}), d),
      QNSector(QN({QNNameVal("Sz",  1)}), d)});
  long idx_size = 3 * d;
};


TEST_F(TestContraction, 1DCase) {
  auto ten = GQTensor({idx});
  srand(0);
  auto dense_ten = new double [idx_size];
  for (long i = 0; i < idx_size; ++i) {
    dense_ten[i] = drand();
    ten({i}) = dense_ten[i];
  }
  auto res = Contract(ten, ten, {{0}, {0}});
  double res0 = 0;
  for (long i = 0; i < idx_size; ++i) { res0 += std::pow(dense_ten[i], 2.0); }
  EXPECT_NEAR(res->scalar, res0, kEpsilon);
}


TEST_F(TestContraction, 2DCase) {
  auto ten = GQTensor({idx, idx});
  srand(0);
  auto dense_ten = new double [idx_size * idx_size];
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      dense_ten[i*idx_size + j] = drand();
      ten({i, j}) = dense_ten[i*idx_size + j];
    }
  }
  auto res = Contract(ten, ten, {{1}, {0}});
  auto res0 = new double [idx_size * idx_size];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      idx_size, idx_size, idx_size,
      1.0,
      dense_ten, idx_size,
      dense_ten, idx_size,
      0.0,
      res0, idx_size);
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      EXPECT_NEAR(res->Elem({i, j}), res0[i*idx_size + j], kEpsilon);
    }
  }
}


TEST_F(TestContraction, 3DCase) {
  auto ten = GQTensor({idx, idx, idx});
  srand(0);
  auto dense_ten = new double [idx_size * idx_size * idx_size];
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      for (long k = 0; k < idx_size; ++k) {
        auto rand_elem = drand();
        ten({i, j, k}) = rand_elem;
        dense_ten[i*(idx_size*idx_size) + j*idx_size + k] = rand_elem;
      }
    }
  }
  auto res = Contract(ten, ten, {{2}, {0}});
  auto res0 = new double [idx_size * idx_size * idx_size * idx_size];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      idx_size*idx_size, idx_size*idx_size, idx_size,
      1.0,
      dense_ten, idx_size,
      dense_ten, idx_size*idx_size,
      0.0,
      res0, idx_size*idx_size);
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      for (long k = 0; k < idx_size; ++k) {
        for (long l = 0; l < idx_size; ++l) {
          EXPECT_NEAR(
              res->Elem({i, j, k, l}),
              res0[(i*idx_size + j)*(idx_size*idx_size) + (k*idx_size + l)],
              kEpsilon);
        }
      }
    }
  }
  res = Contract(ten, ten, {{1, 2}, {0, 1}});
  res0 = new double [idx_size * idx_size];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      idx_size, idx_size, idx_size*idx_size,
      1.0,
      dense_ten, idx_size*idx_size,
      dense_ten, idx_size,
      0.0,
      res0, idx_size);
  for (long i = 0; i < idx_size; ++i) {
    for (long j = 0; j < idx_size; ++j) {
      EXPECT_NEAR(res->Elem({i, j}), res0[i*idx_size + j], kEpsilon);
    }
  }
  res = Contract(ten, ten, {{0, 1, 2}, {0, 1, 2}});
  res0 = new double [1];
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      1, 1, idx_size*idx_size*idx_size,
      1.0,
      dense_ten, idx_size*idx_size*idx_size,
      dense_ten, 1,
      0.0,
      res0, 1);
  EXPECT_NEAR(res->scalar, *res0, kEpsilon);
}


// Test series tensor contraction.
struct TestSeriesContraction : public testing::Test {
  long d = 3;
  Index idx_out = Index({
      QNSector(QN({QNNameVal("Sz", -1)}), d),
      QNSector(QN({QNNameVal("Sz",  0)}), d),
      QNSector(QN({QNNameVal("Sz",  1)}), d)}, OUT);
  Index idx_in;
  QN qn0 = QN({QNNameVal("Sz", 0)});
  QN qn1 = QN({QNNameVal("Sz", 1)});
  QN qnn1 = QN({QNNameVal("Sz", -1)});
  QN qn2 = QN({QNNameVal("Sz", 2)});

  void SetUp(void) {
    idx_in = InverseIndex(idx_out); 
  }
};


void RunTestSeriesContractionCase(
    const std::vector<GQTensor *> &tens,
    const std::vector<std::pair<std::vector<long>,
                                std::vector<long>>> &ctrct_axes_series) {
  auto bnchmrk = Contract(
                     *tens[0], *tens[1],
                     {ctrct_axes_series[0].first, ctrct_axes_series[0].second});
  for (std::size_t i = 1; i < ctrct_axes_series.size(); ++i) {
    auto temp = Contract(
                    *bnchmrk, *tens[i+1],
                    {ctrct_axes_series[i].first, ctrct_axes_series[i].second});
    delete bnchmrk;
    bnchmrk = temp;
  }

  auto res = SeriesContract(tens, ctrct_axes_series);
  EXPECT_EQ(*res, *bnchmrk);
}


TEST_F(TestSeriesContraction, 1DCases) {
  auto ta = new GQTensor({idx_out});
  auto tb = new GQTensor({idx_in});
  srand(0);
  ta->Random(qn0);
  tb->Random(qn1);
  std::vector<long> ta_ctrct_axes = {0};
  std::vector<long> tb_ctrct_axes = {0};
  RunTestSeriesContractionCase(
      {ta, tb},
      {std::make_pair(ta_ctrct_axes, tb_ctrct_axes)});

  srand(0);
  ta->Random(qn0);
  tb->Random(qn0);
  RunTestSeriesContractionCase(
      {ta, tb},
      {std::make_pair(ta_ctrct_axes, tb_ctrct_axes)});

  delete ta;
  delete tb;
}


TEST_F(TestSeriesContraction, 2DCases) {
  auto ta = new GQTensor({idx_in, idx_out});
  auto tb = new GQTensor({idx_in, idx_out});
  auto tc = new GQTensor({idx_in, idx_out});
  auto td = new GQTensor({idx_in, idx_out});
  srand(0);
  ta->Random(qn0);
  tb->Random(qn0);
  tc->Random(qn0);
  td->Random(qn0);
  std::vector<long> ta_ctrct_axes1 = {1};
  std::vector<long> ta_ctrct_axes2 = {0, 1};
  std::vector<long> tb_ctrct_axes1 = {0};
  std::vector<long> tb_ctrct_axes2 = {1, 0};
  RunTestSeriesContractionCase(
      {ta, tb, tc, td},
      {std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1)});

  ta->Random(qn1);
  tb->Random(qnn1);
  tc->Random(qn0);
  td->Random(qn0);
  RunTestSeriesContractionCase(
      {ta, tb, tc, td},
      {std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1)});

  ta->Random(qn0);
  tb->Random(qn0);
  tc->Random(qn0);
  td->Random(qn0);
  RunTestSeriesContractionCase(
      {ta, tb, tc, td},
      {std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes2, tb_ctrct_axes2)});

  delete ta;
  delete tb;
  delete tc;
  delete td;
}


TEST_F(TestSeriesContraction, 3DCases) {
  auto ta = new GQTensor({idx_in, idx_out, idx_out});
  auto tb = new GQTensor({idx_in, idx_out, idx_out});
  auto tc = new GQTensor({idx_in, idx_out, idx_out});
  auto td = new GQTensor({idx_in, idx_out, idx_out});
  srand(0);
  ta->Random(qn0);
  tb->Random(qn0);
  tc->Random(qn0);
  td->Random(qn0);
  std::vector<long> ta_ctrct_axes1 = {2};
  std::vector<long> tb_ctrct_axes1 = {0};
  RunTestSeriesContractionCase(
      {ta, tb, tc, td},
      {std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1),
       std::make_pair(ta_ctrct_axes1, tb_ctrct_axes1)});
}


// Test general GQTensor contraction.
struct TestGETC : public testing::Test {
  long d = 3;
  Index idx_out = Index({
      QNSector(QN({QNNameVal("Sz", -1)}), d),
      QNSector(QN({QNNameVal("Sz",  0)}), d),
      QNSector(QN({QNNameVal("Sz",  1)}), d)}, OUT);
  Index idx_in;
  QN qn0 = QN({QNNameVal("Sz", 0)});
  QN qn1 = QN({QNNameVal("Sz", 1)});
  QN qnn1 = QN({QNNameVal("Sz", -1)});
  QN qn2 = QN({QNNameVal("Sz", 2)});

  void SetUp(void) {
    idx_in = InverseIndex(idx_out); 
  }
};


void RunTestGETCCase(
    const std::vector<long> &ctrct_idxs_a,
    const std::vector<long> &ctrct_idxs_b,
    const double alpha,
    const GQTensor *ta, const GQTensor *tb,
    const double beta,
    GQTensor * &tc) {
  auto ptemp_bnchmrk = Contract(*ta, *tb, {ctrct_idxs_a, ctrct_idxs_b});
  auto bnchmrk = alpha * (*ptemp_bnchmrk); delete ptemp_bnchmrk;
  bnchmrk += beta * *tc;

  gqten_dgetc(
      ctrct_idxs_a, ctrct_idxs_b,
      alpha,
      ta, tb,
      beta,
      tc);

  EXPECT_EQ(*tc, bnchmrk);
}


TEST_F(TestGETC, 1DCases) {
  auto ta = GQTensor({idx_out});
  auto tb = GQTensor({idx_in});
  auto tc = new GQTensor;

  // No matched block pairs case.
  srand(0);
  ta.Random(qn0);
  tb.Random(qn1);
  RunTestGETCCase(
      {0}, {0},
      1.0,
      &ta, &tb,
      0.0,
      tc);
  delete tc;

  // Has matched block pairs case.
  srand(0);
  ta.Random(qn0);
  tb.Random(qn0);
  tc = new GQTensor;
  RunTestGETCCase(
      {0}, {0},
      1.0,
      &ta, &tb,
      0.0,
      tc);
  delete tc;

  // alpha not equal 1 case.
  tc = new GQTensor;
  RunTestGETCCase(
      {0}, {0},
      2.0,
      &ta, &tb,
      0.0,
      tc);
  delete tc;

  // Beta not equal to zero case.
  tc = new GQTensor();
  tc->scalar = 2.333;
  RunTestGETCCase(
      {0}, {0},
      2.0,
      &ta, &tb,
      3.0,
      tc);
}


TEST_F(TestGETC, 2DCases) {
  auto ta = GQTensor({idx_in, idx_out});
  auto tb = GQTensor({idx_in, idx_out});

  // Scalar cases.
  auto tc = new GQTensor();
  tc->scalar = 2.33;
  srand(0);
  ta.Random(qn0);
  tb.Random(qn1);
  RunTestGETCCase(
      {0, 1}, {1, 0},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  delete tc;

  tc = new GQTensor();
  tc->scalar = 2.33;
  srand(0);
  ta.Random(qn0);
  tb.Random(qn0);
  RunTestGETCCase(
      {0, 1}, {1, 0},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  delete tc;

  // Tensor cases.
  tc = new GQTensor({idx_in, idx_out});
  srand(0);
  ta.Random(qn0);
  tb.Random(qn0);
  tc->Random(qn0);
  RunTestGETCCase(
      {1}, {0},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  delete tc;

  tc = new GQTensor({idx_in, idx_out});
  srand(0);
  ta.Random(qn1);
  tb.Random(qnn1);
  tc->Random(qn0);
  RunTestGETCCase(
      {1}, {0},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  EXPECT_EQ(Div(*tc), qn0);
  delete tc;

  tc = new GQTensor({idx_in, idx_out});
  srand(0);
  ta.Random(qn1);
  tb.Random(qn1);
  tc->Random(qn2);
  RunTestGETCCase(
      {1}, {0},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  EXPECT_EQ(Div(*tc), qn2);
}


TEST_F(TestGETC, 3DCases) {
  auto ta = GQTensor({idx_in, idx_out, idx_out});
  auto tb = GQTensor({idx_in, idx_out, idx_in});

  // Scalar cases.
  auto tc = new GQTensor();
  tc->scalar = 2.33;
  srand(0);
  ta.Random(qn0);
  ta.Random(qn1);
  RunTestGETCCase(
      {0, 1, 2}, {1, 0, 2},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  delete tc;

  tc = new GQTensor();
  tc->scalar = 2.33;
  srand(0);
  ta.Random(qn0);
  ta.Random(qn0);
  RunTestGETCCase(
      {0, 1, 2}, {1, 0, 2},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  delete tc;

  // Tensor cases.
  tc = new GQTensor({idx_in, idx_out});
  srand(0);
  ta.Random(qn1);
  tb.Random(qnn1);
  tc->Random(qn0);
  RunTestGETCCase(
      {1, 2}, {0, 2},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  EXPECT_EQ(Div(*tc), qn0);
  delete tc;

  tc = new GQTensor({idx_in, idx_out, idx_in, idx_out});
  srand(0);
  ta.Random(qn1);
  tb.Random(qnn1);
  tc->Random(qn0);
  RunTestGETCCase(
      {1}, {2},
      2.0,
      &ta, &tb,
      3.0,
      tc);
  EXPECT_EQ(Div(*tc), qn0);
  delete tc;
}


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
  auto offsets = CalcDataOffsets(t.shape);
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
