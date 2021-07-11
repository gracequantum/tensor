// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-29 09:33
*
* Description: GraceQ/tensor project. Unittests for tensor contraction.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "gqten/tensor_manipulation/basic_operations.h"     // Dag

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

struct TestContraction : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  int d_s = 3;
  QNSctT qnsct0_s =  QNSctT(qn0,  d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  int d_l = 10;
  QNSctT qnsct0_l =  QNSctT(qn0,  d_l);
  QNSctT qnsctp1_l = QNSctT(qnp1, d_l);
  QNSctT qnsctm1_l = QNSctT(qnm1, d_l);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::OUT);
  IndexT idx_in_l =  IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::IN);
  IndexT idx_out_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::OUT);

  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_1d_l = DGQTensor({idx_out_l});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_2d_l = DGQTensor({idx_in_l, idx_out_l});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_3d_s3 = DGQTensor({idx_in_s, idx_in_s, idx_out_s});
  DGQTensor dten_3d_s4 = DGQTensor({idx_out_s, idx_in_s, idx_in_s});
  DGQTensor dten_3d_l = DGQTensor({idx_in_l, idx_out_l, idx_out_l});

  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_1d_l = ZGQTensor({idx_out_l});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_2d_l = ZGQTensor({idx_in_l, idx_out_l});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_3d_s3 = ZGQTensor({idx_in_s, idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s4 = ZGQTensor({idx_out_s, idx_in_s, idx_in_s});
  ZGQTensor zten_3d_l = ZGQTensor({idx_in_l, idx_out_l, idx_out_l});
};


inline void CblasGemm(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const MKL_INT m, const MKL_INT n, const MKL_INT k,
    const gqten::GQTEN_Double alpha,
    const gqten::GQTEN_Double *a, const MKL_INT lda,
    const gqten::GQTEN_Double *b, const MKL_INT ldb,
    const gqten::GQTEN_Double beta,
    gqten::GQTEN_Double *c, const MKL_INT ldc) {
  cblas_dgemm(
      Layout,
      transa, transb,
      m, n, k,
      alpha,
      a, lda,
      b, ldb,
      beta,
      c, ldc);
}


inline void CblasGemm(
    const CBLAS_LAYOUT Layout,
    const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const MKL_INT m, const MKL_INT n, const MKL_INT k,
    const gqten::GQTEN_Complex alpha,
    const gqten::GQTEN_Complex *a, const MKL_INT lda,
    const gqten::GQTEN_Complex *b, const MKL_INT ldb,
    const gqten::GQTEN_Complex beta,
    gqten::GQTEN_Complex *c, const MKL_INT ldc) {
  cblas_zgemm(
      Layout,
      transa, transb,
      m, n, k,
      &alpha,
      a, lda,
      b, ldb,
      &beta,
      c, ldc);
}


template <typename TenElemT, typename QNT>
void RunTestTenCtrct1DCase(GQTensor<TenElemT, QNT> &t, const QNT &div) {
  t.Random(div);
  GQTensor<TenElemT, QNT> t_res;
  auto t_dag = Dag(t);
  Contract(&t, &t_dag, {{0}, {0}}, &t_res);

  TenElemT res = 0.0;
  for (auto &coors : GenAllCoors(t.GetShape())) {
    res += std::norm(t.GetElem(coors));
  }
  GtestExpectNear(t_res.GetElem({}), res, kEpsilon);

  mkl_free_buffers();
}


TEST_F(TestContraction, 1DCase) {
  RunTestTenCtrct1DCase(dten_1d_s, qn0);
  RunTestTenCtrct1DCase(dten_1d_s, qnp1);
  RunTestTenCtrct1DCase(dten_1d_s, qnm1);

  RunTestTenCtrct1DCase(zten_1d_s, qn0);
  RunTestTenCtrct1DCase(zten_1d_s, qnp1);
  RunTestTenCtrct1DCase(zten_1d_s, qnm1);
}


template <typename TenElemT, typename QNT>
void RunTestTenCtrct2DCase1(
    const GQTensor<TenElemT, QNT> &ta,
    const GQTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[1];
  auto k1 = ta.GetShape()[1];
  auto k2 = tb.GetShape()[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemT [ta_size];
  auto dense_tb =  new TenElemT [tb_size];
  auto dense_res = new TenElemT [m * n];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  GQTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{1}, {0}}, &res);
  idx = 0;
  for (auto &coors : GenAllCoors(res.GetShape())) {
    GtestExpectNear(res.GetElem(coors), dense_res[idx], kEpsilon);
    idx++;
  }

  mkl_free_buffers();
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename TenElemT, typename QNT>
void RunTestTenCtrct2DCase2(
    const GQTensor<TenElemT, QNT> &ta,
    const GQTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[1];
  auto k1 = ta.GetShape()[1];
  auto k2 = tb.GetShape()[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemT [ta_size];
  auto dense_tb =  new TenElemT [tb_size];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem({coors[1], coors[0]});
    idx++;
  }
  TenElemT res_scalar = 0.0;
  for (size_t i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  GQTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{0, 1}, {1, 0}}, &res);
  GtestExpectNear(res.GetElem({}), res_scalar, kEpsilon);

  mkl_free_buffers();
  delete [] dense_ta;
  delete [] dense_tb;
}


TEST_F(TestContraction, 2DCase) {
  auto dten_2d_s2 = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);
  dten_2d_s.Random(qnp1);
  dten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);
  dten_2d_s.Random(qnp1);
  dten_2d_s2.Random(qnm1);
  RunTestTenCtrct2DCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);

  auto zten_2d_s2 = zten_2d_s;
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
  zten_2d_s.Random(qnp1);
  zten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
  zten_2d_s.Random(qnp1);
  zten_2d_s2.Random(qnm1);
  RunTestTenCtrct2DCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
}


template <typename TenElemT, typename QNT>
void RunTestTenCtrct3DCase1(
    const GQTensor<TenElemT, QNT> &ta,
    const GQTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0] * ta.GetShape()[1];
  auto n = tb.GetShape()[1] * tb.GetShape()[2];
  auto k1 = ta.GetShape()[2];
  auto k2 = tb.GetShape()[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemT [ta_size];
  auto dense_tb =  new TenElemT [tb_size];
  auto dense_res = new TenElemT [m * n];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  GQTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{2}, {0}}, &res);
  idx = 0;
  for (auto &coors : GenAllCoors(res.GetShape())) {
    GtestExpectNear(res.GetElem(coors), dense_res[idx], kEpsilon);
    idx++;
  }

  mkl_free_buffers();
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename TenElemT, typename QNT>
void RunTestTenCtrct3DCase2(
    const GQTensor<TenElemT, QNT> &ta,
    const GQTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[2];
  auto k1 = ta.GetShape()[1] * ta.GetShape()[2];
  auto k2 = tb.GetShape()[0] * tb.GetShape()[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemT [ta_size];
  auto dense_tb =  new TenElemT [tb_size];
  auto dense_res = new TenElemT [m * n];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  GQTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{1, 2}, {0, 1}}, &res);
  idx = 0;
  for (auto &coors : GenAllCoors(res.GetShape())) {
    GtestExpectNear(res.GetElem(coors), dense_res[idx], kEpsilon);
    idx++;
  }

  mkl_free_buffers();
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename TenElemT, typename QNT>
void RunTestTenCtrct3DCase3(
    const GQTensor<TenElemT, QNT> &ta,
    const GQTensor<TenElemT, QNT> &tb
) {
  auto m = ta.GetShape()[0];
  auto n = tb.GetShape()[2];
  auto k1 = ta.GetShape()[1] * ta.GetShape()[2];
  auto k2 = tb.GetShape()[0] * tb.GetShape()[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemT [ta_size];
  auto dense_tb =  new TenElemT [tb_size];
  size_t idx = 0;
  for (auto &coors : GenAllCoors(ta.GetShape())) {
    dense_ta[idx] = ta.GetElem(coors);
    idx++;
  }
  idx = 0;
  for (auto &coors : GenAllCoors(tb.GetShape())) {
    dense_tb[idx] = tb.GetElem(coors);
    idx++;
  }
  TenElemT res_scalar = 0.0;
  for (size_t i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  GQTensor<TenElemT, QNT> res;
  Contract(&ta, &tb, {{0, 1, 2}, {0, 1, 2}}, &res);
  GtestExpectNear(res.GetElem({}), res_scalar, kEpsilon);

  mkl_free_buffers();
  delete [] dense_ta;
  delete [] dense_tb;
}


TEST_F(TestContraction, 3DCase) {
  auto dten_3d_s2 = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qn0);
  dten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s3);
  dten_3d_s.Random(qnp1);
  dten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s3);
  dten_3d_s.Random(qnp1);
  dten_3d_s3.Random(qnm1);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s3);
  dten_3d_s.Random(qn0);
  dten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s4);
  dten_3d_s.Random(qnp1);
  dten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s4);
  dten_3d_s.Random(qnp1);
  dten_3d_s4.Random(qnm1);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s4);

  auto zten_3d_s2 = zten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qn0);
  zten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s3);
  zten_3d_s.Random(qnp1);
  zten_3d_s3.Random(qn0);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s3);
  zten_3d_s.Random(qnp1);
  zten_3d_s3.Random(qnm1);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s3);
  zten_3d_s.Random(qn0);
  zten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s4);
  zten_3d_s.Random(qnp1);
  zten_3d_s4.Random(qn0);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s4);
  zten_3d_s.Random(qnp1);
  zten_3d_s4.Random(qnm1);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s4);
}
