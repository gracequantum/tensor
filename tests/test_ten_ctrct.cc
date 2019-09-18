// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-10 19:39
* 
* Description: GraceQ/tensor project. Unittests for tensor contraction functions.
*/
#include "testing_utils.h"
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <cmath>

#include "mkl.h"


using namespace gqten;


struct TestContraction : public testing::Test {
  std::string qn_nm = "qn";
  QN qn0 =  QN({QNNameVal(qn_nm,  0)});
  QN qnp1 = QN({QNNameVal(qn_nm,  1)});
  QN qnp2 = QN({QNNameVal(qn_nm,  2)});
  QN qnm1 = QN({QNNameVal(qn_nm, -1)});
  int d_s = 3;
  QNSector qnsct0_s =  QNSector(qn0,  d_s);
  QNSector qnsctp1_s = QNSector(qnp1, d_s);
  QNSector qnsctm1_s = QNSector(qnm1, d_s);
  int d_l = 10;
  QNSector qnsct0_l =  QNSector(qn0,  d_l);
  QNSector qnsctp1_l = QNSector(qnp1, d_l);
  QNSector qnsctm1_l = QNSector(qnm1, d_l);
  Index idx_in_s =  Index({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  Index idx_out_s = Index({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);
  Index idx_in_l =  Index({qnsctm1_l, qnsct0_l, qnsctp1_l}, IN);
  Index idx_out_l = Index({qnsctm1_l, qnsct0_l, qnsctp1_l}, OUT);

  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_1d_l = DGQTensor({idx_out_l});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_2d_l = DGQTensor({idx_in_l, idx_out_l});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_3d_l = DGQTensor({idx_in_l, idx_out_l, idx_out_l});
};


template <typename TenElemType>
void RunTestTenCtrct1DCase(GQTensor<TenElemType> &t, const QN &div) {
  t.Random(div);
  TenElemType res = 0;
  for (auto &blk : t.cblocks()) {
    for (long i = 0; i < blk->size; ++i) {
      res += std::norm(blk->cdata()[i]);
    }
  }
  GQTensor<TenElemType> t_res;
  auto t_dag = Dag(t);
  Contract(&t, &t_dag, {{0}, {0}}, &t_res);
  EXPECT_NEAR(t_res.scalar, res, kEpsilon);
}


TEST_F(TestContraction, 1DCase) {
  RunTestTenCtrct1DCase(dten_1d_s, qn0);
  RunTestTenCtrct1DCase(dten_1d_s, qnp1);
  RunTestTenCtrct1DCase(dten_1d_s, qnm1);
}


template <typename TenElemType>
void RunTestTenCtrct2DCase1(
    GQTensor<TenElemType> &ta, GQTensor<TenElemType> &tb) {
  auto m = ta.shape[0];
  auto n = tb.shape[1];
  auto k1 = ta.shape[1];
  auto k2 = tb.shape[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  auto dense_res = new TenElemType [m * n];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  GQTensor<TenElemType> res;
  Contract(&ta, &tb, {{1}, {0}}, &res);
  idx = 0;
  for (auto &coor : res.CoorsIter()) {
    EXPECT_NEAR(res.Elem(coor), dense_res[idx], kEpsilon);
    idx++;
  }
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename TenElemType>
void RunTestTenCtrct2DCase2(
    GQTensor<TenElemType> &ta, GQTensor<TenElemType> &tb) {
  auto m = ta.shape[0];
  auto n = tb.shape[1];
  auto k1 = ta.shape[1];
  auto k2 = tb.shape[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem({coor[1], coor[0]});
    idx++;
  }
  TenElemType res_scalar = 0.0;
  for (long i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  GQTensor<TenElemType> res;
  Contract(&ta, &tb, {{0, 1}, {1, 0}}, &res);
  EXPECT_NEAR(res.scalar, res_scalar, kEpsilon);
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
}


template <typename TenElemType>
void RunTestTenCtrct3DCase1(
    GQTensor<TenElemType> &ta,
    GQTensor<TenElemType> &tb) {
  auto m = ta.shape[0] * ta.shape[1];
  auto n = tb.shape[1] * tb.shape[2];
  auto k1 = ta.shape[2];
  auto k2 = tb.shape[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  auto dense_res = new TenElemType [m * n];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  GQTensor<TenElemType> res;
  Contract(&ta, &tb, {{2}, {0}}, &res);
  idx = 0;
  for (auto &coor : res.CoorsIter()) {
    EXPECT_NEAR(res.Elem(coor), dense_res[idx], kEpsilon);
    idx++;
  }
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename TenElemType>
void RunTestTenCtrct3DCase2(
    GQTensor<TenElemType> &ta,
    GQTensor<TenElemType> &tb) {
  auto m = ta.shape[0];
  auto n = tb.shape[2];
  auto k1 = ta.shape[1] * ta.shape[2];
  auto k2 = tb.shape[0] * tb.shape[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  auto dense_res = new TenElemType [m * n];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  GQTensor<TenElemType> res;
  Contract(&ta, &tb, {{1, 2}, {0, 1}}, &res);
  idx = 0;
  for (auto &coor : res.CoorsIter()) {
    EXPECT_NEAR(res.Elem(coor), dense_res[idx], kEpsilon);
    idx++;
  }
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename TenElemType>
void RunTestTenCtrct3DCase3(
    GQTensor<TenElemType> &ta,
    GQTensor<TenElemType> &tb) {
  auto m = ta.shape[0];
  auto n = tb.shape[2];
  auto k1 = ta.shape[1] * ta.shape[2];
  auto k2 = tb.shape[0] * tb.shape[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  TenElemType res_scalar = 0.0;
  for (long i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  GQTensor<TenElemType> res;
  Contract(&ta, &tb, {{0, 1, 2}, {0, 1, 2}}, &res);
  EXPECT_NEAR(res.scalar, res_scalar, kEpsilon);
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
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s2);
}
