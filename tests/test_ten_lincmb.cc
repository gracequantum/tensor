// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-11 09:38
* 
* Description: GraceQ/tensor project. Unittests for tensor linear combination functions.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"


using namespace gqten;


struct TestLinearCombination : public testing::Test {
  std::string qn_nm = "qn";
  QN qn0 =  QN({QNNameVal(qn_nm,  0)});
  QN qnp1 = QN({QNNameVal(qn_nm,  1)});
  QN qnp2 = QN({QNNameVal(qn_nm,  2)});
  QN qnm1 = QN({QNNameVal(qn_nm, -1)});
  int d_s = 3;
  QNSector qnsct0_s =  QNSector(qn0,  d_s);
  QNSector qnsctp1_s = QNSector(qnp1, d_s);
  QNSector qnsctm1_s = QNSector(qnm1, d_s);
  Index idx_in_s =  Index({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  Index idx_out_s = Index({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DGQTensor dten_default = DGQTensor();
  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});

  ZGQTensor zten_default = ZGQTensor();
  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
};


template <typename TenElemType>
void RunTestLinearCombinationCase(
    const std::vector<TenElemType> &coefs,
    const std::vector<GQTensor<TenElemType> *> &pts,
    GQTensor<TenElemType> *res) {
  GQTensor<TenElemType> bnchmrk = *res;
  auto nt = pts.size();
  for (std::size_t i = 0; i < nt; ++i) {
    auto temp = coefs[i] * (*pts[i]);
    bnchmrk += temp;
  }

  LinearCombine(coefs, pts, res);

  EXPECT_EQ(*res, bnchmrk);
}


TEST_F(TestLinearCombination, 0TenCase) {
  RunTestLinearCombinationCase({}, {}, &dten_default);
  RunTestLinearCombinationCase({}, {}, &zten_default);
}


TEST_F(TestLinearCombination, 1TenCases) {
  DGQTensor dten_res;
  dten_res = dten_1d_s;
  dten_1d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res);
  dten_1d_s.Random(qnp1);
  RunTestLinearCombinationCase({drand()}, {&dten_1d_s}, &dten_res);
  dten_res = dten_2d_s;
  dten_2d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res);
  dten_2d_s.Random(qnm1);
  RunTestLinearCombinationCase({drand()}, {&dten_2d_s}, &dten_res);
  dten_res = dten_3d_s;
  dten_3d_s.Random(qn0);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res);
  dten_3d_s.Random(qnp2);
  RunTestLinearCombinationCase({drand()}, {&dten_3d_s}, &dten_res);

  ZGQTensor zten_res;
  zten_res = zten_1d_s;
  zten_1d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res);
  zten_1d_s.Random(qnp1);
  RunTestLinearCombinationCase({zrand()}, {&zten_1d_s}, &zten_res);
  zten_res = zten_2d_s;
  zten_2d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res);
  zten_2d_s.Random(qnm1);
  RunTestLinearCombinationCase({zrand()}, {&zten_2d_s}, &zten_res);
  zten_res = zten_3d_s;
  zten_3d_s.Random(qn0);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res);
  zten_3d_s.Random(qnp2);
  RunTestLinearCombinationCase({zrand()}, {&zten_3d_s}, &zten_res);
}


TEST_F(TestLinearCombination, 2TenCases) {
  DGQTensor dten_res;
  auto dten_1d_s2 = dten_1d_s;
  auto dten_2d_s2 = dten_2d_s;
  auto dten_3d_s2 = dten_3d_s;
  dten_res = dten_1d_s;
  dten_1d_s.Random(qn0);
  dten_1d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2}, &dten_res);
  dten_1d_s.Random(qn0);
  dten_1d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2}, &dten_res);
  dten_1d_s.Random(qnm1);
  dten_1d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_1d_s, &dten_1d_s2}, &dten_res);

  dten_res = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2}, &dten_res);
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2}, &dten_res);
  dten_2d_s.Random(qnm1);
  dten_2d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_2d_s, &dten_2d_s2}, &dten_res);

  dten_res = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2}, &dten_res);
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2}, &dten_res);
  dten_3d_s.Random(qnm1);
  dten_3d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {drand(), drand()},
      {&dten_3d_s, &dten_3d_s2}, &dten_res);

  ZGQTensor zten_res;
  auto zten_1d_s2 = zten_1d_s;
  auto zten_2d_s2 = zten_2d_s;
  auto zten_3d_s2 = zten_3d_s;
  zten_res = zten_1d_s;
  zten_1d_s.Random(qn0);
  zten_1d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2}, &zten_res);
  zten_1d_s.Random(qn0);
  zten_1d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2}, &zten_res);
  zten_1d_s.Random(qnm1);
  zten_1d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_1d_s, &zten_1d_s2}, &zten_res);

  zten_res = zten_2d_s;
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2}, &zten_res);
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2}, &zten_res);
  zten_2d_s.Random(qnm1);
  zten_2d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_2d_s, &zten_2d_s2}, &zten_res);

  zten_res = zten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2}, &zten_res);
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qnp1);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2}, &zten_res);
  zten_3d_s.Random(qnm1);
  zten_3d_s2.Random(qnp2);
  RunTestLinearCombinationCase(
      {zrand(), zrand()},
      {&zten_3d_s, &zten_3d_s2}, &zten_res);
}
