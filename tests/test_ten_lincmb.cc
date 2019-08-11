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


void RunTestLinearCombinationCase(
    const std::vector<double> &coefs,
    const std::vector<GQTensor *> &pts,
    GQTensor *res) {
  GQTensor bnchmrk = *res;
  auto nt = pts.size();
  for (std::size_t i = 0; i < nt; ++i) {
    auto temp = coefs[i] * (*pts[i]);
    bnchmrk += temp;
  }

  LinearCombine(coefs, pts, res);

  EXPECT_EQ(*res, bnchmrk);
}


TEST_F(TestLinearCombination, 0TenCase) {
  auto res = GQTensor({idx_in, idx_out});
  srand(0);
  res.Random(qn0);
  RunTestLinearCombinationCase({}, {}, &res);
}


TEST_F(TestLinearCombination, 1TenCases) {
  auto res = GQTensor({idx_in, idx_out});
  auto t1 = GQTensor({idx_in, idx_out});
  srand(0);
  t1.Random(qn0); 
  RunTestLinearCombinationCase({1.0}, {&t1}, &res);

  res.Random(qn0);
  RunTestLinearCombinationCase({2.0}, {&t1}, &res);
}


TEST_F(TestLinearCombination, 2TenCases) {
  auto res = GQTensor({idx_in, idx_out});
  auto t1 = GQTensor({idx_in, idx_out});
  auto t2 = GQTensor({idx_in, idx_out});
  srand(0);
  t1.Random(qn0);
  t2.Random(qn0);
  RunTestLinearCombinationCase({2.0, 3.0}, {&t1, &t2}, &res);

  res.Random(qn0);
  RunTestLinearCombinationCase({2.0, 3.0}, {&t1, &t2}, &res);

  t1.Random(qn1);
  t2.Random(qnn1);
  EXPECT_EQ(Div(res), qn0);
  RunTestLinearCombinationCase({2.0, 3.0}, {&t1, &t2}, &res);
  EXPECT_EQ(Div(res), QN());
}
