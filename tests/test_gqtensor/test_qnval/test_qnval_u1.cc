// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-30 11:13
*
* Description: GraceQ/tensor project. Unit test for U(1) quantum number value.
*/
#include "gqten/gqtensor/qn/qnval_u1.h"     // U1QNVal

#include "gtest/gtest.h"
#include "../../testing_utility.h"    // RandInt


using namespace gqten;


struct TestU1QNVal : public testing::Test {
  int rand_int = RandInt(-10, 10);
  U1QNVal u1_qnval_0  = U1QNVal();
  U1QNVal u1_qnval  = U1QNVal(rand_int);
  U1QNVal minus_u1_qnval = U1QNVal(-rand_int);
};


TEST_F(TestU1QNVal, dimension) {
  EXPECT_EQ(u1_qnval_0.dim(), 1);
  EXPECT_EQ(u1_qnval.dim(), 1);
}


TEST_F(TestU1QNVal, GetVal) {
  EXPECT_EQ(u1_qnval_0.GetVal(), 0);
  EXPECT_EQ(u1_qnval.GetVal(), rand_int);
  EXPECT_EQ(minus_u1_qnval.GetVal(), -rand_int);
}


TEST_F(TestU1QNVal, Hashable) {
  std::hash<long> Hasher;
  EXPECT_EQ(Hash(u1_qnval_0), Hasher(0));
  EXPECT_EQ(Hash(u1_qnval), Hasher(rand_int));
}


TEST_F(TestU1QNVal, Showable) {
  Show(u1_qnval_0);
  Show(u1_qnval, 1);
  Show(minus_u1_qnval, 2);
}


TEST_F(TestU1QNVal, Clone) {
  auto pu1_qnval_0_clone = u1_qnval_0.Clone();
  EXPECT_TRUE(*pu1_qnval_0_clone == u1_qnval_0);
  delete pu1_qnval_0_clone;

  auto pu1_qnval_clone = u1_qnval.Clone();
  EXPECT_TRUE(*pu1_qnval_clone == u1_qnval);
  delete pu1_qnval_clone;
}


TEST_F(TestU1QNVal, Arithmetic) {
  auto pminus_u1_qnval_0 = u1_qnval_0.Minus();
  EXPECT_TRUE(*pminus_u1_qnval_0 == u1_qnval_0);
  delete pminus_u1_qnval_0;

  auto pminus_u1_qnval = u1_qnval.Minus();
  EXPECT_TRUE(*pminus_u1_qnval == minus_u1_qnval);
  delete pminus_u1_qnval;

  u1_qnval_0.AddAssign(&u1_qnval_0);
  EXPECT_TRUE(u1_qnval_0 == U1QNVal(0));

  minus_u1_qnval.AddAssign(&u1_qnval);
  EXPECT_TRUE(minus_u1_qnval == u1_qnval_0);
}
