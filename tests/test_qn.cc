/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-24 21:32
* 
* Description: GraceQ/tensor project. Unit tests fror quantum number object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <string>


using gqten::QNNameVal;


struct TestQN : public testing::Test {
  gqten::QN qn_default = gqten::QN();
  gqten::QN qn_u1_1 = gqten::QN({QNNameVal("Sz", 0)});
  gqten::QN qn_u1_2 = gqten::QN({QNNameVal("Sz", 1)});
  gqten::QN qn_u1_3 = gqten::QN({QNNameVal("Sz", -1)});
  gqten::QN qn_u1_u1_1 = gqten::QN(
                             {QNNameVal("Sz", 0), QNNameVal("N", 0)});
  gqten::QN qn_u1_u1_2 = gqten::QN(
                             {QNNameVal("Sz", 1), QNNameVal("N", 1)});
};


std::hash<std::string> StrHasher;


TEST_F(TestQN, Hashable) {
  EXPECT_EQ(qn_default.Hash(), 0);
  EXPECT_EQ(qn_u1_1.Hash(), StrHasher("Sz0"));
  EXPECT_EQ(qn_u1_2.Hash(), StrHasher("Sz1"));
  EXPECT_EQ(qn_u1_3.Hash(), StrHasher("Sz-1"));
  EXPECT_EQ(qn_u1_u1_1.Hash(), StrHasher("Sz0")^StrHasher("N0"));
}


TEST_F(TestQN, Equivalent) {
  EXPECT_TRUE(qn_default == gqten::QN());
  EXPECT_TRUE(qn_u1_1 != qn_u1_2);
}


TEST_F(TestQN, Negtivation) {
  EXPECT_EQ(-qn_default, gqten::QN());
  EXPECT_EQ(-qn_u1_1, gqten::QN({QNNameVal("Sz", 0)}));
  EXPECT_EQ(-qn_u1_2, gqten::QN({QNNameVal("Sz", -1)}));
}


TEST_F(TestQN, Summation) {
  auto res = qn_default + qn_default;
  EXPECT_EQ(res, gqten::QN());
  res = qn_u1_2 + qn_u1_2;
  EXPECT_EQ(res, gqten::QN({QNNameVal("Sz", 2)}));
  res = qn_u1_u1_1 + qn_u1_u1_2;
  EXPECT_EQ(res, gqten::QN({
                     QNNameVal("Sz", 1), QNNameVal("N", 1)}));

}


TEST_F(TestQN, Subtraction) {
  auto res = qn_u1_1 - qn_u1_2;
  EXPECT_EQ(res, qn_u1_3);
}


int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
