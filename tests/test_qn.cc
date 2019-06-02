/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-24 21:32
* 
* Description: GraceQ/tensor project. Unit tests fror quantum number object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "vec_hash.h"

#include <string>
#include <fstream>

#include <cstdio>


using namespace gqten;


struct TestQN : public testing::Test {
  QN qn_default = QN();
  QN qn_u1_1 = QN({QNNameVal("Sz", 0)});
  QN qn_u1_2 = QN({QNNameVal("Sz", 1)});
  QN qn_u1_3 = QN({QNNameVal("Sz", -1)});
  QN qn_u1_u1_1 = QN({QNNameVal("Sz", 0), QNNameVal("N", 0)});
  QN qn_u1_u1_2 = QN({QNNameVal("Sz", 1), QNNameVal("N", 1)});
};


TEST_F(TestQN, Hashable) {
  EXPECT_EQ(qn_default.Hash(), 0);
  EXPECT_EQ(qn_u1_1.Hash(), VecStdTypeHasher(std::vector<long>{0}));
  EXPECT_EQ(qn_u1_2.Hash(), VecStdTypeHasher(std::vector<long>{1}));
  EXPECT_EQ(qn_u1_3.Hash(), VecStdTypeHasher(std::vector<long>{-1}));
  EXPECT_EQ(qn_u1_u1_1.Hash(), VecStdTypeHasher(std::vector<long>{0, 0}));
  EXPECT_EQ(qn_u1_u1_2.Hash(), VecStdTypeHasher(std::vector<long>{1, 1}));
}


TEST_F(TestQN, Equivalent) {
  EXPECT_TRUE(qn_default == QN());
  EXPECT_TRUE(qn_u1_1 != qn_u1_2);
  EXPECT_TRUE(qn_u1_1 != qn_u1_u1_2);
}


TEST_F(TestQN, Negtivation) {
  EXPECT_EQ(-qn_default, QN());
  EXPECT_EQ(-qn_u1_1, QN({QNNameVal("Sz", 0)}));
  EXPECT_EQ(-qn_u1_2, QN({QNNameVal("Sz", -1)}));
}


TEST_F(TestQN, Summation) {
  auto res = qn_default + qn_default;
  EXPECT_EQ(res, QN());
  res = qn_u1_2 + qn_u1_2;
  EXPECT_EQ(res, QN({QNNameVal("Sz", 2)}));
  res = qn_u1_u1_1 + qn_u1_u1_2;
  EXPECT_EQ(res, QN({QNNameVal("Sz", 1), QNNameVal("N", 1)}));

}


TEST_F(TestQN, Subtraction) {
  auto res = qn_u1_1 - qn_u1_2;
  EXPECT_EQ(res, qn_u1_3);
}


void RunTestQNFileIOCase(const QN &qn) {
  std::string file = "test.qn";
  std::ofstream out(file, std::ofstream::binary);
  bfwrite(out, qn);
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QN qn_cpy;
  bfread(in, qn_cpy);
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(qn_cpy, qn);
}


TEST_F(TestQN, FileIO) {
  RunTestQNFileIOCase(qn_default);
  RunTestQNFileIOCase(qn_u1_1);
  RunTestQNFileIOCase(qn_u1_u1_1);
}
