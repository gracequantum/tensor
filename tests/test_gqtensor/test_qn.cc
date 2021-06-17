// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-24 21:32
*
* Description: GraceQ/tensor project. Unit tests for quantum number object.
*/
#include "gqten/gqtensor/qn/qn.h"         // QN
#include "gqten/gqtensor/qn/qnval_u1.h"   // U1QNVal
#include "gqten/framework/vec_hash.h"     // VecStdTypeHasher

#include "gtest/gtest.h"
#include "../testing_utility.h"           // RandInt

#include <fstream>


using namespace gqten;


using QN0 = QN<>;
using U1QN = QN<U1QNVal>;
using U1U1QN = QN<U1QNVal, U1QNVal>;


struct TestQN : public testing::Test {
  int rand_int_1 = RandInt(-10, 10);
  int rand_int_2 = RandInt(11, 20);

  QN0 qn_default = QN0();
  U1QN u1_qn_1 = U1QN({QNCard("Sz", U1QNVal(rand_int_1))});
  U1QN u1_qn_2 = U1QN({QNCard("Sz", U1QNVal(rand_int_2))});
  U1U1QN u1u1_qn = U1U1QN({
                       QNCard("Sz", U1QNVal(rand_int_1)),
                       QNCard("N", U1QNVal(rand_int_2))
                   });
};


TEST_F(TestQN, Dimension) {
  EXPECT_EQ(qn_default.dim(), 0);
  EXPECT_EQ(u1_qn_1.dim(), 1);
  EXPECT_EQ(u1_qn_2.dim(), 1);
  EXPECT_EQ(u1u1_qn.dim(), 1);
}


TEST_F(TestQN, Hashable) {
  EXPECT_EQ(qn_default.Hash(), 0);
  EXPECT_EQ(u1_qn_1.Hash(), VecStdTypeHasher(std::vector<int>{rand_int_1}));
  EXPECT_EQ(
      u1u1_qn.Hash(),
      VecStdTypeHasher(std::vector<int>{rand_int_1, rand_int_2})
  );
}


TEST_F(TestQN, Showable) {
  Show(qn_default);
  std::cout << std::endl;
  Show(u1_qn_1);
  std::cout << std::endl;
  Show(u1_qn_2, 1);
  std::cout << std::endl;
  Show(u1u1_qn, 2);
  std::cout << std::endl;
}


TEST_F(TestQN, Equivalent) {
  EXPECT_TRUE(qn_default == QN0());
  EXPECT_TRUE(u1_qn_1 == u1_qn_1);
  EXPECT_TRUE(u1u1_qn == u1u1_qn);
  EXPECT_TRUE(u1_qn_1 != u1_qn_2);
}


TEST_F(TestQN, Negtivation) {
  EXPECT_EQ(-qn_default, QN0());
  EXPECT_EQ(
      -u1_qn_1,
      U1QN({QNCard("Sz", U1QNVal(-rand_int_1))})
  );
  EXPECT_EQ(
      -u1u1_qn,
      U1U1QN({
          QNCard("Sz", U1QNVal(-rand_int_1)),
          QNCard("N", U1QNVal(-rand_int_2))
      })
  );
}


TEST_F(TestQN, Summation) {
  auto res0 = qn_default + qn_default;
  EXPECT_EQ(res0, QN0());
  auto res1 = u1_qn_1 + u1_qn_2;
  EXPECT_EQ(res1, U1QN({QNCard("Sz", U1QNVal(rand_int_1 + rand_int_2))}));
  auto res2 = u1u1_qn + u1u1_qn;
  EXPECT_EQ(
      res2,
      U1U1QN({
          QNCard("Sz", U1QNVal(2 * rand_int_1)),
          QNCard("N", U1QNVal(2 * rand_int_2))
      })
  );
}


TEST_F(TestQN, GetQNVal) {
  EXPECT_EQ(u1_qn_1.GetQNVal(0).GetVal(), rand_int_1);
  EXPECT_EQ(u1_qn_2.GetQNVal(0).GetVal(), rand_int_2);
  EXPECT_EQ(u1u1_qn.GetQNVal(0).GetVal(), rand_int_1);
  EXPECT_EQ(u1u1_qn.GetQNVal(1).GetVal(), rand_int_2);
}


template<typename QNT>
void RunTestQNFileIOCase(const QNT &qn) {
  std::string file = "test.qn";
  std::ofstream out(file, std::ofstream::binary);
  out << qn;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNT qn_cpy;
  in >> qn_cpy;
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(qn_cpy, qn);
}


TEST_F(TestQN, FileIO) {
  RunTestQNFileIOCase(qn_default);
  RunTestQNFileIOCase(u1_qn_1);
  RunTestQNFileIOCase(u1_qn_2);
  RunTestQNFileIOCase(u1u1_qn);
}
