// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-01 15:28
*
* Description: GraceQ/tensor project. Unit tests for QNSector class.
*/
#include "gqten/gqtensor/qnsct.h"           // QNSector
#include "gqten/gqtensor/qn/qn.h"           // QN
#include "gqten/gqtensor/qn/qnval_u1.h"     // U1QNVal

#include "gtest/gtest.h"
#include "../testing_utility.h"           // RandInt, RandUnsignedInt

#include <functional>     // hash
#include <fstream>        // ifstream, ofstream


using namespace gqten;

using U1QN = QN<U1QNVal>;
using U1U1QN = QN<U1QNVal, U1QNVal>;

using QNSctT1 = QNSector<U1QN>;
using QNSctT2 = QNSector<U1U1QN>;


struct TestQNSector : public testing::Test {
  int rand_int_1 = RandInt(-10, 10);
  int rand_int_2 = RandInt(-10, 10);
  size_t dgnc_1 = RandUnsignedInt(1, 10);
  size_t dgnc_2 = RandUnsignedInt(1, 10);
  QNSctT1 qnsct1_default = QNSctT1();
  QNSctT1 qnsct1 = QNSctT1(U1QN({QNCard("Sz", U1QNVal(rand_int_1))}), dgnc_1);
  QNSctT2 qnsct2 = QNSctT2(
                       U1U1QN({
                           QNCard("Sz", U1QNVal(rand_int_1)),
                           QNCard("N",  U1QNVal(rand_int_2))
                       }),
                       dgnc_2
                   );
};


TEST_F(TestQNSector, BasicInfo) {
  EXPECT_EQ(qnsct1_default.GetQn(), U1QN());
  EXPECT_EQ(qnsct1.GetQn(), U1QN({QNCard("Sz", U1QNVal(rand_int_1))}));
  EXPECT_EQ(
      qnsct2.GetQn(),
      U1U1QN({
          QNCard("Sz", U1QNVal(rand_int_1)),
          QNCard("N",  U1QNVal(rand_int_2))
      })
  );
  EXPECT_EQ(qnsct1_default.GetDegeneracy(), 0);
  EXPECT_EQ(qnsct1.GetDegeneracy(), dgnc_1);
  EXPECT_EQ(qnsct2.GetDegeneracy(), dgnc_2);
  EXPECT_EQ(qnsct1_default.dim(), 0);
  EXPECT_EQ(qnsct1.dim(), 1 * dgnc_1);
  EXPECT_EQ(qnsct2.dim(), 1 * dgnc_2);
  for (size_t i = 0; i < qnsct1.dim(); ++i) {
    EXPECT_EQ(qnsct1.CoorToDataCoor(i), i);
  }
  for (size_t i = 0; i < qnsct2.dim(); ++i) {
    EXPECT_EQ(qnsct2.CoorToDataCoor(i), i);
  }
}


TEST_F(TestQNSector, Hashable) {
  std::hash<int> int_hasher;
  EXPECT_EQ(Hash(qnsct1_default), 0);
  EXPECT_EQ(
      Hash(qnsct1),
      Hash(U1QN({QNCard("Sz", U1QNVal(rand_int_1))})) ^ int_hasher(dgnc_1)
  );
  EXPECT_EQ(
      Hash(qnsct2),
      Hash(
          U1U1QN({
              QNCard("Sz", U1QNVal(rand_int_1)),
              QNCard("N",  U1QNVal(rand_int_2))
          })
      ) ^ int_hasher(dgnc_2)
  );
}


TEST_F(TestQNSector, Showable) {
  Show(qnsct1_default);
  std::cout << std::endl;
  Show(qnsct1);
  std::cout << std::endl;
  Show(qnsct1, 1);
  std::cout << std::endl;
  Show(qnsct2, 1);
  std::cout << std::endl;
  Show(qnsct2, 2);
  std::cout << std::endl;
}


TEST_F(TestQNSector, Equivalent) {
  EXPECT_TRUE(qnsct1_default == qnsct1_default);
  EXPECT_TRUE(qnsct1 == qnsct1);
  EXPECT_TRUE(qnsct2 == qnsct2);
  EXPECT_TRUE(qnsct1_default != qnsct1);
}


template <typename QNSctT>
void RunTestQNSectorFileIOCase(const QNSctT &qnsct) {
  std::string file = "test.qnsct";
  std::ofstream out(file, std::ofstream::binary);
  out << qnsct;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNSctT qnsct_cpy;
  in >> qnsct_cpy;
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(qnsct_cpy, qnsct);
}


TEST_F(TestQNSector, FileIO) {
  RunTestQNSectorFileIOCase(qnsct1_default);
  RunTestQNSectorFileIOCase(qnsct1);
  RunTestQNSectorFileIOCase(qnsct2);
}
