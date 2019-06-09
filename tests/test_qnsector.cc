// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 22:44
* 
* Description: GraceQ/tensor project. Unit tests for QNSector class.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


using namespace gqten;


struct TestQNSector : public testing::Test {
  QNSector qnsct_default = QNSector();
  QNSector qnsct1 = QNSector(QN({QNNameVal("Sz", 1)}), 1);
  QNSector qnsct2 = QNSector(QN({QNNameVal("Sz", -1)}), 2);
};


TEST_F(TestQNSector, DataMembers) {
  EXPECT_EQ(qnsct_default.qn, QN());
  EXPECT_EQ(qnsct1.qn, QN({QNNameVal("Sz", 1)}));
  EXPECT_EQ(qnsct1.dim, 1);
}


TEST_F(TestQNSector, Hashable) {
  std::hash<int> int_hasher;
  EXPECT_EQ(qnsct_default.Hash(), (QN().Hash())^int_hasher(0));
  EXPECT_EQ(qnsct1.Hash(), (QN({QNNameVal("Sz", 1)}).Hash())^int_hasher(1));
  EXPECT_EQ(qnsct2.Hash(), (QN({QNNameVal("Sz", -1)}).Hash())^int_hasher(2));
}


TEST_F(TestQNSector, Equivalent) {
  EXPECT_TRUE(qnsct_default == qnsct_default);
  EXPECT_TRUE(qnsct1 == qnsct1);
  EXPECT_TRUE(qnsct_default != qnsct1);
  EXPECT_TRUE(qnsct1 != qnsct2);
}


void RunTestQNSectorFileIOCase(const QNSector &qnsct) {
  std::string file = "test.qnsct";
  std::ofstream out(file, std::ofstream::binary);
  bfwrite(out, qnsct);
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNSector qnsct_cpy;
  bfread(in, qnsct_cpy);
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(qnsct_cpy, qnsct);
}


TEST_F(TestQNSector, FileIO) {
  RunTestQNSectorFileIOCase(qnsct_default);
  RunTestQNSectorFileIOCase(qnsct1);
  RunTestQNSectorFileIOCase(qnsct2);
}
