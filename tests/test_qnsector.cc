/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 22:44
* 
* Description: GraceQ/tensor project. Unit tests fro QNSector class.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


using gqten::QNNameVal;


struct TestQNSector : public testing::Test {
  gqten::QNSector qnsct_default = gqten::QNSector();
  gqten::QNSector qnsct1 = gqten::QNSector(
                               gqten::QN({QNNameVal("Sz", 1)}), 1);
  gqten::QNSector qnsct2 = gqten::QNSector(
                               gqten::QN({QNNameVal("Sz", -1)}), 2);
};


TEST_F(TestQNSector, DataMembers) {
  EXPECT_EQ(qnsct_default.qn, gqten::QN());
  EXPECT_EQ(qnsct1.qn, gqten::QN({QNNameVal("Sz", 1)}));
  EXPECT_EQ(qnsct1.dim, 1);
}


TEST_F(TestQNSector, Hashable) {
  std::hash<int> int_hasher;
  EXPECT_EQ(qnsct_default.hash(), (gqten::QN().hash())^int_hasher(0));
  EXPECT_EQ(
      qnsct1.hash(),
      (gqten::QN({QNNameVal("Sz", 1)}).hash())^int_hasher(1));
  EXPECT_EQ(
      qnsct2.hash(),
      (gqten::QN({QNNameVal("Sz", -1)}).hash())^int_hasher(2));
}


TEST_F(TestQNSector, Equivalent) {
  EXPECT_TRUE(qnsct_default == qnsct_default);
  EXPECT_TRUE(qnsct1 == qnsct1);
  EXPECT_TRUE(qnsct_default != qnsct1);
  EXPECT_TRUE(qnsct1 != qnsct2);
}
