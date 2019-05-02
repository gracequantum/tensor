/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-26 14:54
* 
* Description: GraceQ/tensor project. Unit tests for QNSectorSet class.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <vector>


struct TestQNSectorSet : public testing::Test {
  gqten::QNSectorSet qnscts_default;
  std::vector<gqten::QNSector> qnscts1 = {
      gqten::QNSector(gqten::QN({gqten::QNNameVal("Sz", 0)}), 1)};
  gqten::QNSectorSet qnscts_1sct = gqten::QNSectorSet({qnscts1[0]});
  std::vector<gqten::QNSector> qnscts2 = {
      gqten::QNSector(gqten::QN({gqten::QNNameVal("Sz", 1)}), 1),
      gqten::QNSector(gqten::QN({gqten::QNNameVal("Sz", 1)}), 2)};
  gqten::QNSectorSet qnscts_2sct = gqten::QNSectorSet({qnscts2[0], qnscts2[1]});
};


TEST_F(TestQNSectorSet, DataMembers) {
  std::vector<gqten::QNSector> empty_qnscts;
  EXPECT_EQ(qnscts_default.qnscts, empty_qnscts);
  EXPECT_EQ(qnscts_1sct.qnscts, qnscts1);
  EXPECT_EQ(qnscts_2sct.qnscts, qnscts2);
}


TEST_F(TestQNSectorSet, Equivalent) {
  EXPECT_TRUE(qnscts_default == qnscts_default);
  EXPECT_TRUE(qnscts_1sct == qnscts_1sct);
  EXPECT_TRUE(qnscts_1sct != qnscts_2sct);
}
