// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-26 14:54
* 
* Description: GraceQ/tensor project. Unit tests for QNSectorSet class.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <vector>


using namespace gqten;


struct TestQNSectorSet : public testing::Test {
  QNSectorSet qnscts_default;

  std::vector<QNSector> qnscts1 = {
      QNSector(QN({QNNameVal("Sz", 0)}), 1)};
  QNSectorSet qnscts_1sct = QNSectorSet(qnscts1);

  std::vector<const QNSector *> pqnscts1 = {&qnscts1[0]};
  QNSectorSet qnscts_1sct_from_ptr = QNSectorSet(pqnscts1);

  std::vector<QNSector> qnscts2 = {
      QNSector(QN({QNNameVal("Sz", 0)}), 1),
      QNSector(QN({QNNameVal("Sz", 1)}), 2)};
  QNSectorSet qnscts_2sct = QNSectorSet(qnscts2);
};


TEST_F(TestQNSectorSet, DataMembers) {
  std::vector<QNSector> empty_qnscts;
  EXPECT_EQ(qnscts_default.qnscts, empty_qnscts);
  EXPECT_EQ(qnscts_1sct.qnscts, qnscts1);
  EXPECT_EQ(qnscts_1sct_from_ptr.qnscts, qnscts1);
  EXPECT_EQ(qnscts_2sct.qnscts, qnscts2);
}


TEST_F(TestQNSectorSet, Equivalent) {
  EXPECT_TRUE(qnscts_default == qnscts_default);
  EXPECT_TRUE(qnscts_1sct == qnscts_1sct);
  EXPECT_TRUE(qnscts_2sct == qnscts_2sct);
  EXPECT_TRUE(qnscts_1sct == qnscts_1sct_from_ptr);
  EXPECT_TRUE(qnscts_1sct != qnscts_2sct);
}
