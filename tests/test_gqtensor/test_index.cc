// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-02 17:10
*
* Description: GraceQ/tensor project. Unit tests for Index class.
*/
#include "gqten/gqtensor/index.h"           // Index
#include "gqten/gqtensor/qn/qn.h"           // QN
#include "gqten/gqtensor/qn/qnval_u1.h"     // U1QNVal
#include "gqten/gqtensor/qnsct.h"           // QNSectorVec

#include "gtest/gtest.h"
#include "../testing_utility.h"     // RandInt, RandUnsignedInt

#include <fstream>      // ifstream, ofstream


using namespace gqten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;


struct TestIndex : public testing::Test {
  int qnval_1 = RandInt(-10, 10);
  int qnval_2 = RandInt(-10, 10);
  size_t dgnc_1 = RandUnsignedInt(1, 10);
  size_t dgnc_2 = RandUnsignedInt(1, 10);
  IndexT idx_default = IndexT();
  IndexT idx_1sct_in = IndexT(
                           {
                               QNSctT(
                                   U1QN({QNCard("Sz", U1QNVal(qnval_1))}),
                                   dgnc_1
                               )
                           },
                           GQTenIndexDirType::IN
                       );
  QNSctVecT qnscts2 = {QNSctT(U1QN({QNCard("Sz", U1QNVal(qnval_1))}), dgnc_1),
                       QNSctT(U1QN({QNCard("Sz", U1QNVal(qnval_2))}), dgnc_2)};
  IndexT idx_2sct_out = IndexT(qnscts2, GQTenIndexDirType::OUT);
  IndexT idx_2sct_in = IndexT(qnscts2, GQTenIndexDirType::IN);
};


TEST_F(TestIndex, IndexDirection) {
  EXPECT_EQ(idx_default.GetDir(), GQTenIndexDirType::NDIR);
  EXPECT_EQ(idx_1sct_in.GetDir(), GQTenIndexDirType::IN);
  EXPECT_EQ(idx_2sct_out.GetDir(), GQTenIndexDirType::OUT);
}


TEST_F(TestIndex, Dimension) {
  EXPECT_EQ(idx_default.dim(), 0);
  EXPECT_EQ(idx_1sct_in.dim(), kDimOfU1Repr * dgnc_1);
  EXPECT_EQ(idx_2sct_out.dim(), kDimOfU1Repr * (dgnc_1 + dgnc_2));
}


TEST_F(TestIndex, Hashable) {
  EXPECT_TRUE(idx_default.Hash() == idx_default.Hash());
  EXPECT_TRUE(idx_default.Hash() != idx_1sct_in.Hash());
  EXPECT_TRUE(idx_1sct_in.Hash() == idx_1sct_in.Hash());
  EXPECT_TRUE(idx_2sct_out.Hash() != idx_1sct_in.Hash());

  std::hash<int> int_hasher;
  EXPECT_TRUE(idx_2sct_out.Hash() == (VecHasher(qnscts2) ^ int_hasher(1)));
}


TEST_F(TestIndex, Inversion) {
  EXPECT_EQ(InverseIndex(idx_default), idx_default);
  EXPECT_EQ(InverseIndex(idx_2sct_in), idx_2sct_out);
}


template <typename IndexT>
void RunTestIndexFileIOCase(const IndexT &idx) {
  std::string file = "test.idx";
  std::ofstream out(file, std::ofstream::binary);
  out << idx;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  IndexT idx_cpy;
  in >> idx_cpy;
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(idx_cpy, idx);
}


TEST_F(TestIndex, FileIO) {
  RunTestIndexFileIOCase(idx_default);
  RunTestIndexFileIOCase(idx_1sct_in);
  RunTestIndexFileIOCase(idx_2sct_in);
  RunTestIndexFileIOCase(idx_2sct_out);
}
