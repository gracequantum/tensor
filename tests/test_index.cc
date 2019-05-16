/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-27 14:31
* 
* Description: GraceQ/tensor project. Unit tests for Index object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <cstdio>


using namespace gqten;


struct TestIndex : public testing::Test {
  Index idx_default = Index();
  Index idx_1sct = Index({QNSector(QN({QNNameVal("Sz", 0)}), 1)});
  Index idx_1sct_in = Index({QNSector(QN({QNNameVal("Sz", 0)}), 1)}, IN);
  std::vector<QNSector> qnscts2 = {
      QNSector(QN({QNNameVal("Sz", 1)}), 1),
      QNSector(QN({QNNameVal("Sz", 1)}), 2)};
  Index idx_2sct_out = Index(
      {QNSector(QN({QNNameVal("Sz", 1)}), 1),
       QNSector(QN({QNNameVal("Sz", 1)}), 2)}, OUT);
};


TEST_F(TestIndex, IndexDirection) {
  EXPECT_EQ(idx_default.dir, NDIR);
  EXPECT_EQ(idx_1sct.dir, NDIR);
  EXPECT_EQ(idx_1sct_in.dir, IN);
  EXPECT_EQ(idx_2sct_out.dir, OUT);
}


TEST_F(TestIndex, Dimension) {
  EXPECT_EQ(idx_default.dim, 0);
  EXPECT_EQ(idx_1sct.dim, 1);
  EXPECT_EQ(idx_2sct_out.dim, 3);
  EXPECT_EQ(idx_default.CalcDim(), 0);
  EXPECT_EQ(idx_1sct.CalcDim(), 1);
  EXPECT_EQ(idx_2sct_out.CalcDim(), 3);
}


TEST_F(TestIndex, Tag) {
  EXPECT_EQ(idx_default.tag, "");
  idx_default.tag = "default";
  EXPECT_EQ(idx_default.tag, "default");
}


TEST_F(TestIndex, Hashable) {
  EXPECT_EQ(idx_default.Hash(), idx_default.Hash());
  EXPECT_TRUE(idx_default == idx_default);
  idx_default.tag = "default";
  EXPECT_NE(idx_default.Hash(), gqten::Index().Hash());
  EXPECT_FALSE(idx_default == gqten::Index());
}


TEST_F(TestIndex, InterOffsetAndQnsct) {
  auto res = idx_1sct.CoorOffsetAndQnsct(0);
  EXPECT_EQ(res.inter_offset, 0);
  EXPECT_EQ(res.qnsct, QNSector(QN({QNNameVal("Sz", 0)}), 1));
  res = idx_2sct_out.CoorOffsetAndQnsct(0);
  EXPECT_EQ(res.inter_offset, 0);
  EXPECT_EQ(res.qnsct, QNSector(QN({QNNameVal("Sz", 1)}), 1));
  res = idx_2sct_out.CoorOffsetAndQnsct(1);
  EXPECT_EQ(res.inter_offset, 1);
  EXPECT_EQ(res.qnsct, QNSector(QN({QNNameVal("Sz", 1)}), 2));
  res = idx_2sct_out.CoorOffsetAndQnsct(2);
  EXPECT_EQ(res.inter_offset, 1);
  EXPECT_EQ(res.qnsct, QNSector(QN({QNNameVal("Sz", 1)}), 2));
}


void RunTestIndexFileIOCase(const Index &idx) {
  std::string file = "test.idx";
  std::ofstream out(file, std::ofstream::binary);
  bfwrite(out, idx);
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  Index idx_cpy;
  bfread(in, idx_cpy);
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(idx_cpy, idx);
}

TEST_F(TestIndex, FileIO) {
  RunTestIndexFileIOCase(idx_default);
  RunTestIndexFileIOCase(idx_1sct);
  auto idx_1sct_with_tag = idx_1sct;
  idx_1sct_with_tag.tag = "test";
  RunTestIndexFileIOCase(idx_1sct_with_tag);
  RunTestIndexFileIOCase(idx_1sct_in);
  RunTestIndexFileIOCase(idx_2sct_out);
}
