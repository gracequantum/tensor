// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 10:18
* 
* Description: GraceQ/tensor project. Unittests for QNBlock object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <cstdio>


using namespace gqten;


struct TestQNBlock : public testing::Test {
  QNBlock qnblock_default; 

  QNSector sz0_sct = QNSector(QN({QNNameVal("Sz", 0)}), 1);
  QNSector sz1_sct2 = QNSector(QN({QNNameVal("Sz", 1)}), 2);
  QNBlock qnblock_sz0sct_1d = QNBlock({sz0_sct});
  QNBlock qnblock_sz0sct_2d = QNBlock({sz0_sct, sz0_sct});
  QNBlock qnblock_sz1sct2_2d = QNBlock({sz1_sct2, sz1_sct2});

  std::vector<const QNSector *> psz0_sct_vec = {&sz0_sct};
  QNBlock qnblock_sz0sct_1d_from_ptr = QNBlock(psz0_sct_vec);
};


TEST_F(TestQNBlock, TestNdim) {
  EXPECT_EQ(qnblock_default.ndim, 0);
  EXPECT_EQ(qnblock_sz0sct_1d.ndim, 1);
  EXPECT_EQ(qnblock_sz0sct_1d_from_ptr.ndim, 1);
  EXPECT_EQ(qnblock_sz0sct_2d.ndim, 2);
}


TEST_F(TestQNBlock, TestShape) {
  std::vector<long> lvec0; 
  EXPECT_EQ(qnblock_default.shape, lvec0);
  std::vector<long> lvec1 = {1};
  EXPECT_EQ(qnblock_sz0sct_1d.shape, lvec1);
  EXPECT_EQ(qnblock_sz0sct_1d_from_ptr.shape, lvec1);
  std::vector<long> lvec11 = {1, 1};
  EXPECT_EQ(qnblock_sz0sct_2d.shape, lvec11);
  std::vector<long> lvec22 = {2, 2};
  EXPECT_EQ(qnblock_sz1sct2_2d.shape, lvec22);
}


TEST_F(TestQNBlock, TestElemAssignment) {
  qnblock_sz0sct_1d({0}) = 1;
  EXPECT_EQ(qnblock_sz0sct_1d({0}), 1);

  qnblock_sz0sct_1d_from_ptr({0}) = 1;
  EXPECT_EQ(qnblock_sz0sct_1d_from_ptr({0}), 1);

  qnblock_sz0sct_2d({0, 0}) = 1;
  EXPECT_EQ(qnblock_sz0sct_2d({0, 0}), 1);

  qnblock_sz1sct2_2d({0, 0}) = 1.;
  EXPECT_EQ(qnblock_sz1sct2_2d({0, 0}), 1.);
  EXPECT_EQ(qnblock_sz1sct2_2d({1, 0}), 0.);
  EXPECT_EQ(qnblock_sz1sct2_2d({0, 1}), 0.);
  EXPECT_EQ(qnblock_sz1sct2_2d({1, 1}), 0.);

  qnblock_sz1sct2_2d({1, 0}) = 2.;
  qnblock_sz1sct2_2d({1, 1}) = 3.;
  qnblock_sz1sct2_2d({0, 1}) = 4.;
  EXPECT_EQ(qnblock_sz1sct2_2d({0, 0}), 1.);
  EXPECT_EQ(qnblock_sz1sct2_2d({1, 0}), 2.);
  EXPECT_EQ(qnblock_sz1sct2_2d({1, 1}), 3.);
  EXPECT_EQ(qnblock_sz1sct2_2d({0, 1}), 4.);
}


TEST_F(TestQNBlock, TestPartialHash) {
  EXPECT_EQ(
      qnblock_sz0sct_1d.PartHash({0}),
      QNSectorSet({qnblock_sz0sct_1d.qnscts[0]}).Hash());
  EXPECT_EQ(
      qnblock_sz0sct_2d.PartHash({0, 1}),
      QNSectorSet({qnblock_sz0sct_2d.qnscts[0],
                   qnblock_sz0sct_2d.qnscts[0]}).Hash());
}


TEST_F(TestQNBlock, TestQNSectorSetHash) {
  EXPECT_EQ(qnblock_default.QNSectorSetHash(), 0);
  EXPECT_EQ(
      qnblock_sz0sct_1d.QNSectorSetHash(),
      QNSectorSet(qnblock_sz0sct_1d.qnscts).Hash());
  EXPECT_EQ(
      qnblock_sz1sct2_2d.QNSectorSetHash(),
      QNSectorSet(qnblock_sz1sct2_2d.qnscts).Hash());
}


void RunTestQNBlockFileIOCase(const QNBlock &qnblk) {
  std::string file = "test.qnblk";
  std::ofstream out(file, std::ofstream::binary);
  bfwrite(out, qnblk);
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNBlock qnblk_cpy;
  bfread(in, qnblk_cpy);
  in.close();

  EXPECT_EQ(qnblk_cpy.ndim, qnblk.ndim);
  EXPECT_EQ(qnblk_cpy.size, qnblk.size);
  EXPECT_EQ(qnblk_cpy.shape, qnblk.shape);
  EXPECT_EQ(qnblk_cpy.qnscts, qnblk.qnscts);
  for (long i = 0; i < qnblk_cpy.size; i++) {
    EXPECT_DOUBLE_EQ(qnblk_cpy.DataConstRef()[i], qnblk.DataConstRef()[i]);
  }
}


TEST_F(TestQNBlock, FileIO) {
  RunTestQNBlockFileIOCase(qnblock_default);
  qnblock_sz0sct_1d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz0sct_1d);
  qnblock_sz0sct_2d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz0sct_2d);
  qnblock_sz1sct2_2d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz1sct2_2d);
}
