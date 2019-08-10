// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 10:18
* 
* Description: GraceQ/tensor project. Unittests for QNBlock object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <utility>
#include <cstdio>


using namespace gqten;


struct TestQNBlock : public testing::Test {
  QNBlock qnblock_default; 
  QNSector sz0_sct1 = QNSector(QN({QNNameVal("Sz", 0)}), 1);
  QNSector sz1_sct2 = QNSector(QN({QNNameVal("Sz", 1)}), 2);
  QNBlock qnblock_sz0sct1_1d = QNBlock({sz0_sct1});
  QNBlock qnblock_sz0sct1_2d = QNBlock({sz0_sct1, sz0_sct1});
  QNBlock qnblock_sz1sct2_2d = QNBlock({sz1_sct2, sz1_sct2});
};


TEST_F(TestQNBlock, TestNdim) {
  EXPECT_EQ(qnblock_default.ndim, 0);
  EXPECT_EQ(qnblock_sz0sct1_1d.ndim, 1);
  EXPECT_EQ(qnblock_sz0sct1_2d.ndim, 2);
}


TEST_F(TestQNBlock, TestShape) {
  std::vector<long> lvec0; 
  EXPECT_EQ(qnblock_default.shape, lvec0);
  std::vector<long> lvec1 = {1};
  EXPECT_EQ(qnblock_sz0sct1_1d.shape, lvec1);
  std::vector<long> lvec11 = {1, 1};
  EXPECT_EQ(qnblock_sz0sct1_2d.shape, lvec11);
  std::vector<long> lvec22 = {2, 2};
  EXPECT_EQ(qnblock_sz1sct2_2d.shape, lvec22);
}


TEST_F(TestQNBlock, TestElemAssignment) {
  qnblock_sz0sct1_1d({0}) = 1;
  EXPECT_EQ(qnblock_sz0sct1_1d({0}), 1);

  qnblock_sz0sct1_2d({0, 0}) = 1;
  EXPECT_EQ(qnblock_sz0sct1_2d({0, 0}), 1);

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


TEST_F(TestQNBlock, TestCopyConstructors) {
  qnblock_sz0sct1_1d({0}) = 1;
  QNBlock qnblock_sz0sct1_1d_copyed(qnblock_sz0sct1_1d);
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed.qnscts, qnblock_sz0sct1_1d.qnscts);
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed.ndim, qnblock_sz0sct1_1d.ndim);
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed.shape, qnblock_sz0sct1_1d.shape);
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed.size, qnblock_sz0sct1_1d.size);
  for (long i = 0; i < qnblock_sz0sct1_1d.size; ++i) {
    EXPECT_DOUBLE_EQ(
        qnblock_sz0sct1_1d_copyed.cdata()[i],
        qnblock_sz0sct1_1d.cdata()[i]);
  }

  auto qnblock_sz0sct1_1d_copyed2 = qnblock_sz0sct1_1d;
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed2.qnscts, qnblock_sz0sct1_1d.qnscts);
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed2.ndim, qnblock_sz0sct1_1d.ndim);
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed2.shape, qnblock_sz0sct1_1d.shape);
  EXPECT_EQ(qnblock_sz0sct1_1d_copyed2.size, qnblock_sz0sct1_1d.size);
  for (long i = 0; i < qnblock_sz0sct1_1d.size; ++i) {
    EXPECT_DOUBLE_EQ(
        qnblock_sz0sct1_1d_copyed2.cdata()[i],
        qnblock_sz0sct1_1d.cdata()[i]);
  }
}


TEST_F(TestQNBlock, TestMoveConstructors) {
  qnblock_sz0sct1_1d({0}) = 1;
  auto qnblock_sz0sct1_1d_tomove = qnblock_sz0sct1_1d;
  QNBlock qnblock_sz0sct1_1d_moved(std::move(qnblock_sz0sct1_1d_tomove));
  EXPECT_EQ(qnblock_sz0sct1_1d_moved.qnscts, qnblock_sz0sct1_1d_tomove.qnscts);
  EXPECT_EQ(qnblock_sz0sct1_1d_moved.ndim, qnblock_sz0sct1_1d_tomove.ndim);
  EXPECT_EQ(qnblock_sz0sct1_1d_moved.shape, qnblock_sz0sct1_1d_tomove.shape);
  EXPECT_EQ(qnblock_sz0sct1_1d_moved.size, qnblock_sz0sct1_1d_tomove.size);
  for (long i = 0; i < qnblock_sz0sct1_1d.size; ++i) {
    EXPECT_DOUBLE_EQ(
        qnblock_sz0sct1_1d_moved.cdata()[i],
        qnblock_sz0sct1_1d.cdata()[i]);
  }
  EXPECT_EQ(qnblock_sz0sct1_1d_tomove.cdata(), nullptr);

  auto qnblock_sz0sct1_1d_tomove2 = qnblock_sz0sct1_1d;
  auto qnblock_sz0sct1_1d_moved2 = std::move(qnblock_sz0sct1_1d_tomove2);
  EXPECT_EQ(
      qnblock_sz0sct1_1d_moved2.qnscts,
      qnblock_sz0sct1_1d_tomove2.qnscts);
  EXPECT_EQ(qnblock_sz0sct1_1d_moved2.ndim, qnblock_sz0sct1_1d_tomove2.ndim);
  EXPECT_EQ(qnblock_sz0sct1_1d_moved2.shape, qnblock_sz0sct1_1d_tomove2.shape);
  EXPECT_EQ(qnblock_sz0sct1_1d_moved2.size, qnblock_sz0sct1_1d_tomove2.size);
  for (long i = 0; i < qnblock_sz0sct1_1d.size; ++i) {
    EXPECT_DOUBLE_EQ(
        qnblock_sz0sct1_1d_moved2.cdata()[i],
        qnblock_sz0sct1_1d.cdata()[i]);
  }
  EXPECT_EQ(qnblock_sz0sct1_1d_tomove2.cdata(), nullptr);
}


TEST_F(TestQNBlock, TestPartialHash) {
  EXPECT_EQ(
      qnblock_sz0sct1_1d.PartHash({0}),
      QNSectorSet({qnblock_sz0sct1_1d.qnscts[0]}).Hash());
  EXPECT_EQ(
      qnblock_sz0sct1_2d.PartHash({0, 1}),
      QNSectorSet({qnblock_sz0sct1_2d.qnscts[0],
                   qnblock_sz0sct1_2d.qnscts[1]}).Hash());
}


TEST_F(TestQNBlock, TestQNSectorSetHash) {
  EXPECT_EQ(qnblock_default.QNSectorSetHash(), 0);
  EXPECT_EQ(
      qnblock_sz0sct1_1d.QNSectorSetHash(),
      QNSectorSet(qnblock_sz0sct1_1d.qnscts).Hash());
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
    EXPECT_DOUBLE_EQ(qnblk_cpy.cdata()[i], qnblk.cdata()[i]);
  }
}


TEST_F(TestQNBlock, FileIO) {
  RunTestQNBlockFileIOCase(qnblock_default);
  qnblock_sz0sct1_1d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz0sct1_1d);
  qnblock_sz0sct1_2d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz0sct1_2d);
  qnblock_sz1sct2_2d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz1sct2_2d);
}
