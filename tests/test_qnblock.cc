// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 10:18
* 
* Description: GraceQ/tensor project. Unittests for QNBlock object.
*/
#include <utility>
#include <cstdio>

#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "testing_utils.h"


using namespace gqten;


template <typename ElemType>
void QNBlockEq(const QNBlock<ElemType> &lhs, const QNBlock<ElemType> &rhs) {
  EXPECT_EQ(rhs.qnscts, lhs.qnscts);
  EXPECT_EQ(rhs.ndim, lhs.ndim);
  EXPECT_EQ(rhs.shape, lhs.shape);
  EXPECT_EQ(rhs.size, lhs.size);
  GtestArrayEq(rhs.cdata(), lhs.cdata(), lhs.size);
}


// Testing QNBlock<double>.
typedef QNBlock<double> DQNBlock;


struct TestDQNBlock : public testing::Test {
  DQNBlock qnblock_default;
  QNSector sz0_sct1 = QNSector(QN({QNNameVal("Sz", 0)}), 1);
  QNSector sz1_sct2 = QNSector(QN({QNNameVal("Sz", 1)}), 2);
  DQNBlock qnblock_sz0sct1_1d = DQNBlock({sz0_sct1});
  DQNBlock qnblock_sz0sct1_2d = DQNBlock({sz0_sct1, sz0_sct1});
  DQNBlock qnblock_sz1sct2_2d = DQNBlock({sz1_sct2, sz1_sct2});
};


TEST_F(TestDQNBlock, TestNdim) {
  EXPECT_EQ(qnblock_default.ndim, 0);
  EXPECT_EQ(qnblock_sz0sct1_1d.ndim, 1);
  EXPECT_EQ(qnblock_sz0sct1_2d.ndim, 2);
}


TEST_F(TestDQNBlock, TestShape) {
  std::vector<long> lvec0; 
  EXPECT_EQ(qnblock_default.shape, lvec0);
  std::vector<long> lvec1 = {1};
  EXPECT_EQ(qnblock_sz0sct1_1d.shape, lvec1);
  std::vector<long> lvec11 = {1, 1};
  EXPECT_EQ(qnblock_sz0sct1_2d.shape, lvec11);
  std::vector<long> lvec22 = {2, 2};
  EXPECT_EQ(qnblock_sz1sct2_2d.shape, lvec22);
}


TEST_F(TestDQNBlock, TestElemAssignment) {
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


TEST_F(TestDQNBlock, TestCopyConstructors) {
  qnblock_sz0sct1_1d({0}) = 1;
  DQNBlock qnblock_sz0sct1_1d_copyed(qnblock_sz0sct1_1d);
  QNBlockEq(qnblock_sz0sct1_1d, qnblock_sz0sct1_1d_copyed);

  auto qnblock_sz0sct1_1d_copyed2 = qnblock_sz0sct1_1d;
  QNBlockEq(qnblock_sz0sct1_1d, qnblock_sz0sct1_1d_copyed2);
}


TEST_F(TestDQNBlock, TestMoveConstructors) {
  qnblock_sz0sct1_1d({0}) = 1;
  auto qnblock_sz0sct1_1d_tomove = qnblock_sz0sct1_1d;
  DQNBlock qnblock_sz0sct1_1d_moved(std::move(qnblock_sz0sct1_1d_tomove));
  QNBlockEq(qnblock_sz0sct1_1d_moved, qnblock_sz0sct1_1d);
  EXPECT_EQ(qnblock_sz0sct1_1d_tomove.cdata(), nullptr);

  auto qnblock_sz0sct1_1d_tomove2 = qnblock_sz0sct1_1d;
  auto qnblock_sz0sct1_1d_moved2 = std::move(qnblock_sz0sct1_1d_tomove2);
  QNBlockEq(qnblock_sz0sct1_1d_moved2, qnblock_sz0sct1_1d);
  EXPECT_EQ(qnblock_sz0sct1_1d_tomove2.cdata(), nullptr);
}


TEST_F(TestDQNBlock, TestPartialHash) {
  EXPECT_EQ(
      qnblock_sz0sct1_1d.PartHash({0}),
      QNSectorSet({qnblock_sz0sct1_1d.qnscts[0]}).Hash());
  EXPECT_EQ(
      qnblock_sz0sct1_2d.PartHash({0, 1}),
      QNSectorSet({qnblock_sz0sct1_2d.qnscts[0],
                   qnblock_sz0sct1_2d.qnscts[1]}).Hash());
}


TEST_F(TestDQNBlock, TestQNSectorSetHash) {
  EXPECT_EQ(qnblock_default.QNSectorSetHash(), 0);
  EXPECT_EQ(
      qnblock_sz0sct1_1d.QNSectorSetHash(),
      QNSectorSet(qnblock_sz0sct1_1d.qnscts).Hash());
  EXPECT_EQ(
      qnblock_sz1sct2_2d.QNSectorSetHash(),
      QNSectorSet(qnblock_sz1sct2_2d.qnscts).Hash());
}


// Test rand a QNBlock.
template <typename ElemType>
void RunTestRandQNBlockCase(QNBlock<ElemType> &qnblk) {
  auto size = qnblk.size;
  auto rand_array = new ElemType[size]();
  srand(0);
  for (long i = 0; i < size; ++i) {
    rand_array[i] = ElemType(rand()) / RAND_MAX;
  }
  srand(0);
  qnblk.Random();
  GtestArrayEq(rand_array, qnblk.cdata(), size);
}


TEST_F(TestDQNBlock, TestRandQNBlock) {
  RunTestRandQNBlockCase(qnblock_default);
  RunTestRandQNBlockCase(qnblock_sz0sct1_1d);
  RunTestRandQNBlockCase(qnblock_sz0sct1_2d);
  RunTestRandQNBlockCase(qnblock_sz1sct2_2d);
}


// Test QNBlock transpose.
template <typename ElemType>
void RunTestQNBlockTransposeCase(
    const QNBlock<ElemType> &blk, const std::vector<long> &axes) {
  auto transed_blk = blk;
  transed_blk.Transpose(axes);
  for (size_t i = 0; i < axes.size(); ++i) {
    EXPECT_EQ(transed_blk.shape[i], transed_blk.shape[axes[i]]);
  }
  for (auto blk_coors : GenAllCoors(blk.shape)) {
    EXPECT_EQ(transed_blk(TransCoors(blk_coors, axes)), blk(blk_coors));
  }
}


TEST_F(TestDQNBlock, TestQNBlockTranspose) {
  srand(0);
  qnblock_sz0sct1_1d.Random();
  RunTestQNBlockTransposeCase(qnblock_sz0sct1_1d, {0});
  srand(0);
  qnblock_sz0sct1_2d.Random();
  RunTestQNBlockTransposeCase(qnblock_sz0sct1_2d, {0, 1});
  RunTestQNBlockTransposeCase(qnblock_sz0sct1_2d, {1, 0});
  srand(0);
  qnblock_sz1sct2_2d.Random();
  RunTestQNBlockTransposeCase(qnblock_sz1sct2_2d, {0, 1});
  RunTestQNBlockTransposeCase(qnblock_sz1sct2_2d, {1, 0});
}


// Test file I/O
void RunTestQNBlockFileIOCase(const DQNBlock &qnblk) {
  std::string file = "test.qnblk";
  std::ofstream out(file, std::ofstream::binary);
  bfwrite(out, qnblk);
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  DQNBlock qnblk_cpy;
  bfread(in, qnblk_cpy);
  in.close();

  EXPECT_EQ(qnblk_cpy.ndim, qnblk.ndim);
  EXPECT_EQ(qnblk_cpy.size, qnblk.size);
  EXPECT_EQ(qnblk_cpy.shape, qnblk.shape);
  EXPECT_EQ(qnblk_cpy.qnscts, qnblk.qnscts);
  GtestArrayEq(qnblk_cpy.cdata(), qnblk.cdata(), qnblk_cpy.size);
}


TEST_F(TestDQNBlock, FileIO) {
  RunTestQNBlockFileIOCase(qnblock_default);
  qnblock_sz0sct1_1d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz0sct1_1d);
  qnblock_sz0sct1_2d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz0sct1_2d);
  qnblock_sz1sct2_2d.Random();
  RunTestQNBlockFileIOCase(qnblock_sz1sct2_2d);
}
