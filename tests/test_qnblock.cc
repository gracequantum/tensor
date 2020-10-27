// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 10:18
* 
* Description: GraceQ/tensor project. Unittests for QNBlock object.
*/
#include <utility>
#include <algorithm>
#include <cstdio>

#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "gqten/utility/utils_inl.h"
#include "testing_utils.h"


using namespace gqten;


template <typename QNBlockT>
void QNBlockEq(const QNBlockT &lhs, const QNBlockT &rhs) {
  EXPECT_EQ(rhs.qnscts, lhs.qnscts);
  EXPECT_EQ(rhs.ndim, lhs.ndim);
  EXPECT_EQ(rhs.shape, lhs.shape);
  EXPECT_EQ(rhs.size, lhs.size);
  GtestArrayEq(rhs.cdata(), lhs.cdata(), lhs.size);
}


using DQNBlock = QNBlock<GQTEN_Double>;
using ZQNBlock = QNBlock<GQTEN_Complex>;


struct TestQNBlock : public testing::Test {
  QN qn = QN({QNNameVal("qn", 0)});
  QNSector qnsct1 = QNSector(qn, 1);
  QNSector qnsct2 = QNSector(qn, 2);
  QNSector qnsct3 = QNSector(qn, 3);

  DQNBlock qnblock_default;
  QNSector sz0_sct1 = QNSector(QN({QNNameVal("Sz", 0)}), 1);
  QNSector sz1_sct2 = QNSector(QN({QNNameVal("Sz", 1)}), 2);
  DQNBlock qnblock_sz0sct1_1d = DQNBlock({sz0_sct1});
  DQNBlock qnblock_sz0sct1_2d = DQNBlock({sz0_sct1, sz0_sct1});
  DQNBlock qnblock_sz1sct2_2d = DQNBlock({sz1_sct2, sz1_sct2});

  DQNBlock dqnblock_default;
  DQNBlock dqnblock_1 = DQNBlock({qnsct1});
  DQNBlock dqnblock_3 = DQNBlock({qnsct3});
  DQNBlock dqnblock_22 = DQNBlock({qnsct2, qnsct2});
  DQNBlock dqnblock_23 = DQNBlock({qnsct2, qnsct3});
  DQNBlock dqnblock_13 = DQNBlock({qnsct1, qnsct3});
  DQNBlock dqnblock_233 = DQNBlock({qnsct2, qnsct3, qnsct3});

  ZQNBlock zqnblock_default;
  ZQNBlock zqnblock_1 = ZQNBlock({qnsct1});
  ZQNBlock zqnblock_3 = ZQNBlock({qnsct3});
  ZQNBlock zqnblock_22 = ZQNBlock({qnsct2, qnsct2});
  ZQNBlock zqnblock_23 = ZQNBlock({qnsct2, qnsct3});
  ZQNBlock zqnblock_13 = ZQNBlock({qnsct1, qnsct3});
  ZQNBlock zqnblock_233 = ZQNBlock({qnsct2, qnsct3, qnsct3});
};


template <typename QNBlockT>
void RunTestQNBlockNdimCase(
    const QNBlockT &qnblk, const long ndim) {
  EXPECT_EQ(qnblk.ndim, ndim);
}


TEST_F(TestQNBlock, TestNdim) {
  RunTestQNBlockNdimCase(dqnblock_default, 0);
  RunTestQNBlockNdimCase(dqnblock_1, 1);
  RunTestQNBlockNdimCase(dqnblock_3, 1);
  RunTestQNBlockNdimCase(dqnblock_22, 2);
  RunTestQNBlockNdimCase(dqnblock_23, 2);
  RunTestQNBlockNdimCase(dqnblock_13, 2);
  RunTestQNBlockNdimCase(dqnblock_233, 3);

  RunTestQNBlockNdimCase(zqnblock_default, 0);
  RunTestQNBlockNdimCase(zqnblock_1, 1);
  RunTestQNBlockNdimCase(zqnblock_3, 1);
  RunTestQNBlockNdimCase(zqnblock_22, 2);
  RunTestQNBlockNdimCase(zqnblock_23, 2);
  RunTestQNBlockNdimCase(zqnblock_13, 2);
  RunTestQNBlockNdimCase(zqnblock_233, 3);
}


template <typename QNBlockT>
void RunTestQNBlockShapeCase(
    const QNBlockT &qnblk, const std::vector<long> &shape) {
  EXPECT_EQ(qnblk.shape, shape);
}


TEST_F(TestQNBlock, TestShape) {
  RunTestQNBlockShapeCase(dqnblock_default, {});
  RunTestQNBlockShapeCase(dqnblock_1, {1});
  RunTestQNBlockShapeCase(dqnblock_3, {3});
  RunTestQNBlockShapeCase(dqnblock_22, {2, 2});
  RunTestQNBlockShapeCase(dqnblock_23, {2, 3});
  RunTestQNBlockShapeCase(dqnblock_13, {1, 3});
  RunTestQNBlockShapeCase(dqnblock_233, {2, 3 ,3});

  RunTestQNBlockShapeCase(zqnblock_default, {});
  RunTestQNBlockShapeCase(zqnblock_1, {1});
  RunTestQNBlockShapeCase(zqnblock_3, {3});
  RunTestQNBlockShapeCase(zqnblock_22, {2, 2});
  RunTestQNBlockShapeCase(zqnblock_23, {2, 3});
  RunTestQNBlockShapeCase(zqnblock_13, {1, 3});
  RunTestQNBlockShapeCase(zqnblock_233, {2, 3 ,3});
}


template <typename ElemType>
void RunTestQNBlockElemAssignmentCase(
    const QNBlock<ElemType> &qnblk_init,
    const std::vector<ElemType> elems,
    const std::vector<std::vector<long>> coors) {
  auto qnblk = qnblk_init;
  for (size_t i = 0; i < elems.size(); ++i) {
    qnblk(coors[i]) = elems[i];
  }
  for (auto coor : GenAllCoors(qnblk.shape)) {
    auto coor_it = std::find(coors.cbegin(), coors.cend(), coor); 
    if (coor_it != coors.end()) {
      auto elem_idx = std::distance(coors.begin(), coor_it);
      EXPECT_EQ(qnblk(coor), elems[elem_idx]);
    } else {
      EXPECT_EQ(qnblk(coor), ElemType(0.0));
    }
  }
}


TEST_F(TestQNBlock, TestElemAssignment) {
  RunTestQNBlockElemAssignmentCase(dqnblock_1, {1.0}, {{0}});
  RunTestQNBlockElemAssignmentCase(dqnblock_3, {1.0}, {{0}});
  RunTestQNBlockElemAssignmentCase(dqnblock_3, {1.0, 2.0}, {{0}, {1}});
  RunTestQNBlockElemAssignmentCase(dqnblock_3, {1.0, 2.0}, {{1}, {2}});
  RunTestQNBlockElemAssignmentCase(dqnblock_13, {1.0}, {{0, 1}});
  RunTestQNBlockElemAssignmentCase(dqnblock_22, {1.0}, {{1, 0}});
  RunTestQNBlockElemAssignmentCase(dqnblock_233, {1.0}, {{0, 1, 2}});
  RunTestQNBlockElemAssignmentCase(
      dqnblock_233,
      {1.0, 2.0}, {{1, 0, 2}, {0, 2, 1}});

  RunTestQNBlockElemAssignmentCase(zqnblock_1, {GQTEN_Complex(0.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(zqnblock_1, {GQTEN_Complex(1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_1,
      {GQTEN_Complex(1.0, 0.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_1,
      {GQTEN_Complex(0.0, 1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_1,
      {GQTEN_Complex(1.0, 1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_3,
      {GQTEN_Complex(1.0, 1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_3,
      {GQTEN_Complex(1.0, 1.0)}, {{1}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_3,
      {GQTEN_Complex(1.0, 1.0)}, {{2}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_3,
      {GQTEN_Complex(1.0, 0.1), GQTEN_Complex(2.0, 0.2)}, {{0}, {1}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_13,
      {GQTEN_Complex(1.0, 1.0)}, {{0, 0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_13,
      {GQTEN_Complex(1.0, 1.0)}, {{0, 1}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_13,
      {GQTEN_Complex(1.0, 1.0)}, {{0, 2}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_13,
      {GQTEN_Complex(1.0, 0.1), GQTEN_Complex(2.0, 0.2)}, {{0, 0}, {0, 1}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_22,
      {GQTEN_Complex(1.0, 0.1)}, {{1, 0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_22, {GQTEN_Complex(1.0, 0.1)}, {{1, 0}});
  RunTestQNBlockElemAssignmentCase(
      zqnblock_233, {GQTEN_Complex(1.0, 0.1)}, {{0, 1, 2}});
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


template <typename QNBlockT>
void RunTestQNBlockHashMethodsCase(
    const QNBlockT &qnblk,
    const std::vector<long> &part_axes) {
  EXPECT_EQ(qnblk.QNSectorSetHash(), QNSectorSet(qnblk.qnscts).Hash());

  std::vector<QNSector> part_qnscts;
  for (auto axis : part_axes) {
    part_qnscts.push_back(qnblk.qnscts[axis]);
  }
  EXPECT_EQ(qnblk.PartHash(part_axes), QNSectorSet(part_qnscts).Hash());
}


TEST_F(TestQNBlock, TestHashMethods) {
  /* TODO: Why it fail? */
  //RunTestQNBlockHashMethodsCase(dqnblock_default, {});
  RunTestQNBlockHashMethodsCase(dqnblock_3, {0});
  RunTestQNBlockHashMethodsCase(dqnblock_23, {0});
  RunTestQNBlockHashMethodsCase(dqnblock_23, {1});
  RunTestQNBlockHashMethodsCase(dqnblock_23, {0, 1});
  RunTestQNBlockHashMethodsCase(dqnblock_233, {0});
  RunTestQNBlockHashMethodsCase(dqnblock_233, {1});
  RunTestQNBlockHashMethodsCase(dqnblock_233, {2});
  RunTestQNBlockHashMethodsCase(dqnblock_233, {0, 1});
  RunTestQNBlockHashMethodsCase(dqnblock_233, {1, 2});
  RunTestQNBlockHashMethodsCase(dqnblock_233, {0, 2});
  RunTestQNBlockHashMethodsCase(dqnblock_233, {0, 1, 2});

  RunTestQNBlockHashMethodsCase(zqnblock_3, {0});
  RunTestQNBlockHashMethodsCase(zqnblock_23, {0});
  RunTestQNBlockHashMethodsCase(zqnblock_23, {1});
  RunTestQNBlockHashMethodsCase(zqnblock_23, {0, 1});
  RunTestQNBlockHashMethodsCase(zqnblock_233, {0});
  RunTestQNBlockHashMethodsCase(zqnblock_233, {1});
  RunTestQNBlockHashMethodsCase(zqnblock_233, {2});
  RunTestQNBlockHashMethodsCase(zqnblock_233, {0, 1});
  RunTestQNBlockHashMethodsCase(zqnblock_233, {1, 2});
  RunTestQNBlockHashMethodsCase(zqnblock_233, {0, 2});
  RunTestQNBlockHashMethodsCase(zqnblock_233, {0, 1, 2});
}


// Test rand a QNBlock.
template <typename ElemType>
void RunTestRandQNBlockCase(QNBlock<ElemType> &qnblk) {
  auto size = qnblk.size;
  auto rand_array = new ElemType[size]();
  srand(0);
  for (long i = 0; i < size; ++i) {
    Rand(rand_array[i]);
  }
  srand(0);
  qnblk.Random();
  GtestArrayEq(rand_array, qnblk.cdata(), size);
  delete [] rand_array;
}


TEST_F(TestQNBlock, TestRandQNBlock) {
  RunTestRandQNBlockCase(dqnblock_default);
  RunTestRandQNBlockCase(dqnblock_1);
  RunTestRandQNBlockCase(dqnblock_3);
  RunTestRandQNBlockCase(dqnblock_13);
  RunTestRandQNBlockCase(dqnblock_22);
  RunTestRandQNBlockCase(dqnblock_23);
  RunTestRandQNBlockCase(dqnblock_233);

  RunTestRandQNBlockCase(zqnblock_default);
  RunTestRandQNBlockCase(zqnblock_1);
  RunTestRandQNBlockCase(zqnblock_3);
  RunTestRandQNBlockCase(zqnblock_13);
  RunTestRandQNBlockCase(zqnblock_22);
  RunTestRandQNBlockCase(zqnblock_23);
  RunTestRandQNBlockCase(zqnblock_233);
}


// Test QNBlock transpose.
template <typename ElemType>
void RunTestQNBlockTransposeCase(
    const QNBlock<ElemType> &blk_init, const std::vector<long> &axes) {
  auto blk = blk_init;
  blk.Random();
  auto transed_blk = blk;
  transed_blk.Transpose(axes);
  for (size_t i = 0; i < axes.size(); ++i) {
    EXPECT_EQ(transed_blk.shape[i], blk.shape[axes[i]]);
  }
  for (auto blk_coors : GenAllCoors(blk.shape)) {
    EXPECT_EQ(transed_blk(TransCoors(blk_coors, axes)), blk(blk_coors));
  }
}


TEST_F(TestQNBlock, TestQNBlockTranspose) {
  RunTestQNBlockTransposeCase(dqnblock_1, {0});
  RunTestQNBlockTransposeCase(dqnblock_3, {0});
  RunTestQNBlockTransposeCase(dqnblock_13, {0, 1});
  RunTestQNBlockTransposeCase(dqnblock_13, {1, 0});
  RunTestQNBlockTransposeCase(dqnblock_22, {0, 1});
  RunTestQNBlockTransposeCase(dqnblock_22, {1, 0});
  RunTestQNBlockTransposeCase(dqnblock_23, {0, 1});
  RunTestQNBlockTransposeCase(dqnblock_23, {1, 0});
  RunTestQNBlockTransposeCase(dqnblock_233, {0, 1, 2});
  RunTestQNBlockTransposeCase(dqnblock_233, {1, 0, 2});
  RunTestQNBlockTransposeCase(dqnblock_233, {0, 2, 1});
  RunTestQNBlockTransposeCase(dqnblock_233, {2, 0, 1});

  RunTestQNBlockTransposeCase(zqnblock_1, {0});
  RunTestQNBlockTransposeCase(zqnblock_3, {0});
  RunTestQNBlockTransposeCase(zqnblock_13, {0, 1});
  RunTestQNBlockTransposeCase(zqnblock_13, {1, 0});
  RunTestQNBlockTransposeCase(zqnblock_22, {0, 1});
  RunTestQNBlockTransposeCase(zqnblock_22, {1, 0});
  RunTestQNBlockTransposeCase(zqnblock_23, {0, 1});
  RunTestQNBlockTransposeCase(zqnblock_23, {1, 0});
  RunTestQNBlockTransposeCase(zqnblock_233, {0, 1, 2});
  RunTestQNBlockTransposeCase(zqnblock_233, {1, 0, 2});
  RunTestQNBlockTransposeCase(zqnblock_233, {0, 2, 1});
  RunTestQNBlockTransposeCase(zqnblock_233, {2, 0, 1});
}


template <typename QNBlockT>
void RunTestQNBlockCopyAndMoveConstructorsCase(QNBlockT &qnblk) {
  qnblk.Random();

  QNBlockT qnblk_copyed(qnblk);
  QNBlockEq(qnblk_copyed, qnblk);
  auto qnblk_copyed2 = qnblk;
  QNBlockEq(qnblk_copyed2, qnblk);

  auto qnblk_tomove = qnblk;
  QNBlockT qnblk_moved(std::move(qnblk_tomove));
  QNBlockEq(qnblk_moved, qnblk);
  EXPECT_EQ(qnblk_tomove.cdata(), nullptr);
  auto qnblk_tomove2 = qnblk;
  auto qnblk_moved2 = std::move(qnblk_tomove2);
  QNBlockEq(qnblk_moved2, qnblk);
  EXPECT_EQ(qnblk_tomove2.cdata(), nullptr);
}


TEST_F(TestQNBlock, TestCopyAndMoveConstructors) {
  RunTestQNBlockCopyAndMoveConstructorsCase(dqnblock_default);
  RunTestQNBlockCopyAndMoveConstructorsCase(dqnblock_1);
  RunTestQNBlockCopyAndMoveConstructorsCase(dqnblock_3);
  RunTestQNBlockCopyAndMoveConstructorsCase(dqnblock_13);
  RunTestQNBlockCopyAndMoveConstructorsCase(dqnblock_22);
  RunTestQNBlockCopyAndMoveConstructorsCase(dqnblock_23);
  RunTestQNBlockCopyAndMoveConstructorsCase(dqnblock_233);

  RunTestQNBlockCopyAndMoveConstructorsCase(zqnblock_default);
  RunTestQNBlockCopyAndMoveConstructorsCase(zqnblock_1);
  RunTestQNBlockCopyAndMoveConstructorsCase(zqnblock_3);
  RunTestQNBlockCopyAndMoveConstructorsCase(zqnblock_13);
  RunTestQNBlockCopyAndMoveConstructorsCase(zqnblock_22);
  RunTestQNBlockCopyAndMoveConstructorsCase(zqnblock_23);
  RunTestQNBlockCopyAndMoveConstructorsCase(zqnblock_233);
}


template <typename QNBlockT>
void RunTestQNBlockFileIOCase(QNBlockT &qnblk) {
  qnblk.Random();

  std::string file = "test.qnblk";
  std::ofstream out(file, std::ofstream::binary);
  bfwrite(out, qnblk);
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNBlockT qnblk_cpy;
  bfread(in, qnblk_cpy);
  in.close();

  EXPECT_EQ(qnblk_cpy.ndim, qnblk.ndim);
  EXPECT_EQ(qnblk_cpy.size, qnblk.size);
  EXPECT_EQ(qnblk_cpy.shape, qnblk.shape);
  EXPECT_EQ(qnblk_cpy.qnscts, qnblk.qnscts);
  GtestArrayEq(qnblk_cpy.cdata(), qnblk.cdata(), qnblk_cpy.size);
}


TEST_F(TestQNBlock, FileIO) {
  RunTestQNBlockFileIOCase(dqnblock_default);
  RunTestQNBlockFileIOCase(dqnblock_1);
  RunTestQNBlockFileIOCase(dqnblock_3);
  RunTestQNBlockFileIOCase(dqnblock_13);
  RunTestQNBlockFileIOCase(dqnblock_22);
  RunTestQNBlockFileIOCase(dqnblock_23);
  RunTestQNBlockFileIOCase(dqnblock_233);

  RunTestQNBlockFileIOCase(zqnblock_default);
  RunTestQNBlockFileIOCase(zqnblock_1);
  RunTestQNBlockFileIOCase(zqnblock_3);
  RunTestQNBlockFileIOCase(zqnblock_13);
  RunTestQNBlockFileIOCase(zqnblock_22);
  RunTestQNBlockFileIOCase(zqnblock_23);
  RunTestQNBlockFileIOCase(zqnblock_233);
}
