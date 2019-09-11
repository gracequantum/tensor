// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 13:39
* 
* Description: GraceQ/tensor project. Unittests for GQTensor object.
*/
#include "testing_utils.h"
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <utility>
#include <cmath>
#include <cstdio>


using namespace gqten;


const double kEpsilon = 1.0E-12;


typedef GQTensor<double> DGQTensor;
typedef QNBlock<double> DQNBlock;


struct TestGQTensor : public testing::Test {
  QNSector szup = QNSector(QN({QNNameVal("Sz", 1)}), 1);
  QNSector szdn = QNSector(QN({QNNameVal("Sz", -1)}), 1);
  Index bpb = Index({szdn, szup}, IN, "b");
  Index tpb = Index({szdn, szup}, OUT, "t");
  DGQTensor vec = DGQTensor({tpb});
  DGQTensor site = DGQTensor({bpb, tpb});
  int d = 3;
  Index idx_in = Index({
                     QNSector(QN({QNNameVal("Sz", -1)}), d),
                     QNSector(QN({QNNameVal("Sz",  0)}), d),
                     QNSector(QN({QNNameVal("Sz",  1)}), d)}, IN);
  Index idx_out = Index({
                      QNSector(QN({QNNameVal("Sz", -1)}), d),
                      QNSector(QN({QNNameVal("Sz",  0)}), d),
                      QNSector(QN({QNNameVal("Sz",  1)}), d)}, OUT);
  DGQTensor vec_rand_up = DGQTensor({tpb});

  void SetUp(void) {
    vec_rand_up.Random(QN({QNNameVal("Sz", 1)}));
  }
};


TEST_F(TestGQTensor, Initialization) {
  // Test default constructor.
  DGQTensor gqten_default = DGQTensor(); 
  EXPECT_EQ(gqten_default.indexes, std::vector<Index>());

  // Test common constructor.
  EXPECT_EQ(vec.indexes, std::vector<Index>{tpb});
  EXPECT_EQ(site.indexes, (std::vector<Index>{bpb, tpb}));

  // Test copy constructor.
  DGQTensor vec_rand_up_cpy(vec_rand_up);
  EXPECT_EQ(vec_rand_up_cpy.indexes, std::vector<Index>{tpb});
  EXPECT_NE(vec_rand_up_cpy.cblocks()[0], vec_rand_up.cblocks()[0]);
  EXPECT_EQ(
      vec_rand_up_cpy.cblocks()[0]->cdata()[0],
      vec_rand_up.cblocks()[0]->cdata()[0]);

  auto vec_rand_up_cpy2 = vec_rand_up;
  EXPECT_EQ(vec_rand_up_cpy2.indexes, std::vector<Index>{tpb});
  EXPECT_NE(vec_rand_up_cpy2.cblocks()[0], vec_rand_up.cblocks()[0]);
  EXPECT_EQ(
      vec_rand_up_cpy2.cblocks()[0]->cdata()[0],
      vec_rand_up.cblocks()[0]->cdata()[0]);

  // Test move constructor.
  DGQTensor vec_rand_up_tomove(vec_rand_up);
  DGQTensor vec_rand_up_moved(std::move(vec_rand_up_tomove));
  EXPECT_EQ(vec_rand_up_moved.indexes, std::vector<Index>{tpb});
  EXPECT_EQ(
      vec_rand_up_moved.cblocks()[0]->cdata()[0],
      vec_rand_up.cblocks()[0]->cdata()[0]);
  EXPECT_EQ(vec_rand_up_tomove.cblocks(), std::vector<DQNBlock *>{});

  DGQTensor vec_rand_up_tomove2(vec_rand_up);
  auto vec_rand_up_moved2 = std::move(vec_rand_up_tomove2);
  EXPECT_EQ(vec_rand_up_moved2.indexes, std::vector<Index>{tpb});
  EXPECT_EQ(
      vec_rand_up_moved2.cblocks()[0]->cdata()[0],
      vec_rand_up.cblocks()[0]->cdata()[0]);
  EXPECT_EQ(vec_rand_up_tomove2.cblocks(), std::vector<DQNBlock *>{});
}


TEST_F(TestGQTensor, ElementSetGet) {
  vec({0}) = 1;
  EXPECT_EQ(vec.Elem({0}), 1);
  EXPECT_EQ(vec.Elem({1}), 0);

  site({0, 0}) = 1;
  EXPECT_EQ(site.Elem({0, 0}), 1);
  EXPECT_EQ(site.Elem({1, 0}), 0);
  EXPECT_EQ(site.Elem({0, 1}), 0);
  EXPECT_EQ(site.Elem({1, 1}), 0);

  site({1, 1}) = 1;
  EXPECT_EQ(site.Elem({0, 0}), 1);
  EXPECT_EQ(site.Elem({1, 0}), 0);
  EXPECT_EQ(site.Elem({0, 1}), 0);
  EXPECT_EQ(site.Elem({1, 1}), 1);
}


TEST_F(TestGQTensor, Random) {
  // Small case.
  srand(0);
  vec.Random(QN({QNNameVal("Sz", 1)}));
  srand(0);
  EXPECT_EQ(vec.Elem({0}), 0);
  EXPECT_EQ(vec.Elem({1}), double(rand())/RAND_MAX);

  // Large case.
  auto lidx_in = Index({
                     QNSector(QN({QNNameVal("Sz", -1)}), 10),
                     QNSector(QN({QNNameVal("Sz",  0)}), 10),
                     QNSector(QN({QNNameVal("Sz",  1)}), 10)}, IN);
  auto lidx_out = InverseIndex(lidx_in);
  auto lt2 = DGQTensor({lidx_in, lidx_out});
  srand(0);
  lt2.Random(QN({QNNameVal("Sz", 1)}));
  srand(0);
  std::vector<QNSector> qnscts = {
      QNSector(QN({QNNameVal("Sz", -1)}), 10),
      QNSector(QN({QNNameVal("Sz",  0)}), 10)};
  EXPECT_EQ(lt2.cblocks()[0]->qnscts, qnscts);
  for (long i = 0; i < lt2.cblocks()[0]->size; i++) {
    EXPECT_EQ(lt2.cblocks()[0]->cdata()[i], double(rand())/RAND_MAX);
  }
  qnscts = {
      QNSector(QN({QNNameVal("Sz", 0)}), 10),
      QNSector(QN({QNNameVal("Sz", 1)}), 10)};
  EXPECT_EQ(lt2.cblocks()[1]->qnscts, qnscts);
  for (long i = 0; i < lt2.cblocks()[1]->size; i++) {
    EXPECT_EQ(lt2.cblocks()[1]->cdata()[i], double(rand())/RAND_MAX);
  }
}


void RunTestTransposeCase(
    const DGQTensor &t, const std::vector<long> &axes) {
  auto transed_t = t;
  transed_t.Transpose(axes);
  for (size_t i = 0; i < axes.size(); ++i) {
    EXPECT_EQ(transed_t.indexes[i], t.indexes[axes[i]]);
    EXPECT_EQ(transed_t.shape[i], t.shape[axes[i]]);
  }
  for (auto &coors : t.CoorsIter()) {
    EXPECT_EQ(transed_t.Elem(TransCoors(coors, axes)), t.Elem(coors));
  }
}


TEST_F(TestGQTensor, TestTranspose) {
  // 2D case.
  auto ten = DGQTensor({idx_in, idx_out});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 0)}));
  RunTestTransposeCase(ten, {0, 1});
  RunTestTransposeCase(ten, {1, 0});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 1)}));
  RunTestTransposeCase(ten, {1, 0});
  // Diff indexes size.
  auto ten2 = DGQTensor({bpb, idx_out});
  srand(0);
  ten2.Random(QN({QNNameVal("Sz", 0)}));
  RunTestTransposeCase(ten2, {0, 1});
  RunTestTransposeCase(ten2, {1, 0});

  // 3D case.
  ten = DGQTensor({idx_in, idx_out, idx_out});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 0)}));
  RunTestTransposeCase(ten, {1, 0, 2});
  // Diff indexes size.
  ten2 = DGQTensor({bpb, idx_out, tpb});
  srand(0);
  ten2.Random(QN({QNNameVal("Sz", 0)}));
  RunTestTransposeCase(ten2, {1, 0, 2});
}


void RunTestNormalizeCase(DGQTensor &t, const QN &div) {
  srand(0);
  t.Random(div);

  t.Normalize();

  auto norm = 0.0;
  for (auto &blk : t.cblocks()) {
    for (long i = 0; i < blk->size; ++i) {
      norm += std::pow(blk->cdata()[i], 2.0);
    }
  }
  EXPECT_NEAR(norm, 1.0, kEpsilon);
}


TEST_F(TestGQTensor, TestNormalize) {
  // 2D case.
  auto ten = DGQTensor({idx_in, idx_out});
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 0)}));
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 1)}));

  // 3D case.
  ten = DGQTensor({idx_in, idx_out, idx_out});
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 0)}));
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 1)}));
}


TEST_F(TestGQTensor, TestDag) {
  auto ten = DGQTensor({idx_in, idx_out});
  auto ten_dag = Dag(ten);
  EXPECT_EQ(ten_dag.indexes[0], idx_out);
  EXPECT_EQ(ten_dag.indexes[1], idx_in);
}


TEST_F(TestGQTensor, TestDiv) {
  auto ten = DGQTensor({idx_in, idx_out});

  auto div = QN({QNNameVal("Sz", 0)});
  ten.Random(div);
  EXPECT_EQ(Div(ten), div);

  div = QN({QNNameVal("Sz", 1)});
  ten.Random(div);
  EXPECT_EQ(Div(ten), div);
}


TEST_F(TestGQTensor, TestSummation) {
  auto ten1 = DGQTensor({idx_in, idx_out});
  srand(0);
  ten1.Random(QN({QNNameVal("Sz", 0)}));
  auto ten2 = DGQTensor({idx_in, idx_out});
  ten2.Random(QN({QNNameVal("Sz", 1)}));
  auto sum1 = ten1 + ten2;
  auto sum2 = DGQTensor(ten1);
  sum2 += ten2;
  for (auto &coors : sum1.CoorsIter()) {
    EXPECT_NEAR(sum1.Elem(coors), sum2.Elem(coors), kEpsilon);
  }
}


TEST_F(TestGQTensor, TestDotMultiplication) {
  auto ten = DGQTensor();
  ten.scalar = 1.0;
  auto multed_ten = 2.33 * ten;
  EXPECT_DOUBLE_EQ(multed_ten.scalar, 2.33);

  ten = DGQTensor({idx_in,  idx_out});
  ten.Random(QN({QNNameVal("Sz", 0)}));
  multed_ten = 2.33  * ten;
  for (size_t i = 0; i < ten.cblocks().size(); ++i) {
    for (long j = 0; j < ten.cblocks()[i]->size; j++) {
        EXPECT_DOUBLE_EQ(
          multed_ten.cblocks()[i]->cdata()[j],
          2.33 * ten.cblocks()[i]->cdata()[j]);
    }
  }
}


TEST_F(TestGQTensor, TestEq) {
  auto ten1 = DGQTensor({idx_in,  idx_out});
  ten1.Random(QN({QNNameVal("Sz", 0)}));
  EXPECT_TRUE(ten1 == ten1);

  auto ten2 = DGQTensor({idx_in, idx_out});
  ten2.Random(QN({QNNameVal("Sz", 1)}));
  EXPECT_TRUE(ten1 != ten2);
}


void RunTestGQTensorFileIOCase(const DGQTensor &t) {
  std::string file = "test.gqten";
  std::ofstream out(file, std::ofstream::binary);
  bfwrite(out, t);
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  DGQTensor t_cpy;
  bfread(in, t_cpy);
  in.close();
  EXPECT_EQ(t_cpy, t);
}


TEST_F(TestGQTensor, FileIO) {
  // Default case.
  RunTestGQTensorFileIOCase(DGQTensor());

  // Small case.
  RunTestGQTensorFileIOCase(vec_rand_up);

  // Large case.
  auto lidx_in = Index({
                     QNSector(QN({QNNameVal("Sz", -1)}), 10),
                     QNSector(QN({QNNameVal("Sz",  0)}), 10),
                     QNSector(QN({QNNameVal("Sz",  1)}), 10)}, IN);
  auto lidx_out = InverseIndex(lidx_in);
  auto lt2 = DGQTensor({lidx_in, lidx_out});
  srand(0);
  lt2.Random(QN({QNNameVal("Sz", 1)}));
  RunTestGQTensorFileIOCase(lt2);
}
