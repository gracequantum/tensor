/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 13:39
* 
* Description: GraceQ/tensor project. Unittests for GQTensor object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <cmath>


using namespace gqten;


struct TestGQTensor : public testing::Test {
  QNSector szup = QNSector(QN({QNNameVal("Sz", 1)}), 1);
  QNSector szdn = QNSector(QN({QNNameVal("Sz", -1)}), 1);
  Index bpb = Index({szdn, szup}, IN, "b");
  Index tpb = Index({szdn, szup}, OUT, "t");
  GQTensor vec = GQTensor({tpb});
  GQTensor site = GQTensor({bpb, tpb});
  int d = 3;
  Index idx_in = Index({
                     QNSector(QN({QNNameVal("Sz", -1)}), d),
                     QNSector(QN({QNNameVal("Sz",  0)}), d),
                     QNSector(QN({QNNameVal("Sz",  1)}), d)},
                     IN);
  Index idx_out = Index({
                      QNSector(QN({QNNameVal("Sz", -1)}), d),
                      QNSector(QN({QNNameVal("Sz",  0)}), d),
                      QNSector(QN({QNNameVal("Sz",  1)}), d)},
                      OUT);
  GQTensor vec_rand_up = GQTensor({tpb});

  void SetUp(void) {
    vec_rand_up.Random(QN({QNNameVal("Sz", 1)}));
  }
};


TEST_F(TestGQTensor, Initialization) {
  // Test default constructor.
  GQTensor gqten_default = GQTensor(); 
  EXPECT_EQ(gqten_default.indexes, std::vector<Index>());
  // Test regular constructor.
  EXPECT_EQ(vec.indexes, std::vector<Index>{tpb});
  EXPECT_EQ(site.indexes, (std::vector<Index>{bpb, tpb}));
  // Test copy constructor.
  GQTensor vec_rand_up2(vec_rand_up);
  EXPECT_EQ(vec_rand_up2.indexes, std::vector<Index>{tpb});
  EXPECT_NE(vec_rand_up2.BlksConstRef()[0], vec_rand_up.BlksConstRef()[0]);
  EXPECT_EQ(
      vec_rand_up2.BlksConstRef()[0]->DataConstRef()[0],
       vec_rand_up.BlksConstRef()[0]->DataConstRef()[0]);
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
  srand(0);
  vec.Random(QN({QNNameVal("Sz", 1)}));
  srand(0);
  EXPECT_EQ(vec.Elem({0}), 0);
  EXPECT_EQ(vec.Elem({1}), double(rand())/RAND_MAX);
  // Large case.
  auto lidx_in = Index(
                   {QNSector(QN({QNNameVal("Sz", -1)}), 10),
                    QNSector(QN({QNNameVal("Sz",  0)}), 10),
                    QNSector(QN({QNNameVal("Sz",  1)}), 10)}, IN);
  auto lidx_out = InverseIndex(lidx_in);
  auto lt2 = GQTensor({lidx_in, lidx_out});
  srand(0);
  lt2.Random(QN({QNNameVal("Sz", 1)}));
  srand(0);
  std::vector<QNSector> qnscts = {
      QNSector(QN({QNNameVal("Sz", -1)}), 10),
      QNSector(QN({QNNameVal("Sz",  0)}), 10)};
  EXPECT_EQ(lt2.BlksConstRef()[0]->qnscts, qnscts);
  for (long i = 0; i < lt2.BlksConstRef()[0]->size; i++) {
    EXPECT_EQ(
        lt2.BlksConstRef()[0]->DataConstRef()[i],
        double(rand())/RAND_MAX);
  }
  qnscts = {
      QNSector(QN({QNNameVal("Sz", 0)}), 10),
      QNSector(QN({QNNameVal("Sz", 1)}), 10)
  };
  EXPECT_EQ(lt2.BlksConstRef()[1]->qnscts, qnscts);
  for (long i = 0; i < lt2.BlksConstRef()[1]->size; i++) {
    EXPECT_EQ(
        lt2.BlksConstRef()[1]->DataConstRef()[i],
        double(rand())/RAND_MAX);
  }
}


void RunTestTransposeCase(
    const GQTensor &t, const std::vector<long> &axes) {
  auto transed_t = t;
  transed_t.Transpose(axes);
  for (auto &coors : t.CoorsIter()) {
    EXPECT_EQ(transed_t.Elem(TransCoors(coors, axes)), t.Elem(coors));
  }
}


TEST_F(TestGQTensor, TestTranspose) {
  // 2D case.
  auto ten = GQTensor({idx_in, idx_out});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 0)}));
  RunTestTransposeCase(ten, {0, 1});
  RunTestTransposeCase(ten, {1, 0});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 1)}));
  RunTestTransposeCase(ten, {1, 0});
  // 3D case.
  ten = GQTensor({idx_in, idx_out, idx_out});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 0)}));
  RunTestTransposeCase(ten, {1, 0, 2});
}


void RunTestNormalizeCase(GQTensor &t, const QN &div) {
  srand(0);
  t.Random(div);
  t.Normalize();
  auto norm = 0.0;
  for (auto &blk : t.BlksConstRef()) {
    for (long i = 0; i < blk->size; ++i) {
      norm += std::pow(blk->DataConstRef()[i], 2.0);
    }
  }
  EXPECT_DOUBLE_EQ(norm, 1.0);
}


TEST_F(TestGQTensor, TestNormalize) {
  // 2D case.
  auto ten = GQTensor({idx_in, idx_out});
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 0)}));
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 1)}));
  // 3D case.
  ten = GQTensor({idx_in, idx_out, idx_out});
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 0)}));
  RunTestNormalizeCase(ten, QN({QNNameVal("Sz", 1)}));
}


TEST_F(TestGQTensor, TestDag) {
  // 2D case
  auto ten = GQTensor({idx_in, idx_out});
  auto ten_dag = Dag(ten);
  EXPECT_EQ(ten_dag.indexes[0], idx_out);
  EXPECT_EQ(ten_dag.indexes[1], idx_in);
}


TEST_F(TestGQTensor, TestSubtraction) {
  auto ten = GQTensor({idx_in, idx_out});
  ten.Random(QN({QNNameVal("Sz", 0)}));
  auto zero_t = ten - ten;
  for (auto &coors : zero_t.CoorsIter()) {
    std::cout << zero_t.Elem(coors) << std::endl;
    EXPECT_DOUBLE_EQ(zero_t.Elem(coors), 0.0);
  }
}
