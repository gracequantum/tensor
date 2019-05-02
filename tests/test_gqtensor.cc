/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 13:39
* 
* Description: GraceQ/tensor project. Unittests for GQTensor object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"


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
};


TEST_F(TestGQTensor, Initialization) {
  GQTensor gqten_default = GQTensor(); 
  EXPECT_EQ(gqten_default.indexes, std::vector<Index>());
  EXPECT_EQ(vec.indexes, std::vector<Index>{tpb});
  EXPECT_EQ(site.indexes, (std::vector<Index>{bpb, tpb}));
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
