// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-28 19:44
* 
* Description: GraceQ/tensor project. Unittests for distributed tensor numerical functions.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include "mkl.h"
#include "mpi.h"


using namespace gqten;


void RunTestDistributedContraction(
    GQTensor &t,
    const std::vector<std::vector<long>> &axes_set,
    const int workers) {
  auto res = GQTEN_MPI_Contract(t, t, axes_set, MPI_COMM_WORLD, workers);
  auto res0 = Contract(t, t, axes_set);
  EXPECT_EQ(*res, *res0);
}


struct TestDistributedContraction : public testing::Test {
  long d = 5;
  Index idx_out = Index({
      QNSector(QN({QNNameVal("Sz", -2)}), d),
      QNSector(QN({QNNameVal("Sz", -1)}), d),
      QNSector(QN({QNNameVal("Sz",  0)}), d),
      QNSector(QN({QNNameVal("Sz",  1)}), d),
      QNSector(QN({QNNameVal("Sz",  2)}), d)}, OUT);
  Index idx_in = InverseIndex(idx_out);
};


TEST_F(TestDistributedContraction, 2DCase) {
  auto ten = GQTensor({idx_in, idx_out});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 0)}));
  RunTestDistributedContraction(ten, {{1}, {0}}, 1);
  RunTestDistributedContraction(ten, {{1}, {0}}, 2);
  RunTestDistributedContraction(ten, {{1}, {0}}, 3);

  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 1);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 2);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 3);

  ten.Random(QN({QNNameVal("Sz", 1)}));
  RunTestDistributedContraction(ten, {{1}, {0}}, 1);
  RunTestDistributedContraction(ten, {{1}, {0}}, 2);
  RunTestDistributedContraction(ten, {{1}, {0}}, 3);

  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 1);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 2);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 3);

  ten.Random(QN({QNNameVal("Sz", 2)}));
  RunTestDistributedContraction(ten, {{1}, {0}}, 1);
  RunTestDistributedContraction(ten, {{1}, {0}}, 2);
  RunTestDistributedContraction(ten, {{1}, {0}}, 3);

  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 1);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 2);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 3);
}


TEST_F(TestDistributedContraction, 3DCase) {
  auto ten = GQTensor({idx_in, idx_out, idx_out});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 0)}));
  RunTestDistributedContraction(ten, {{2}, {0}}, 1);
  RunTestDistributedContraction(ten, {{2}, {0}}, 2);
  RunTestDistributedContraction(ten, {{2}, {0}}, 3);

  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 1);
  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 2);
  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 3);

  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 1);
  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 2);
  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 3);

  ten.Random(QN({QNNameVal("Sz", 1)}));
  RunTestDistributedContraction(ten, {{2}, {0}}, 1);
  RunTestDistributedContraction(ten, {{2}, {0}}, 2);
  RunTestDistributedContraction(ten, {{2}, {0}}, 3);

  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 1);
  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 2);
  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 3);

  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 1);
  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 2);
  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 3);

  ten.Random(QN({QNNameVal("Sz", 2)}));
  RunTestDistributedContraction(ten, {{2}, {0}}, 1);
  RunTestDistributedContraction(ten, {{2}, {0}}, 2);
  RunTestDistributedContraction(ten, {{2}, {0}}, 3);

  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 1);
  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 2);
  RunTestDistributedContraction(ten, {{1, 2}, {0, 1}}, 3);

  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 1);
  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 2);
  RunTestDistributedContraction(ten, {{0, 1, 2}, {0, 1, 2}}, 3);
}


int main(int argc, char *argv[]) {
  int result = 0;
  testing::InitGoogleTest(&argc, argv); 
  MPI_Init(&argc, &argv);
  result = RUN_ALL_TESTS();
  for (int i = 1; i <= 3; ++i) {
    MPI_SendGemmWorkerStat(kGemmWorkerStatStop, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return result;
}
