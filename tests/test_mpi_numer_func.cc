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


TEST(TestDistributedContraction, 2DCase) {
  long d = 30;
  Index idx = Index({
      QNSector(QN({QNNameVal("Sz", -2)}), d),
      QNSector(QN({QNNameVal("Sz", -1)}), d),
      QNSector(QN({QNNameVal("Sz",  0)}), d),
      QNSector(QN({QNNameVal("Sz",  1)}), d),
      QNSector(QN({QNNameVal("Sz",  2)}), d)});
  auto ten = GQTensor({idx, idx});
  srand(0);
  ten.Random(QN({QNNameVal("Sz", 0)}));
  RunTestDistributedContraction(ten, {{1}, {0}}, 1);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 1);

  ten.Random(QN({QNNameVal("Sz", 2)}));
  RunTestDistributedContraction(ten, {{1}, {0}}, 1);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 1);

  ten.Random(QN({QNNameVal("Sz", 3)}));
  RunTestDistributedContraction(ten, {{1}, {0}}, 1);
  RunTestDistributedContraction(ten, {{0, 1}, {1, 0}}, 1);
}


int main(int argc, char *argv[]) {
  int result = 0;
  testing::InitGoogleTest(&argc, argv); 
  MPI_Init(&argc, &argv);
  result = RUN_ALL_TESTS();
  MPI_SendGemmWorkerStat(kGemmWorkerStatStop, 1, MPI_COMM_WORLD);
  MPI_Finalize();
  return result;
}
