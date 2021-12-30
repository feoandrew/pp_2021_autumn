// Copyright 2021 Feoktistov Andrew
#include <gtest/gtest.h>
#include <vector>
#include "../../modules/task_3/feoktistov_a_optimization/ops_mpi.h"
#include <gtest-mpi-listener.hpp>

TEST(Parallel_Operations_MPI, Rectangle) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double Xleft = -5.0;
  double Xright = 5.0;
  double Yleft = -5.0;
  double Yright = 5.0;
  double presision = 0.01;

  double rez = ParallelOptimization(presision, Xleft, Xright, Yleft, Yright);
  if (rank == 0) {
    double seq_rez =
        SeqentalOptimization(presision, Xleft, Xright, Yleft, Yright, size);
    ASSERT_NEAR(rez, seq_rez, 2);
  }
}
TEST(Parallel_Operations_MPI, Error) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double Xleft = -5.5;
  double Xright = 3.2;
  double Yleft = -4.9;
  double Yright = 3.8;
  double presision = 0.01;

  if (rank == 0) {
    ASSERT_ANY_THROW(
        SeqentalOptimization(presision, Xright, Xleft, Yleft, Yright, size));
  }
}
TEST(Parallel_Operations_MPI, Acceleration) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double Xleft = -5.5;
  double Xright = 3.2;
  double Yleft = -4.9;
  double Yright = 3.7;
  double presision = 0.01;
  double beginTime;
  double endtime;
  double t1;

  if (rank == 0) {
    beginTime = MPI_Wtick();
  }
  ParallelOptimization(presision, Xleft, Xright, Yleft, Yright);
  if (rank == 0) {
    endtime = MPI_Wtick();
    t1 = beginTime - endtime;
    beginTime = MPI_Wtick();
    SeqentalOptimization(presision, Xleft, Xright, Yleft, Yright, size);
    endtime = MPI_Wtick();
    ASSERT_LE(t1, beginTime - endtime);
  }
}
TEST(Parallel_Operations_MPI, ShortYtest) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double Xleft = -5.5;
  double Xright = 3.2;
  double Yleft = -0.5;
  double Yright = 0.8;
  double presision = 0.001;

  double rez = ParallelOptimization(presision, Xleft, Xright, Yleft, Yright);
  if (rank == 0) {
    double seq_rez =
        SeqentalOptimization(presision, Xleft, Xright, Yleft, Yright, size);
    ASSERT_NEAR(rez, seq_rez, 2);
  }
}
TEST(Parallel_Operations_MPI, ShortXtest) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  double Xleft = -0.5;
  double Xright = 0.2;
  double Yleft = -10.5;
  double Yright = 10.8;
  double presision = 0.001;

  double rez = ParallelOptimization(presision, Xleft, Xright, Yleft, Yright);
  if (rank == 0) {
     double seq_rez =
        SeqentalOptimization(presision, Xleft, Xright, Yleft, Yright, size);
    ASSERT_NEAR(rez, seq_rez, 2);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
