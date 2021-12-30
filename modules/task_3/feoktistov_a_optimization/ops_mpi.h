// Copyright 2021 Feoktistov Andrew
#ifndef MODULES_TASK_3_FEOKTISTOV_A_OPTIMIZATION_OPS_MPI_H_
#define MODULES_TASK_3_FEOKTISTOV_A_OPTIMIZATION_OPS_MPI_H_
#define _USE_MATH_DEFINES

#include <string>
#include <vector>

std::vector<double> getPoints(double left, double right, int size);
double ParallelOptimization(double presision, double Xleft, double Xright,
                            double Yleft, double Yright);
double SeqentalOptimization(double presision, double Xleft, double Xright,
                            double Yleft, double Yright, int dividers = 3);

double RastriginFunc(double x, double y);

#endif  // MODULES_TASK_3_FEOKTISTOV_A_OPTIMIZATION_OPS_MPI_H_
