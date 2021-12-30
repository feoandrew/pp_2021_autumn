// Copyright 2021 Feoktistov Andrew
#define _USE_MATH_DEFINES
#include "../../modules/task_3/feoktistov_a_optimization/ops_mpi.h"
#include <math.h>
#include <mpi.h>
#include <cfloat>
#include <iostream>
#include <set>
#include <string>
#include <vector>
struct interval {
  double begin;
  double end;
  double value;
};
struct custom_compare final {
  bool operator()(const interval& left, const interval& right) const {
    int nLeft = left.value;
    int nRight = right.value;
    return nLeft < nRight;
  }
};
std::vector<double> getPoints(double left, double right, int size) {
  if (left > right) throw("invalid interval");
  double step = abs(right - left) / size;
  std::vector<double> vec;
  for (int i = 1; i <= size; i++) {
    vec.push_back(left + step * i);
  }

  return vec;
}
double RastriginFunc(double x, double y) {
  double sum = 20;

  sum += pow(x, 2) - 10 * cos(2 * M_PI * x);
  sum += pow(y, 2) - 10 * cos(2 * M_PI * y);

  return sum;
}

double SeqentalOptimization(double presision, double Xleft, double Xright,
                            double Yleft, double Yright, int divide) {
  int divcount;
  if (divide < 3) {
    divcount = 4;
  } else {
    divcount = divide;
  }

  std::vector<double> points;
  std::vector<double> results(divcount);
  std::vector<interval> pointvector;
  std::multiset<interval, custom_compare> intervals;
  std::multiset<interval>::iterator iter;
  std::multiset<interval>::iterator iter2;
  double step;

  double global_min = 0;

  points = getPoints(Xleft, Xright, divcount);
  for (int i = 1; i < divcount; i++) {
    interval inter;
    if (i == 1) {
      inter.begin = Xleft;
      inter.end = points[0];
      inter.value = 0;
      pointvector.push_back(inter);
    }

    inter.begin = points[i - 1];
    inter.end = points[i];
    inter.value = 0;
    pointvector.push_back(inter);
  }

  do {
    for (int i = 0; i < divcount; i++) {
      double* minvalue = new double;
      *minvalue = DBL_MAX;
      for (double cur = Yleft; cur < Yright; cur += presision) {
        double rez = RastriginFunc(points[i], cur);

        if (rez < *minvalue) {
          *minvalue = rez;
        }
      }
      results[i] = *minvalue;
      pointvector[i].value = *minvalue;
      intervals.insert(pointvector[i]);
    }
    iter = intervals.begin();
    interval temp1 = *iter;
    ++iter;
    interval temp2 = *iter;
    step = (abs(temp1.end - temp2.end));
    if (step <= presision) break;
    iter = intervals.begin();
    for (int i = 0; i < divcount; i++) {
      temp1 = *iter;

      if (i % 2 == 0) {
        points[i] = temp1.end - ((temp1.end - temp1.begin) * 9 / 10.0);

        temp2.end = points[i];
        temp2.begin = temp1.begin;

        intervals.erase(iter);

        temp1.begin = points[i];
        intervals.insert(temp1);
        pointvector[i] = temp2;
        iter = intervals.begin();
        for (int j = 0; j < i; j++) {
          ++iter;
        }
      } else {
        points[i] = temp1.end + ((temp1.end - temp1.begin) * 9 / 10.0);

        temp2.end = points[i];
        temp2.begin = temp1.end;
        pointvector[i] = temp2;
        ++iter;
      }
    }
  } while (step > presision);

  iter = intervals.begin();
  interval out = *iter;
  global_min = out.value;
  return global_min;
}

double ParallelOptimization(double presision, double Xleft, double Xright,
                            double Yleft, double Yright) {
  int size, rank;

  double point = 0;

  double minvalue = 0;
  double step = Xright - Xleft;
  double global_min = DBL_MAX;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size > 2) {
    if (rank == 0) {
      int divcount = size - 1;
      std::vector<double> points;
      std::vector<double> results(divcount);
      std::vector<interval> pointvector;
      std::multiset<interval, custom_compare> intervals;
      std::multiset<interval>::iterator iter;
      std::multiset<interval>::iterator iter2;

      points = getPoints(Xleft, Xright, divcount);
      for (int i = 1; i < divcount; i++) {
        interval inter;
        if (i == 1) {
          inter.begin = Xleft;
          inter.value = 0;
          inter.end = points[0];
          pointvector.push_back(inter);
        }
        inter.begin = points[i - 1];
        inter.end = points[i];
        inter.value = 0;
        pointvector.push_back(inter);
      }
      while (step > presision) {
        for (int proc = 1; proc < size; proc++) {
          int idx = proc - 1;
          double a = points[idx];

          MPI_Send(&a, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }

        for (int proc = 1; proc < size; proc++) {
          MPI_Recv(&minvalue, 1, MPI_DOUBLE, proc, MPI_ANY_TAG, MPI_COMM_WORLD,
                   MPI_STATUSES_IGNORE);

          int idx = proc - 1;
          results[idx] = minvalue;

          pointvector[idx].value = minvalue;
          intervals.insert(pointvector[idx]);
        }

        iter = intervals.begin();
        interval temp1 = *iter;
        ++iter;
        interval temp2 = *iter;

        step = (abs(temp1.end - temp2.end));
        MPI_Bcast(&step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        iter = intervals.begin();

        for (int i = 0; i < divcount; i++) {
          temp1 = *iter;

          if (i % 2 == 0) {
            points[i] = temp1.end - ((temp1.end - temp1.begin) * 9 / 10.0);

            temp2.end = points[i];
            temp2.begin = temp1.begin;

            intervals.erase(iter);

            temp1.begin = points[i];
            intervals.insert(temp1);
            pointvector[i] = temp2;
            iter = intervals.begin();
            for (int j = 0; j < i; j++) {
              ++iter;
            }
          } else {
            points[i] = temp1.end + ((temp1.end - temp1.begin) * 9 / 10.0);

            temp2.end = points[i];
            temp2.begin = temp1.end;
            pointvector[i] = temp2;
            ++iter;
          }
        }
      }

      iter = intervals.begin();
      interval out = *iter;
      global_min = out.value;

    } else {
      while (step > presision) {
        MPI_Recv(&point, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
                 MPI_STATUSES_IGNORE);
        double min = DBL_MAX;
        for (double cur = Yleft; cur < Yright; cur += presision) {
          double rez = RastriginFunc(point, cur);
          if (rez < min) {
            min = rez;
          }
        }
        MPI_Send(&min, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Bcast(&step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
      global_min = 0;
    }
  } else {
    if (rank == 0) {
      global_min =
          SeqentalOptimization(presision, Xleft, Xright, Yleft, Yright);
    }
  }
  return global_min;
}
