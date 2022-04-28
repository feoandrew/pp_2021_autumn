// Copyright 2021 Lebedev Alexey
#include <gtest/gtest.h>
#include <random>
#include <iostream>
#include "./lsd_sort.h"
#include <gtest-mpi-listener.hpp>


std::vector<int> get_random_vector(size_t size, int low, int high) {
    std::vector<int> random_vector(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(low, high);
    for (size_t i = 0; i < size; i++) {
        random_vector[i] = distrib(gen);
    }
    return random_vector;
}


TEST(LSD_SORT_MPI, TEST_SORT_SEQENTUAL) {
    std::vector<int> v = get_random_vector(10000, -10000, 10000);
    lsd_sort(v.begin(), v.end());
    ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
}


TEST(LSD_SORT_MPI, Test_SIGNED_VALUES) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> v;
    std::vector<int> tmp;

    if (rank == 0) {
        v = get_random_vector(1000000, -10000, 10000);
        tmp = v;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    lsd_sort_parallel(v.begin(), v.end());
    double end = MPI_Wtime();

    if (rank == 0) {
        double time_p = end - start;
        start = MPI_Wtime();
        lsd_sort(tmp.begin(), tmp.end());
        end = MPI_Wtime();
        double time_s = end - start;
        std::cout << "Sequential time: " << time_s << std::endl;
        std::cout << "Parallel time: " << time_p << std::endl;
        std::cout << "SpeedUp: " << time_p / time_s << std::endl;
        ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
    }
}


TEST(LSD_SORT_MPI, Test_NEGATIVE_VALUES_ONLY) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> v;

    if (rank == 0) {
        v = get_random_vector(10000, -1000000, 0);
    }

    lsd_sort_parallel(v.begin(), v.end());

    if (rank == 0) {
        ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
    }
}


TEST(LSD_SORT_MPI, Test_POSITIVE_VALUES_ONLY) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> v;

    if (rank == 0) {
        v = get_random_vector(10000, 0, 1000000);
    }

    lsd_sort_parallel(v.begin(), v.end());

    if (rank == 0) {
        ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
    }
}


TEST(LSD_SORT_MPI, Test_EMPTY_ARRAY) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> v;

    if (rank == 0) {
        v = get_random_vector(0, -1000000, 1000000);
    }

    lsd_sort_parallel(v.begin(), v.end());

    if (rank == 0) {
        ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
    }
}


TEST(LSD_SORT_MPI, Test_SORTED_ARRAY) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> v;

    if (rank == 0) {
        v = get_random_vector(1000, -1000, 1000);
        std::sort(v.begin(), v.end());
    }

    lsd_sort_parallel(v.begin(), v.end());

    if (rank == 0) {
        ASSERT_TRUE(std::is_sorted(v.begin(), v.end()));
    }
}


TEST(LSD_SORT_MPI, Test_RANGE) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> v;

    if (rank == 0) {
        v = get_random_vector(1000, -1000, 1000);
        std::sort(v.begin(), v.end());
    } else {
        v.resize(1000);
    }

    lsd_sort_parallel(v.begin() + 100, v.end() - 500);

    if (rank == 0) {
        ASSERT_TRUE(std::is_sorted(v.begin() + 100, v.end() - 500));
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
