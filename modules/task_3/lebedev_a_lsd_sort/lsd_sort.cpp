// Copyright 2021 Lebedev Alexey
#include "../../../modules/task_3/lebedev_a_lsd_sort/lsd_sort.h"

#include <mpi.h>
#include <math.h>

#include <algorithm>
#include <cstring>
#include <string>


#define MIN_JOB 100

size_t get_max_digits(std::vector<int>::iterator first, std::vector<int>::iterator last) {
    size_t max_digit = 0;
    for (auto iter = first; iter != last; iter++) {
        size_t digit = (*iter < 0) ? std::to_string(*iter).size() - 1 : std::to_string(*iter).size();
        if (digit > max_digit) {
            max_digit = digit;
        }
    }
    return max_digit;
}

void lsd_sort(std::vector<int>::iterator first, std::vector<int>::iterator last) {
    if (first == last) {
        return;
    }
    static const size_t range = 19;
    std::vector<int> containers[range];
    size_t max_digits = get_max_digits(first, last);
    int ten_pow_dig = 1;
    for (size_t cur_dig = 0; cur_dig < max_digits; cur_dig++) {
        for (auto iter = first; iter != last; iter++) {
            containers[*iter / ten_pow_dig % 10 + 9].push_back(*iter);
        }
        ten_pow_dig *= 10;
        int* write_it = &(*first);
        for (size_t i = 0; i < range; i++) {
            std::memcpy(write_it, containers[i].data(), containers[i].size() * sizeof(int));
            write_it += containers[i].size();
            containers[i].clear();
        }
    }
}

void lsd_sort_parallel(std::vector<int>::iterator first, std::vector<int>::iterator last) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int max_size = ((last - first) % MIN_JOB == 0) ? (last - first) / MIN_JOB : (last - first) / MIN_JOB + 1;
        size = std::min(size, max_size);
        if (first == last) {
            size = 0;
        }
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm NEW_WORLD;
    int color = (rank < size) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &NEW_WORLD);
    if (NEW_WORLD == MPI_COMM_NULL) {
        return;
    }

    std::vector<int> local_v{0};

    if (rank == 0) {
        int jobs_per_proc = (last - first) / size;
        int last_jobs = (last - first) % size;
        int local_size = (last_jobs == 0) ? jobs_per_proc : jobs_per_proc + 1;
        local_v.resize(local_size);
        std::memcpy(&local_v[0], &(*last), local_size);
        int offset = local_size;
        for (int pid = 1; pid < size; pid++) {
            int count = (pid < last_jobs) ? jobs_per_proc + 1 : jobs_per_proc;
            MPI_Send(&(*last) + offset, count, MPI_INT, pid, 0, NEW_WORLD);
            offset += count;
        }
    } else {
        int local_size;
        MPI_Status status;
        MPI_Probe(0, 0, NEW_WORLD, &status);
        MPI_Get_count(&status, MPI_FLOAT, &local_size);
        local_v.resize(local_size);
        MPI_Recv(local_v.data(), local_size, MPI_INT, 0, 0, NEW_WORLD, &status);
    }

    lsd_sort(local_v.begin(), local_v.end());

    if (rank == 0) {
        std::vector<int> tmp(last - first);
        if (size > 1) {
            std::memcpy(tmp.data(), local_v.data(), local_v.size() * sizeof(int));
        } else {
            std::memcpy(&(*first), local_v.data(), local_v.size() * sizeof(int));
        }
        int offset = local_v.size();
        for (int i = 1; i < size; i++) {
            int count;
            MPI_Status status;
            MPI_Probe(i, 0, NEW_WORLD, &status);
            MPI_Get_count(&status, MPI_FLOAT, &count);
            MPI_Recv(&tmp[offset], count, MPI_INT, i, 0, NEW_WORLD, &status);
            auto last_merged = tmp.begin() + offset;
            std::merge(tmp.begin(), last_merged, last_merged, last_merged + count, first);
            offset += count;
            std::memcpy(&tmp[0], &(*first), offset * sizeof(int));
        }
    } else {
        MPI_Send(local_v.data(), local_v.size(), MPI_INT, 0, 0, NEW_WORLD);
    }

    MPI_Comm_free(&NEW_WORLD);
}
