#include <iostream>
#include <cstring>
#include <omp.h>
#include <chrono>

int* a;
int* b;
int* tb;
int* r;

int* a2;
int* b2;
int* r2;

int N = 512;
int R = 10;

// Build: g++ matrix_mul.cpp -O4 -fopenmp -std=c++11 -o matrix_mul
// Run: OMP_PLACES=cores OMP_PROC_BIND=close numactl -m 0 ./matrix_mul 2048

int main(int argc, char *argv[])
{
    if (argc >= 2)
    {
        N = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        R = atoi(argv[2]);
    }

    srand((unsigned)time(0));

    unsigned long avg_time = 0;

    a  = (int*)aligned_alloc(16, N * N * sizeof(int));
    b  = (int*)aligned_alloc(16, N * N * sizeof(int));
    tb = (int*)aligned_alloc(16, N * N * sizeof(int));
    r  = (int*)aligned_alloc(16, N * N * sizeof(int));
    
    a2 = (int*)malloc(N * N * sizeof(int));
    b2 = (int*)malloc(N * N * sizeof(int));
    r2 = (int*)malloc(N * N * sizeof(int));
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = rand() % N + 3;
            b[i * N + j] = (rand() % N) + 5;
	    a2[i * N + j] = a[i * N + j];
	    b2[i * N + j] = b[i * N + j];
        }
    }; 

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tb[i * N + j] = b[j * N + i];
        }
    }

    for (int round = 0; round < R; round++)
    {
	memset(r, 0, N*N*sizeof(int));

        unsigned long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        #pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
		int tmp = 0;
                for (int k = 0; k < N; k++)
                {
                    tmp += a[i * N + k] * tb[j * N + k];
                }
		r[i * N + j] = tmp;
            }
        }

        unsigned long t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	unsigned long new_time = t2 - t1;
	avg_time = (avg_time * round + new_time)/(round + 1);
    }

    std::cout << "Average execution time (optimized): " << avg_time << "ms" << std::endl;

    avg_time = 0;

    for (int round = 0; round < R; round++)
    {
	memset(r2, 0, N*N*sizeof(int));

	unsigned long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	
	#pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < N; k++)
                {
                    r2[i * N + j] += a2[i * N + k] * b2[k * N + j];
                }
            }
        }

	unsigned long t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        unsigned long new_time = t2 - t1;
        avg_time = (avg_time * round + new_time)/(round + 1);
    }

    std::cout << "Average execution time (naive): " << avg_time << "ms" << std::endl;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
	    if (r[i * N + j] != r2[i * N + j]) {
		std::cerr << "Error: Wrong result" << std::endl;
		return -1;
	    }
        }
    }

    return 0;
}

