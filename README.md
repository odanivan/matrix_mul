#matrix mul

## Optimizing matrix multiplication: Cache-friendly code

A small experiment with optimizing a parallelized implementation of matrix multiplication of two N x N matrices. Inspired by Scott Meyers' talk at code::dive 2014 'Cpu Caches and Why You Care'.

I use OpenMP for multi-threaded execution (a simple parallel clause around the outer for-loop). I transpose the second matrix to ensure row-wise memory access pattern for both matrices. (Traversal order matters!)

I use a local variable per thread for the temporary result of calculating the dot product of the two corresponding row vectors, and only update the result matrix at the end of the calculation. This avoids the effect of false sharing which would otherwise occur: Even though each thread writes to distinct entries within the result matrix, the threads would invalidate each other's cache lines because subsequent matrix entries are aligned in memory (plain array) and thus on the same cache line. With a thread-local temporary variable, the false sharing performance drop is eliminated.

## How to build and run
```
g++ matrix_mul.cpp -O4 -fopenmp -std=c++11 -o matrix_mul
OMP_PLACES=cores OMP_PROC_BIND=close numactl -m 0 ./matrix_mul 2048

```
