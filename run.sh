#!/bin/bash
make clean && make

echo ""
echo "Size: 1000x1000 * 1000x1000"
echo "Sequential"
./matmult 1000 1000 1000
echo "Parallel"
./matmult_omp 1000 1000 1000
echo ""

echo "Size: 1000x2000 * 2000x5000"
echo "Sequential"
./matmult 1000 2000 5000
echo "Parallel"
./matmult_omp 1000 2000 5000
echo ""

echo "Size: 9000x2000 * 2000x3750"
echo "Sequential"
./matmult 9000 2500 3750
echo "Parallel"
./matmult_omp 9000 2500 3750
echo ""