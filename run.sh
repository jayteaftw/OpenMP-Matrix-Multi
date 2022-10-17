#!/bin/bash

make clean && make

echo ""
echo "Size: 1000x1000 * 1000x1000"
./matmult_omp 1000 1000 1000 | tee 1000x1000*1000x1000.log
echo ""

echo "Size: 1000x2000 * 2000x5000"
./matmult_omp 1000 2000 5000 | tee 1000x2000*2000x5000.log
echo ""

echo "Size: 9000x2000 * 2000x3750"
./matmult_omp 9000 2500 3750 | tee 9000x2000*2000x3750.log
echo ""