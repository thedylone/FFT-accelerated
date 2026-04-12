#!/bin/bash
#SBATCH --job-name=fft_omp
#SBATCH --output=fft_omp_%j.out
#SBATCH --error=fft_omp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

# Activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

echo "--- Compiling and running ... ---"
cmake -B build .
cmake --build build

echo "FFT openmp"

echo "threads,n,time_ms" > results.csv

for T in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$T
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    OUTPUT=$(build/main $T)
    echo "$OUTPUT"

    # Append only numeric data rows to CSV
    echo "$OUTPUT" | awk -F',' '$1~/^[0-9]+$/ { print }' >> results.csv
done

echo "--- Tidying ---"
rm -r build
