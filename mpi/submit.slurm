#!/bin/bash
#SBATCH --job-name=fft_mpi
#SBATCH --output=fft_mpi_%j.out
#SBATCH --error=fft_mpi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

# Activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

echo "--- Compiling and running ... ---"
cmake -B build .
cmake --build build

echo "FFT MPI"

echo "processes,n,time_ms" > results.csv

for P in 1 2 4 8 16 32; do
    OUTPUT=$(srun -n $P build/main)
    echo "$OUTPUT"

    # Append only numeric data rows to CSV
    echo "$OUTPUT" | awk -F',' '$1~/^[0-9]+$/ { print }' >> results.csv
done

echo "--- Tidying ---"
rm -r build
