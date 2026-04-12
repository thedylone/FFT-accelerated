#!/bin/bash
#SBATCH --job-name=fft_serial
#SBATCH --output=fft_serial_%j.out
#SBATCH --error=fft_serial_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

# Activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

echo "--- Compiling and running ... ---"
cmake -B build .
cmake --build build

echo "FFT serial"

OUTPUT=$(build/main)
echo "$OUTPUT"

echo "n,time_ms" > results.csv
# Append only numeric data rows to CSV
echo "$OUTPUT" | awk -F',' '$1~/^[0-9]+$/ { print }' >> results.csv

echo "--- Tidying ---"
rm -r build
