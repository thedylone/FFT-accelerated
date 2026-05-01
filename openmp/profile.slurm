#!/bin/bash
#SBATCH --job-name=fft_omp_vtune
#SBATCH --output=fft_omp_vtune_%j.out
#SBATCH --error=fft_omp_vtune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00

# Activate spack env
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

cd "$SLURM_SUBMIT_DIR"

echo "--- Compiling ---"
cmake -B build .
cmake --build build

export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "--- VTune hotspots ---"
vtune -collect hotspots -result-dir vtune_hotspots -- build/main 32

echo "--- VTune threading ---"
vtune -collect threading -result-dir vtune_threading -- build/main 32

echo "--- Generating reports ---"
vtune -report summary -result-dir vtune_hotspots > vtune_hotspots_summary.txt
vtune -report summary -result-dir vtune_threading > vtune_threading_summary.txt

echo "--- Tidying ---"
rm -r build
