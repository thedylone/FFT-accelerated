#!/bin/bash
#SBATCH --job-name=fft_jax
#SBATCH --output=fft_jax_%j.out
#SBATCH --error=fft_jax_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

# Activate spack env for Python, then install JAX
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu
export PATH="$HOME/.local/bin:$PATH"
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install "jax[cuda12]"

echo "--- Running JAX FFT ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "backend,n,time_ms" > results.csv

# GPU run
JAX_PLATFORMS=cuda OUTPUT=$(python source/main.py gpu)
echo "$OUTPUT"
echo "$OUTPUT" | awk -F',' '$1~/^(gpu|cpu)$/ { print }' >> results.csv

# CPU run
JAX_PLATFORMS=cpu OUTPUT=$(python source/main.py cpu)
echo "$OUTPUT"
echo "$OUTPUT" | awk -F',' '$1~/^(gpu|cpu)$/ { print }' >> results.csv
