#!/bin/bash
#SBATCH --job-name=fft_jax
#SBATCH --output=fft_jax_%j.out
#SBATCH --error=fft_jax_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

# Activate spack env for Python, then use uv to install JAX
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu
export PATH="$HOME/.local/bin:$PATH"
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install "jax[cuda12]"

echo "--- Running JAX FFT (GPU) ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

OUTPUT=$(python source/main.py)
echo "$OUTPUT"

echo "n,time_ms" > results.csv
echo "$OUTPUT" | awk -F',' '$1~/^[0-9]+$/ { print }' >> results.csv
