#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=KFAC
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --time=00:59:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

cd $SLURM_TMPDIR
cp -r ~/scratch/K-FAC .
cd K-FAC

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python main.py
