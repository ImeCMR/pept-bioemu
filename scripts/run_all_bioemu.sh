#!/bin/bash
#SBATCH --job-name=bioemu
#SBATCH --output=logs/run_%x_%j.out
#SBATCH --error=logs/run_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=imeshr150@gmail.com
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40000mb
#SBATCH --time=15:00:00
#SBATCH --qos=alberto.perezant

ml conda
conda activate /orange/alberto.perezant/imesh.ranaweera/projects/7_bioEMU_peptide/pept-bioemu_env

cd /orange/alberto.perezant/imesh.ranaweera/projects/7_bioEMU_peptide/pept-bioemu

# Use GPU for JAX (BioEmu sampling) and TensorFlow (AlphaFold embeddings)
# JAX_PLATFORMS=cpu was causing slow TensorFlow initialization on CPU
pept-bioemu run-all --config /orange/alberto.perezant/imesh.ranaweera/projects/7_bioEMU_peptide/pept-bioemu/config/peptides.yaml