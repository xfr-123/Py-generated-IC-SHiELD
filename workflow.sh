#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=sandybridge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1

#SBATCH --job-name="BCdry_restart"

#SBATCH --output=SHIELD_BC

#SBATCH --mail-type=END
#SBATCH --mail-user=mingfei5@illinois.edu

#SBATCH --account=bbwv-hydro

./prep_cold.sh
./run.sh

source ~/.bashrc

conda activate my_env

python ic_generator.py --IsPerturbation --Shift 10

rm -rf ./RESTART/fv_core*

ls ./RESTART

cp ./CACHE/fv_core* ./RESTART

ls ./RESTART

./prep_warm_dry.sh

srun --ntasks=24 --cpus-per-task=1 singularity exec --no-home /u/xfr123/container/shield_sandbox_checksum.sif /SHiELD_build/Build/bin/SOLO_nh.prod.32bit.gnu.x
