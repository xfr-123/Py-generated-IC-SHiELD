#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=sandybridge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1

#SBATCH --job-name="shifted_jet"

#SBATCH --output=SHIELD_BC

#SBATCH --mail-type=END
#SBATCH --mail-user=email@server

#SBATCH --account=running account

srun --ntasks=24 --cpus-per-task=1 singularity exec --no-home /u/xfr123/container/shield_sandbox_checksum.sif /SHiELD_build/Build/bin/SOLO_nh.prod.32bit.gnu.x
#srun --mpi=pmix --export=ALL,PMIX_MCA_psec='^munge' --ntasks=24 --cpus-per-task=1 singularity exec --bind $PWD:$PWD --no-home /data/keeling/a/mingfei5/container/shield_sandbox_checksum.sif /SHiELD_build/Build/bin/SOLO_nh.prod.32bit.gnu.x
