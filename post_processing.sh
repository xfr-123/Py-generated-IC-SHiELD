#!/bin/bash
### set the wallclock time
#SBATCH --time=1:00:00

### set the number of nodes, tasks per node, and cpus per task for the job
#SBATCH --partition=milan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

### set the job name
#SBATCH --job-name="SHIELD_post"

### set a file name for the stdout and stderr from the job
### the %j parameter will be replaced with the job ID.
### By default, stderr and stdout both go to the --output
### file, but you can optionally specify a --error file to
### keep them separate
#SBATCH --output=SHIELD_post.o%j

srun --partition=rome --ntasks=1 --cpus-per-task=1 singularity exec --no-home /u/xfr123/container/ufs-srw.sif /opt/ufs-srweather-app/container-bin/make_hgrid --grid_type gnomonic_ed --nlon 192
srun --partition=rome --ntasks=1 --cpus-per-task=1 singularity exec --no-home /u/xfr123/container/ufs-srw.sif /opt/ufs-srweather-app/container-bin/make_solo_mosaic --num_tiles 6 \
--dir . --tile_file horizontal_grid.tile1.nc,horizontal_grid.tile2.nc,horizontal_grid.tile3.nc,horizontal_grid.tile4.nc,horizontal_grid.tile5.nc,horizontal_grid.tile6.nc

srun --partition=rome --ntasks=1 --cpus-per-task=1 singularity exec --no-home /u/xfr123/container/ufs-srw.sif /opt/ufs-srweather-app/container-bin/fregrid \
--input_mosaic solo_mosaic.nc --nlon 360 --nlat 180 --input_file atmos_daily --scalar_field PWAT,h_plev,u_plev,v_plev,t_plev,PRESsfc,VORT850,VORT500,VORT200,omg500
#,q_plev,omg_plev

srun --partition=rome --ntasks=1 --cpus-per-task=1 singularity exec --no-home /u/xfr123/container/ufs-srw.sif /opt/ufs-srweather-app/container-bin/fregrid \
--input_mosaic solo_mosaic.nc --nlon 360 --nlat 180 --input_file atmos_4x_hourly --scalar_field PWAT,h_plev,u_plev,v_plev,t_plev,q_plev,omg_plev,PRESsfc