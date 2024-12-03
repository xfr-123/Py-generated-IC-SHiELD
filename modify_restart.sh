#conda init bash
mkdir -p CACHE
cd CACHE
gridspec-create gcs 96
gridspec-create latlon 180 360
ESMF_RegridWeightGen --source c96_gridspec.nc --destination regular_lat_lon_180x360.nc --method conserve --weight c96_to_180x360_weights.nc
ESMF_RegridWeightGen --source regular_lat_lon_180x360.nc --destination c96_gridspec.nc --method conserve --weight 180x360_to_c96_weights.nc
