#conda init bash
mkdir -p CACHE
cd CACHE
gridspec-create gcs 96
gridspec-create latlon 1800 3600
ESMF_RegridWeightGen --source c96_gridspec.nc --destination regular_lat_lon_1800x3600.nc --method conserve2nd --weight c96_to_1800x3600_weights.nc
ESMF_RegridWeightGen --source regular_lat_lon_1800x3600.nc --destination c96_gridspec.nc --method conserve2nd --weight 1800x3600_to_c96_weights.nc
