# Py-generated-IC-SHiELD

This project aims to use python to generate idealized initial conditions and test the development of phonomenon in the atmosphere. 

## Work Flow

1. **Cold Start**  
   By using cold start method, we could generate the files that is required for the restart process.

    ```
    ./prep_cold.sh
    ./run.sh
    ```


2. **Prepare transform data**  
   To convert the latitude - longitude files to 6 tile files that is need for the model input, we need to calculate the weight first.
   ```
   ./modify_restart.sh
   ```
   Note: The python packages need must be installed and the environment must be activated before this process.

3. **Python generate IC**  
   We could use python to generate the initial conditions we want to test in the next step.
   ```
   python ic_generator.py --IsPerturbation --Shift 10
   ```

   If we want to do a stable test, we could remove the `--IsPerturbation` order, like:
   ```
   python ic_generator.py --Shift 10
   ```


4. **Prepare Restart**  
   Remove the original restart IC files.
   ```
   rm -rf ./RESTART/fv_core*
   ```

   Then copy all the IC files generated from `./CACHE` to `./RESTART`.

   ```
   cp ./CACHE/fv_core* ./RESTART/
   ```



5. **Restart the Model**  
   First, prepare the warm start configuration:
   ```
   ./prep_warm.sh
   ```
   for moist case, or:
   ```
   ./prep_warm_dry.sh
   ```
   for dry case.

   Then we could submit the task to our server:

   ```
   sbatch run.sh
   ```



6. **Post-processing**  
   As the output of SHiELD model is still 6 tile files, we need to add a post-processing step to convert it to a lat-lon grid file.

   ```
   ./post_processing.sh
   ```

7. **Visulization**

   You can check your results with `viz.ipynb`


## Reletive Slides

The experiment's preliminary results could be found at this [link](https://uillinoisedu-my.sharepoint.com/:p:/g/personal/gzhang13_illinois_edu/EVL23E-EaN9Mo2Rmm3fQgR4BWDUfOQoiToMX_txF01IJnA?e=ERn9o6).

## Reference

This work is based on the [GFDL's SHiELD build system](https://github.com/NOAA-GFDL/SHiELD_build/tree/main).


