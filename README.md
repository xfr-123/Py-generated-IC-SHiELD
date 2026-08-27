# Py-generated-IC-SHiELD

This project aims to use python to generate idealized initial conditions and test the development of phonomenon in the atmosphere. 

## Requirements

This project requires two distinct Python environments to handle different stages of the workflow:

* **`modify_restart.sh`:** To run this script, the **`esmpy`** environment must be activated. Refer to `esmpy_requirements.txt` for the specific dependency list.
* **IC Generation:** For all other Python-based Initial Condition (IC) generation processes, please use the **`pygen_clean`** environment as detailed in `pygen_requirements.txt`.

---

## Environment Setup
If you need to recreate these environments from the requirements files, you can use the following commands:

```bash
# Create the esmpy environment
conda create --name esmpy --file esmpy_requirements.txt

# Create the pygen_clean environment
conda create --name pygen_clean --file pygen_requirements.txt
```
## Work Flow

1. **Cold Start**  
   By using cold start method, we could generate the files that is required for the restart process.

    ```
    ./prep_cold.sh
    ./run.sh
    ```


2. **Prepare transform weights**  
   To convert the latitude - longitude files to 6 tile files that is need for the model input, we need to calculate the weight first.
   ```
   ./modify_restart.sh
   ```
   Note: The python packages need must be installed and the environment must be activated before this process.

3. **Python generate IC**  
   We could use python to generate the initial conditions we want to test in the next step.
   ```
   python ic_generator.py --IsPerturbation --Shift 10 --b 2 --n 3 --RH0 0.80
   ```

   If we want to do a stable test, we could remove the `--IsPerturbation` order, like:
   ```
   python ic_generator.py --Shift 10 --b 2 --n 3 --RH0 0.80
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


## Reference

This work is based on the [GFDL's SHiELD build system](https://github.com/NOAA-GFDL/SHiELD_build/tree/main).

## Publication and Revision Notebooks

The reproducible publication notebooks are stored in `publication_notebooks/`. The historical August 2, 2026 figure notebooks are retained, and the current reviewer-stage updates are consolidated in:

- `publication_notebooks/06_revision_updates_20260823.ipynb`
- `publication_notebooks/revision_20260823/`

The revision bundle includes finalized PNG/PDF figures, exact analysis/plotting scripts, compact supporting tables, a machine-readable manifest, validation notes, and SHA-256 checksums. Large raw SHiELD simulation files are not duplicated in this repository.

## Latest EKE-domain revision

The latest EKE-related update is maintained separately from the historical August 23 snapshot:

- `publication_notebooks/07_revision_updates_20260827.ipynb`
- `publication_notebooks/revision_20260827/`

It recomputes area- and mass-weighted EKE and initial Eady context over 25–90°N, rediagnoses the Figure 7 50–80% EKE-growth windows, and updates the eddy-heat-flux profiles. The exact scripts, compact tables, final PNG/PDF figures, environment record, validation notes, and SHA-256 manifest are included. Original multi-GB model files remain external inputs and are not copied into the repository.
