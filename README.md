# machine-learning-ozone-parameterization

Code for the paper **"mloz: A Highly Efficient Machine Learning-Based Ozone Parameterization for Climate Sensitivity Simulations"**, published on the *Journal of Advances in Modeling Earth Systems* (JAMES).

**Contributors:** 
Yiling Ma — yiling.ma@kit.edu — Karlsruhe Institute of Technology (KIT), 
Luke Nathan Abraham - University of Cambridge

## Overview

mloz replaces expensive full-chemistry ozone solvers in climate models (UKESM & ICON) with a ridge regression machine learning (ML) model trained offline on UKESM full-chemistry simulations. It predicts the 3D ozone field based on the local temperature column at each grid point and at a daily frequency. The mloz is implemented as Fortran modules in ICON-ART and UKESM-UKCA.

## Repository Structure

```
training/                  # Offline training scripts
implementation/
    ICON/                  # key Fortran modules for mloz in ICON-ART
    UKESM/                 # key Fortran module for mloz in UKESM-UKCA
visualization/             # Figure scripts
    FigureS1/              # Offline experiment scripts + FigureS1
```

### Training (`training/`)

| File | Description |
|------|-------------|
| `Ridge_train.sample` | Template for training one ridge model per longitude batch (48 batches × 4 longitudes). |
| `Ridge_train.csh` | SLURM submission script; generates and submits parallel jobs from the template. |
| `Ridge_train_create_ncfiles.py` | Consolidates batch pickle files into NetCDF files (coefficients, scalings) for use in climate models. |

### Fortran Implementations

| File | Description |
|------|-------------|
| `implementation/UKESM/ukca_mach_learn_mod.F90` | Core mloz subroutines in UKESM-UKCA: normalizes temperature, applies ridge regression, returns ozone prediction each day. |
| `implementation/ICON/mo_art_read_mloz.f90` | Reads mloz coefficient files into ICON-ART memory at initialization. |
| `implementation/ICON/mo_art_mloz.f90` | Core mloz subroutines in ICON-ART. |

### Figure Scripts (`visualization/`)

| File | Figure |
|------|--------|
| `Figure2.py` | Fig. 2 — Ozone time series at representative grid points |
| `Figure3.py` | Fig. 3 — Ozone probability density functions at representative grid points |
| `Figure4.py` | Fig. 4 — Climatological zonal-mean cross-sections of ozone & ozone response to 4xCO2 |
| `Figure5.py` | Fig. 5 — Percentage bias of mloz vs. full chemistry |
| `Figure6_column_ozone_integration.py` | Preprocessing for Fig. 6 — column ozone integration |
| `Figure6.py` | Fig. 6 — Global maps of total column ozone bias |
| `Figure7.py` | Fig. 7 — Tropical-mean temperature response profiles |
| `FigureS1/FigureS1.py` | Fig. S1 — Offline R² comparison across prediction schemes |
| `FigureS1/FigureS1_Offline_Zonal_Input.py` | Ridge regression with zonal temperature input |
| `FigureS1/FigureS1_Offline_Global_Input.py` | Ridge regression with global temperature input (PCA-reduced) |
| `FigureS1/FigureS1_Offline_NN.py` | Feedforward neural network offline experiment |

## Dependencies

**Python:** `numpy`, `xarray`, `netCDF4`, `scipy`, `matplotlib`, `cartopy`, `scikit-learn`, `joblib`, `torch` (PyTorch, required only for `FigureS1_Offline_NN.py`)

## Data

Training uses UKESM piCTRL and 4×CO₂ full-chemistry simulations (1961–1999). The training pipeline produces separate NetCDF files for each experiment, including temperature scalings (Scaler_x_*.nc), ozone scalings (Scaler_y_*.nc), and ridge regression coefficients (coefs_*.nc). These files are subsequently used by climate models during online testing.

A coarse-grained (60°×10°) version of the training data (downsampled due to storage constraints), along with the online ozone predictions from mloz in UKESM and ICON and the full-resolution mloz coefficients, is archived at:

> Ma, Y., & Abraham, N. L. (2026). Datasets for mloz: A Highly Efficient Machine Learning-Based Ozone Parameterization for Climate Sensitivity Simulations [Data set]. Zenodo. https://doi.org/10.5281/zenodo.19056391

The original full-resolution training data and online ozone predictions are available from the author upon request.

## Procedures

1. **Get training data** — download the coarse-grained full chemistry simulations from Zenodo or use your own ESM full-chemistry output.
2. **Train** — run `csh Ridge_train.csh` on an HPC system to submit parrelel jobs; train seperately for piCTRL and 4×CO₂. 
3. **Consolidate** — run `python Ridge_train_create_ncfiles.py` to produce the NetCDF coefficient files.
4. **Run model** — implement the Fortran modules into ICON-ART or UKESM-UKCA and use the offline-trained coefficients for mloz calculation.
5. **Plot** — run figure scripts in `visualization/`. Run `Figure6_column_ozone_integration.py` before `Figure6.py`. Run `FigureS1/FigureS1_Offline_*.py` scripts before `FigureS1/FigureS1.py`.

## License

Code in this repository is licensed under the **Creative Commons Attribution 4.0 International**.

## Citation

Ma, Y., & Abraham, N. L. (2026). Code for mloz: A Highly Efficient Machine Learning-Based Ozone Parameterization for Climate Sensitivity Simulations. Zenodo. https://doi.org/10.5281/zenodo.19076781
