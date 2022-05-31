# Satellite burned area dataset
This repository contains the code to run the experiments on the Satellite Burned Area Dataset (https://doi.org/10.5281/zenodo.6597139).
This dataset contains satellitery imagery collected from Sentinel-1 (GRD) and Sentinel-2 (L2A) missions (Copernicus) about regions previously affected by forest wildfires.
The goal of this dataset is to provide an open and extended dataset for Earth Observation in the context of wildfire detection, burned area detection and severity estimation.

## Scripts
 - `compute_si.py`: script used to compute the Separability Index of different burned area indexes (BAI, BAIS2, NBR, NBR2) and evaluate the best one;
 - `otsu_baseline.py`: script used to compute the performance of the Otsu thresholding baseline over all the folds;
 - `cross_validation.py`: script used to run the 7-fold cross-validation with a UNet binary model;

An example of data loading using the `SatelliteDataset` class can be found [here](https://github.com/lccol/burned-area-baseline/blob/3ec40a05a2142aba99058030f987ffbdfec688af/otsu_baseline.py#L66).
