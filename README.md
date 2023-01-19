# Satellite burned area dataset
This repository contains the code to run the experiments on the Satellite Burned Area Dataset (https://doi.org/10.5281/zenodo.6597139) for the resource paper "A Dataset for Burned Area Delineation and Severity Estimation from Satellite Imagery" published at CIKM'22.
This dataset contains satellitery imagery collected from Sentinel-1 (GRD) and Sentinel-2 (L2A) missions (Copernicus) about regions previously affected by forest wildfires.
The goal of this dataset is to provide an open and extended dataset for Earth Observation in the context of wildfire detection, burned area detection and severity estimation.

## Scripts
 - `compute_si.py`: script used to compute the Separability Index of different burned area indexes (BAI, BAIS2, NBR, NBR2) and evaluate the best one;
 - `otsu_baseline.py`: script used to compute the performance of the Otsu thresholding baseline over all the folds;
 - `cross_validation.py`: script used to run the 7-fold cross-validation with a UNet binary model;

An example of data loading using the `SatelliteDataset` class can be found [here](https://github.com/lccol/burned-area-baseline/blob/3ec40a05a2142aba99058030f987ffbdfec688af/otsu_baseline.py#L66).

# Citation
If you used our dataset in your work, please use the following BibTeX entry:
```
@inproceedings{10.1145/3511808.3557528,
    author = {Colomba, Luca and Farasin, Alessandro and Monaco, Simone and Greco, Salvatore and Garza, Paolo and Apiletti, Daniele and Baralis, Elena and Cerquitelli, Tania},
    title = {A Dataset for Burned Area Delineation and Severity Estimation from Satellite Imagery},
    year = {2022},
    isbn = {9781450392365},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3511808.3557528},
    doi = {10.1145/3511808.3557528},
    booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
    pages = {3893–3897},
    numpages = {5},
    location = {Atlanta, GA, USA},
    series = {CIKM '22}
}
```
