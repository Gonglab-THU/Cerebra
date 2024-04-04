<h1 align="center">The training process</h1>

## Data Preparation

1. Download `OpenProteinSet` dataset from https://registry.opendata.aws/openfold/ and place it in the "rawdata" folder, and download the cif files from https://www.rcsb.org and place them in the "rawdata/cif" directory.

2. Generate cache file using openfold/scripts/generate_chain_data_cache.py(`The script in the standard version of OpenFold https://github.com/aqlaboratory/openfold`).
3. Transform the data using data_transform.py into a format accepted by the model.

## Training

* You could modify the loss function in train.py.


```bash
python train.py
```
