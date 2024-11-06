<h1 align="center">Cerebra</h1>
<p align="center">A computationally efficient framework for accurate protein structure prediction</p>

## Install software on Linux

1. download `Cerebra`

```bash
git clone https://github.com/Gonglab-THU/Cerebra.git
cd Cerebra
```

2. install `Anaconda` / `Miniconda` software

3. install Python packages

```bash
conda create -n cerebra python=3.11
conda activate cerebra

conda install pytorch 
pip install click
pip install numpy
pip install ml_collections
pip install scipy
pip install einops
pip install dm-tree
pip install biopython
```

## Usage

* You should modify the [MSA file](example/test.a3m)
* You should download the [model1-6 parameters](https://doi.org/10.5281/zenodo.10608345) and move it into the `model` folder

```bash
wget https://zenodo.org/records/10608346/files/model.zip?download=1
tar -zcvf model.zip ./
bash run.sh -i example/test.a3m -o example
```

* If you need to analyze the pearson correlation between path metrics and PSA attention weights, please use `model6.pth`.
* As searching MSA can be time-consuming, please use `search_MSA.py` to search MSA if you want to obtain results consistent with those in Cerebra.

## Reference

[Cerebra: a computationally efficient framework for accurate protein structure prediction](https://doi.org/10.1101/2024.02.02.578551)
