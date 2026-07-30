<h1 align="center">Cerebra</h1>
<p align="center">A computationally efficient framework for accurate protein structure prediction</p>

Cerebra predicts protein structures from either a single protein sequence or an
input multiple sequence alignment. This release provides a unified inference
script for FASTA, FASTA directory, A3M, and A3M directory inputs, with ESM2
embeddings generated automatically during inference.

## Install Software on Linux

1. Download `Cerebra`

```bash
git clone https://github.com/Gonglab-THU/Cerebra.git
cd Cerebra
```

2. Install `Anaconda` or `Miniconda`

3. Create the Cerebra environment

```bash
conda env create -f environment.yml
conda activate cerebra_MSA
```

The provided `environment.yml` contains the tested inference environment:

- Python 3.11
- PyTorch 2.9.1 with CUDA 12.8 wheels
- NumPy 2.4
- SciPy 1.17
- Biopython 1.87
- OpenMM 8.5 and PDBFixer 1.12 for optional Amber relaxation
- `fair-esm` for local ESM2 `.pt` files
- `transformers` for the default HuggingFace ESM2 backend

If your CUDA driver cannot run CUDA 12.8 PyTorch wheels, please edit the PyTorch
entries in `environment.yml` according to your local CUDA version before
creating the environment. CPU inference is supported, but is much slower.

## Download Model Parameters

Cerebra model parameters will be released through Zenodo.

Zenodo link: [https://zenodo.org/records/21698980](https://zenodo.org/records/21698980)

After downloading the model parameters, put them into the `checkpoint` folder:

```text
checkpoint/
|-- model_1.pt
|-- model_2.pt
|-- model_3.pt
|-- model_4.pt
`-- model_5.pt
```

By default, `inference_model.py` loads `checkpoint/model_1.pt`. You can select a
different model by passing `--model_id`, or provide an explicit checkpoint path
with `--checkpoint`.

## Input Files

### FASTA

Use FASTA input for single-sequence prediction without an external MSA. This
mode is intended for the hallucination-based de novo protein backbone design
setting described in the paper.

```text
>target_name
MSEQUENCE...
```

Each FASTA file should contain exactly one target sequence. For multiple
targets, put one sequence in each FASTA file and use `--fasta_dir`.

### A3M

Use A3M input for MSA-based prediction.

```text
>target_name
MSEQUENCE...
>homolog_1
MSEQUENCE...
```

The first A3M sequence is treated as the target sequence. Lowercase insertion
characters and `.` characters are removed automatically. Columns where the
target sequence has a gap are ignored.

For standard protein structure prediction, we recommend using A3M input whenever
an MSA is available, as MSA-based inference usually gives the best prediction
results.

## Usage

Single-sequence prediction from one FASTA file:

```bash
python inference_model.py \
  --fasta example/9kgz.fasta \
  --model_id 1 \
  -o predictions/
```

Batch single-sequence prediction from a FASTA directory:

```bash
python inference_model.py \
  --fasta_dir fastas/ \
  --model_id 1 \
  -o predictions/
```

MSA-based prediction from an A3M file:

```bash
python inference_model.py \
  --a3m example/9azo_A.a3m \
  --model_id 1 \
  -o predictions/
```

Batch MSA-based prediction from an A3M directory:

```bash
python inference_model.py \
  --a3m_dir a3ms/ \
  --model_id 1 \
  -o predictions/
```

Use a specific Cerebra checkpoint:

```bash
python inference_model.py \
  --fasta example/9kgz.fasta \
  --checkpoint /path/to/model_3.pt \
  -o predictions/
```

Run Amber relaxation after prediction:

```bash
python inference_model.py \
  --a3m example/9azo_A.a3m \
  --model_id 1 \
  -o predictions/ \
  --relax
```

Relaxation runs on CPU by default. If your OpenMM CUDA installation is
compatible, you can request GPU relaxation:

```bash
python inference_model.py \
  --a3m example/9azo_A.a3m \
  --model_id 1 \
  -o predictions/ \
  --relax \
  --relax_device cuda:0
```

## ESM2 Embeddings

If `--esm2_model` is not provided, Cerebra uses the HuggingFace model
`facebook/esm2_t36_3B_UR50D` by default. The model will be downloaded into the
HuggingFace cache on first use, so internet access is required unless the model
is already cached.

You can also specify an ESM2 model manually:

```bash
python inference_model.py \
  --fasta example/9kgz.fasta \
  --esm2_model /path/to/esm2_t36_3B_UR50D.pt \
  --model_id 1 \
  -o predictions/
```

`--esm2_model` accepts:

- A local fair-esm `.pt` file, such as `esm2_t36_3B_UR50D.pt`
- A HuggingFace model id, such as `facebook/esm2_t36_3B_UR50D`
- A local directory containing a HuggingFace-format ESM2 model

## Common Options

| Option | Description |
| --- | --- |
| `--fasta PATH` | Predict one target from a FASTA file. |
| `--fasta_dir DIR` | Predict one target from each FASTA file in a directory. |
| `--a3m PATH` | Predict one target from an A3M alignment. |
| `--a3m_dir DIR` | Predict one target from each A3M file in a directory. |
| `-o, --output PATH` | Output PDB file or output directory. |
| `--model_id N` | Load `checkpoint/model_N.pt`; default is `1`. |
| `--checkpoint PATH` | Load an explicit Cerebra checkpoint. |
| `--checkpoint_dir DIR` | Directory containing `model_1.pt`, `model_2.pt`, etc. |
| `--esm2_model VALUE` | Optional ESM2 source. Omit to use HuggingFace ESM2. |
| `--device DEVICE` | Inference device, such as `cuda:0` or `cpu`. |
| `--precision auto|bf16|fp32` | Default is `auto`; bf16 is used only on supported CUDA GPUs. |
| `--cycles N` | Number of recycle cycles; default is `4`. |
| `--relax` | Write an additional Amber-relaxed PDB. |
| `--relax_device DEVICE` | Relaxation device; default is `cpu`. |

For all available options:

```bash
python inference_model.py --help
```

## Output Files

When `--output` is a directory, predictions are written as:

```text
<target_name>_model_<model_id>.pdb
```

If `--relax` is used, an additional relaxed structure is written as:

```text
<target_name>_model_<model_id>_relaxed.pdb
```

The script also prints the mean pLDDT score for each predicted target.

## Training

A complete training script will be released in a later update.

## Notes

- For best prediction results, provide an MSA with `--a3m` or `--a3m_dir`.
- To run without an MSA for hallucination-based de novo protein backbone design,
  provide a FASTA file with `--fasta` or a FASTA directory with `--fasta_dir`.
- If OpenMM reports CUDA compatibility errors during relaxation, use CPU
  relaxation by omitting `--relax_device` or setting `--relax_device cpu`.

## Reference

[Cerebra: a computationally efficient framework for accurate protein structure prediction](https://doi.org/10.1101/2024.02.02.578551)
