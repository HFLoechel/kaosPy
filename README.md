
![kaosPy Logo](docs/img/logo.png)

kaosPy is a Python package for generating **Chaos Game Representation (CGR)**
and **Frequency Chaos Game Representation (FCGR)** of symbolic sequences.

It transforms sequences with arbitrary alphabets into two-dimensional
representations. CGR provides a geometric representation of sequence
composition, while FCGR converts the representation into a frequency-based
matrix.

kaosPy can be used for applications such as:

- sequence embeddings for machine learning,
- alignment-free sequence comparison,
- visualization and exploration of biological sequences,
- phylogenetic and comparative sequence analysis.

## Installation

```` bash
pip install kaospy
````

## Features

- Chaos Game Representation

- Frequency Chaos Game Representation

- Frequency Chaos Game Collections

## Quickstart

```bash
from kaospy import CGR, FCGR

sequence = "ACGTACGTACGT"

cgr = CGR(sequence, symbols="DNA")
fcgr = FCGR(cgr, resolution=16)

fcgr.plot()

```

## Examples

| Notebook                                                              | Description                       |
|-----------------------------------------------------------------------|-----------------------------------|
| [Basic CGR](examples/Basic_CGR_Example.ipynb)                         | Introduction to CGR               |
| [Basic FCGR](examples/Basic_FCGR_Example.ipynb)                       | Introduction to FCGR              |
| [Basic FCGRCollections](examples/Basic_FCGRCollections_Example.ipynb) | Introduction to FCGRCollections   |

## Applications

| Notebook                                                   | Description                                          |
|------------------------------------------------------------|------------------------------------------------------|
| [N-flake Example](applications/N-Flake_example.ipynb)      | Example application using synthetic random sequences |
| [DNA Example](applications/Application_DNA.ipynb)          | Example application using DNA sequences              |
| [Protein Example](applications/Application_Proteins.ipynb) | Example application using protein sequences          |

## Experiments

| Notebook                                                 | Description                                               |
|----------------------------------------------------------|-----------------------------------------------------------|
| [DNA Subtype Analysis](experiments/HIV_Subtype.ipynb)      | PCA of HIV subtypes                                       |
| [Protein Classification](experiments/HIV_Tropism.ipynb)    | V3 loop classification using machine learning             |
| [Phylogenetic Analysis](experiments/Betacoronavirus.ipynb) | Alignment-free phylogenetic analysis of betacoronaviruses |

## Documentation

Full documentation is currently under development.
API documentation and tutorials will be available soon.

## Citation
Please cite: 
TBA