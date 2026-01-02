

# ReverSea: A Depth-Aware CNN for Low-Light Underwater Image Enhancement ([Paper](https://ieeexplore.ieee.org/abstract/document/11104786))

[![License](https://img.shields.io/badge/License-Research%20Only-informational.svg)](#license)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)

ReverSea is a lightweight, **depth-aware CNN** for enhancing **low-light underwater images**.  
It directly processes raw images by incorporating **transmission maps** to restore natural colors, improve visibility, and handle non-uniform illumination.

<div align="center">
  <img src="Reversea.png" width="75%"/>
</div>

## Highlights
- **Depth-aware enhancement** using transmission maps
- **Lightweight CNN** architecture
- Effective in **deep-sea** and **low-light** conditions

---

## Repository Structure (Quick Tour)
- `Reversea/train+inf.ipynb` : **Training + Inference** notebook (main entrypoint)
- `Reversea/Weights/` : pretrained weights for SeeThru and OceanDark Datasets
- `req.txt` : python dependencies

---

## Requirements
- **Python 3.12** (recommended / tested)
- Install dependencies from `req.txt`

---

## Setup & Installation

### 1) Clone the repository
```bash
git clone https://github.com/AIRLabIISc/REVERSEA.git
cd Reversea
```

### 2) Create and activate a virtual environment
```bash
python3.12 -m venv reversea
source reversea/bin/activate
```

### 3) Install requirements
```bash
pip install --upgrade pip
pip install -r req.txt
```

## Usage

To execute ReverSea Model use the provided notebook:
train+inf.ipynb

## Paper
**ReverSea: A Depth-Aware CNN for Low-Light Underwater Image Enhancement** :
https://ieeexplore.ieee.org/abstract/document/11104786

## Citation
```bibtex
@inproceedings{makam2025reversea,
  title={ReverSea: An Enhancement Technique for Low-Lit Underwater Images},
  author={Makam, Rajini and Shankari, Dhatri and Patil, Sharanya and Sundram, Suresh},
  booktitle={OCEANS 2025 Brest},
  pages={1--8},
  year={2025},
  organization={IEEE}
}
