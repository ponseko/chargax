# Chargax: A JAX Accelerated EV Charging Simulator

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

---

## 📦 Installation

For those using [uv](https://docs.astral.sh/uv/getting-started/installation/), it is possible to run a standard PPO implementation with default settings by directly running `uv run main.py`.

```bash
git clone git@github.com:ponseko/chargax.git
cd chargax
uv run main.py
```

Alternatively, install the project as an editable package in your favourite virtual environment software. E.g. using conda:

```bash
git clone git@github.com:ponseko/chargax.git
cd chargax
conda create -n chargax python=3.11
conda activate chargax
pip install -e .

python main.py
```

for CUDA support, additionally run `pip install jax[cuda]`.

---

## 📑 Citing

```bibtex
@misc{ponse2025chargaxjaxacceleratedev,
      title={Chargax: A JAX Accelerated EV Charging Simulator}, 
      author={Koen Ponse, Jan Felix Kleuker, Aske Plaat, Thomas Moerland},
      year={2025},
      eprint={2507.01522},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.01522}, 
}
```
