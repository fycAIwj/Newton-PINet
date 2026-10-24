
# Newton-PINet: A fast physics-informed neural network with Newton linearization for meta-learning nonlinear PDEs

**Yuchen Fan, Chang Wei, Pao-Hsiung Chiu, Chin Chun Ooi, Heyang Wang, Jian Cheng Wong**

**Paper:** [NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/poster/148607)  
**OpenReview:** [Reviews and Discussion](https://openreview.net/forum?id=wUNTVhsOr7#discussion)

## Overview

Existing deep learning methods for nonlinear PDEs often require substantial training data or costly optimization during task adaptation.

**Newton-PINet** addresses these limitations through three key components:
1. **Efficient architecture:** A skip-connected network with the output-layer weights computed using Tikhonov regularization.
2. **Two-stage learning:** Meta-learn task-agnostic representations spanning a family of PDEs and enable fast generalization to unseen tasks via fine-tuning only the output-layer weights using the Tikhonov regularization.
3. **Newton linearization:** Accelerates the Tikhonov regularization iterations for nonlinear PDEs.


<p align="center">
  <img src="Newton-PINet%20model%20framework.jpg"
       alt="Newton-PINet model framework"
       width="90%">
</p>

**Significance:** Newton-PINet reduces training data requirements while enabling fast, accurate, and physics-consistent generalization to unseen PDE tasks, advancing the meta-learning of PINNs toward data-efficient, fast, and generalizable physics solvers.

## Install

<p>
  <img src="https://img.shields.io/badge/JAX-blue?logo=python" alt="JAX">
  <img src="https://img.shields.io/badge/jaxlib-orange" alt="jaxlib">
  <img src="https://img.shields.io/badge/EvoJAX-green" alt="EvoJAX">
  <img src="https://img.shields.io/badge/Flax-purple" alt="Flax">
  <img src="https://img.shields.io/badge/Optax-red" alt="Optax">
</p>

## Environment

- **Language:** Python
- **OS:** Linux
- **GPU:** NVIDIA Tesla V100
- **NVIDIA Driver:** 580
- **CUDA:** 13.0

