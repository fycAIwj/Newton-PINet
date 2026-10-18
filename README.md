
# Newton-PINet: A Fast Physics-Informed Neural Network with Newton Linearization for Meta-Learning Nonlinear PDEs

**Yuchen Fan, Chang Wei, Pao-Hsiung Chiu, Chin Chun Ooi, Heyang Wang, Jian Cheng Wong**

🎉 **Accepted at NeurIPS 2026!** 🎉

**Paper:** [OpenReview – Paper and Discussion](https://openreview.net/forum?id=wUNTVhsOr7#discussion)

## Overview

Existing deep learning methods for nonlinear PDEs often require substantial training data or costly optimization during task adaptation.

**Newton-PINet** addresses these limitations through three key components:
1. **Efficient architecture:** A physics-informed neural network with skip connections and a least-squares output layer.
2. **Two-stage learning:** Gradient-based meta-learning followed by gradient-free output-layer fine-tuning for rapid adaptation to new PDE tasks.
3. **Newton linearization:** Accelerates least-squares iterations for nonlinear PDEs.

<p align="center">
  <img src="Newton-PINet%20model%20framework.jpg"
       alt="Newton-PINet model framework"
       width="90%">
</p>

**Impact:** Newton-PINet reduces training data requirements while enabling fast, accurate, and physics-consistent generalization to unseen PDE tasks, advancing the meta-learning of PINNs toward data-efficient, fast, and generalizable physics solvers.

## Install

<p>
  <img src="https://img.shields.io/badge/jax-0.4.13-blue" alt="jax 0.4.13">
  <img src="https://img.shields.io/badge/jaxlib-0.3.10%2Bcuda11.cudnn805-orange" alt="jaxlib 0.3.10">
  <img src="https://img.shields.io/badge/evojax-0.2.17-red" alt="evojax 0.2.17">
  <img src="https://img.shields.io/badge/flax-0.6.4-lightgrey" alt="flax 0.6.4">
  <img src="https://img.shields.io/badge/optax-0.1.8-blue" alt="optax 0.1.8">
</p>

## Environment

- **Language:** Python
- **OS:** Linux
- **GPU:** NVIDIA Tesla V100
- **NVIDIA Driver:** 580
- **CUDA:** 13.0

