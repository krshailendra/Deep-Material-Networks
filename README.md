# Deep Material Network for Multiscale Topology Learning

A PyTorch implementation of a **Deep Material Network (DMN)** for efficient multiscale modeling of heterogeneous materials.

The project implements a physics-consistent, hierarchical material network that learns to predict the **effective homogenized compliance tensor** of a heterogeneous Representative Volume Element (RVE) from the properties of its constituent material phases.

---

## 📌 Overview

Multiscale modeling of heterogeneous materials is computationally expensive when using high-fidelity approaches such as **Direct Numerical Simulation (DNS)** and **Finite Element Methods (FEM)**.

Traditional machine-learning approaches can be computationally efficient, but may fail to preserve the underlying physical behavior of the material.

The **Deep Material Network (DMN)** addresses this trade-off by combining:

* Physics-based homogenization
* Analytical building blocks
* Neural-network-based parameter optimization
* Hierarchical binary-tree topology
* Tensor rotation
* Gradient-based learning

Instead of directly learning a black-box mapping, the DMN represents a complex microstructure using a hierarchy of physically meaningful building blocks.

---

## 🎯 Objective

The primary objective is:

> **Learn a compact material-network representation that accurately predicts the effective homogenized response of heterogeneous materials while maintaining physical consistency.**

The model takes the compliance matrices of two constituent material phases and predicts the homogenized compliance matrix obtained from DNS data.

---

## 🧠 Deep Material Network Architecture

The DMN is structured as a **hierarchical binary tree**.

Each node represents a **building block** that performs two operations:

1. **Homogenization** — combines the properties of two child materials.
2. **Rotation** — accounts for the orientation of the resulting material structure.

The information flows from the individual material phases at the bottom of the tree toward a single effective material representation at the root.

### Network Structure

```text
                 Effective Material
                        │
                  Building Block
                 /              \
        Building Block      Building Block
          /       \            /       \
        ...       ...        ...       ...
         │         │          │         │
       Phase 1   Phase 2    Phase 1   Phase 2
```

For a network of depth `N`, the number of bottom-layer nodes grows as:

```text
2^N
```

The network learns:

* Bottom-layer activation values
* Volume-fraction-related weights
* Rotation angles
* Hierarchical material topology

The architecture and data flow are described in the presentation on pages 3–5.

---

## 🔬 Physics-Based Building Block

The fundamental unit of the network is the **BuildingBlock** module.

Given two material phases:

* Compliance matrix `D₁`
* Compliance matrix `D₂`
* Volume fractions `f₁` and `f₂`
* Rotation angle `θ`

the building block first performs homogenization and then rotates the resulting compliance tensor.

Conceptually:

```text
Material 1 ─────┐
                ├──► Homogenization ──► Rotation ──► Effective Material
Material 2 ─────┘
```

This allows the network to preserve important physical constraints such as:

* Stress equilibrium
* Strain compatibility
* Directional dependence
* Positive strain energy

The implemented `BuildingBlock` module follows this formulation directly.

---

## ⚙️ Activation & Volume Fraction Modeling

The bottom-layer weights are controlled using **ReLU activation**:

```text
w = max(z, 0)
```

This provides two important properties:

* Inactive nodes can automatically be removed.
* The network can be simplified through compression.

Volume fractions are represented using node weights. Parent-node weights are recursively computed from their child-node weights, allowing the network to maintain the appropriate material proportions.

---

## 🔄 Tensor Rotation

Material orientation is incorporated through rotation matrices defined in **Mandel notation**.

The implementation provides GPU-optimized functions for:

* Constructing batched rotation matrices
* Converting `3 × 3` symmetric compliance matrices into 6-component Mandel vectors
* Converting Mandel vectors back into matrices
* Performing batched tensor rotations

PyTorch's `torch.bmm` is used for efficient batch matrix multiplication during GPU computation.

---

## 📊 Dataset

The model is trained using a dataset generated using **Direct Numerical Simulation (DNS)**.

### Input

The model receives compliance information for two material phases.

**Phase 1**

```text
D11_p1
D12_p1
D13_p1
D22_p1
D23_p1
D33_p1
```

**Phase 2**

```text
D11_p2
D12_p2
D13_p2
D22_p2
D23_p2
D33_p2
```

Total input features:

```text
12
```

### Target

The homogenized material response consists of:

```text
D11_hom
D12_hom
D13_hom
D22_hom
D23_hom
D33_hom
```

Total output features:

```text
6
```

The dataset therefore represents the mapping:

```text
Phase 1 Properties
        +
Phase 2 Properties
        ↓
Deep Material Network
        ↓
Homogenized Compliance Tensor
```

The dataset format and training workflow are documented on pages 13 and 24–26.

---

## 🛠️ Data Preprocessing

The preprocessing pipeline:

1. Extracts phase stiffness matrices and effective stiffness matrices.
2. Converts stiffness tensors to compliance tensors using:

```text
D = C⁻¹
```

3. Converts symmetric `3 × 3` compliance matrices into 6-component Mandel vectors.
4. Splits the dataset into:

   * Training set
   * Validation set
   * Test set
5. Converts the data into PyTorch tensors.
6. Creates PyTorch `DataLoader`s for batch-based training.

---

## 🏗️ Implementation

The implementation is divided into several major components.

### 1. Dataset & Environment Setup

* PyTorch
* NumPy
* Matplotlib
* GPU/CUDA detection
* Reproducible random seeds
* `.mat` DNS dataset loading

The implementation automatically selects CUDA when available.

---

### 2. Rotation Functions

GPU-optimized tensor rotation functions handle:

```text
Compliance Matrix
       ↓
Mandel Representation
       ↓
Rotation Matrix
       ↓
Rotated Compliance
```

---

### 3. BuildingBlock

The `BuildingBlock` module:

* Receives two compliance tensors.
* Computes `f₂ = 1 - f₁`.
* Performs two-layer homogenization.
* Constructs the intermediate compliance tensor.
* Applies tensor rotation.
* Returns the effective compliance vector.

---

### 4. MaterialNetwork

The `MaterialNetwork` class implements the complete hierarchical DMN.

It:

* Creates the binary tree.
* Initializes bottom-layer activation parameters.
* Creates learnable rotation angles.
* Stores reference compliance tensors.
* Computes bottom-layer material weights.
* Performs recursive homogenization.
* Computes local volume fractions.
* Applies rotations at every node.
* Produces the final effective compliance tensor.
* Applies regularization to stabilize training.

---

### 5. Forward Pass

The forward pass operates from the bottom of the binary tree toward the root:

```text
Phase Compliance
       ↓
Weighted Bottom Nodes
       ↓
Homogenization
       ↓
Rotation
       ↓
Higher-Level Nodes
       ↓
Homogenization
       ↓
Rotation
       ↓
       ...
       ↓
Root Node
       ↓
Effective Compliance
```

The final output is a single homogenized compliance tensor representing the heterogeneous material.

---

## 📉 Loss Function

The model uses a **normalized squared Frobenius norm** to measure the difference between the predicted and DNS homogenized compliance tensors.

Conceptually:

```text
Loss =
|| D_DNS - D_DMN ||²_F
-----------------------
     || D_DNS ||²_F
```

Normalization reduces the influence of differences in overall material-property scale.

A regularization term is additionally used to constrain the network parameters and improve training stability.

---

## 🔁 Training

The training framework uses:

* **PyTorch**
* **Adam optimizer**
* Gradient descent
* Backpropagation
* Regularization
* Gradient clipping
* Learning-rate scheduling
* Separate training and validation loops

The optimizer updates the learnable:

```text
Activation parameters (z)
Rotation angles (θ)
```

using gradients obtained through backpropagation.

The learning-rate scheduler reduces the learning rate when validation loss plateaus.

---

## 🧮 Backpropagation

Since the loss is defined only at the final output, the error must be propagated through the entire material network.

The implementation computes gradients through the hierarchy using the chain rule:

```text
Output Error
     ↓
Root Node
     ↓
Parent Nodes
     ↓
Child Nodes
     ↓
Bottom Layer
     ↓
Activation & Rotation Parameters
```

Gradients are calculated with respect to both:

* Activation/weight parameters
* Rotation angles

This enables end-to-end optimization of the material network.

---

## 🗜️ Model Compression

The DMN can be compressed to reduce unnecessary network complexity.

Two compression strategies are described:

### Node Deletion

Nodes with inactive/irrelevant weights can be removed.

### Subtree Merging

Similar substructures can be merged to eliminate redundancy.

The network is reordered using its weighting functions before similarity-based subtree merging. The presentation describes performing compression periodically during training to reduce training complexity.

---

## 📈 Results

The project compares multiple implementations and optimization strategies.

### PyTorch + Adam

The best reported result was obtained using the **PyTorch implementation with the Adam optimizer**.

Reported convergence:

```text
Training Loss    ≈ 0.24
Validation Loss  ≈ 0.17
```

The conclusion slide reports these values after approximately **1000 epochs**.

### PyTorch + SGD

The SGD implementation converged early, around approximately epoch 100, after which the training and validation losses remained nearly constant because the learning rate became very small.

### Scratch Implementation

A separate implementation using **finite differences for gradient calculation** was also explored.

For larger datasets and deeper networks, this approach was estimated to require approximately **50–60 hours** for long training runs, making it significantly less practical than the PyTorch automatic-differentiation implementation.

---

## 🧪 Evaluation on Unseen Data

After training, the model is evaluated on previously unseen test samples.

The evaluation pipeline:

1. Switches the model to evaluation mode.
2. Disables gradient computation.
3. Generates predictions for test samples.
4. Computes normalized Frobenius-norm error.
5. Calculates average test loss.
6. Prints sample predictions alongside their target compliance tensors.

The trained PyTorch model is then saved for future use.

---

## 💾 Model Saving

The trained model parameters are saved using PyTorch:

```python
torch.save(model.state_dict(), "dmn_pytorch.pt")
```

This allows the learned DMN parameters to be reused without retraining the entire network.

---

## 🔬 Key Features

* Physics-informed material modeling
* Deep Material Network architecture
* Hierarchical binary-tree topology
* Analytical homogenization
* Tensor rotation using Mandel notation
* GPU-optimized tensor operations
* PyTorch automatic differentiation
* Backpropagation through the material network
* Adam and SGD optimization
* Frobenius-norm-based loss
* Regularization
* Gradient clipping
* Learning-rate scheduling
* Model compression
* Train/validation/test evaluation
* PyTorch model serialization

---

## 📚 Reference

The implementation is based on the Deep Material Network formulation described in:

**Zeliang Liu, C. T. Wu, and M. Koishi (2019)**
*“A Deep Material Network for Multiscale Topology Learning and Accelerated Nonlinear Modeling of Heterogeneous Materials.”*

The project also references Zeliang Liu's Deep Material Network blog.

---

## 👥 Authors

**Kumar Shailendra**
IIT Guwahati
Roll No.: 230103125

**Devanandika P**
IIT Guwahati
Roll No.: 230103033

---

## 🚀 Project Summary

This project demonstrates how a **physics-aware neural architecture** can be used to learn the effective behavior of heterogeneous materials.

Rather than treating material homogenization as a purely black-box machine-learning problem, the Deep Material Network embeds **homogenization mechanics, material orientation, volume fractions, and hierarchical topology directly into the architecture**.

The resulting model combines the advantages of:

```text
Physics-Based Modeling
        +
Deep Learning
        +
GPU Acceleration
        ↓
Efficient Multiscale Material Modeling
```
