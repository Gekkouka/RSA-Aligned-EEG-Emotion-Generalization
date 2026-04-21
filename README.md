# RSA-Aligned EEG Emotion Generalization

Implementation of **Learning Domain-Invariant Representations for EEG Emotion Generalization via Representational Similarity Alignment**.

## Overview

This work studies **cross-subject EEG emotion recognition** under the **domain generalization** setting.  
We propose an **RSA-based structural consistency regularizer** to align representational dissimilarity structures across subjects, encouraging the model to learn more subject-invariant and emotion-relevant latent representations. :contentReference[oaicite:1]{index=1}

## Datasets

Experiments are conducted on:

- **SEED**
- **SEED-IV**

The implementation uses the pre-computed **Differential Entropy (DE)** features provided by the datasets.  
Each sample is represented as a **310-dimensional** feature vector (62 channels × 5 frequency bands).

## Experimental Setting

- Framework: PyTorch
- Optimizer: RMSprop
- Learning rate: 1e-3
- Maximum training iterations: 1000
- Evaluation protocol: **Leave-One-Subject-Out (LOSO)**
- Random seed: 20 :contentReference

## Results

| Method | SEED Avg. (%) | SEED-IV Avg. (%) |
|--------|---------------|------------------|
| Baseline | 81.53 | 66.46 |
| DANN | 81.86 | 68.53 |
| MMD | 83.19 | 68.83 |
| CORAL | 82.02 | 68.64 |
| DAAN | 82.16 | 68.34 |
| RSA (ours) | **83.70** | 68.78 |

The proposed method achieves the best accuracy on **SEED** and competitive performance with the lowest standard deviation on **SEED-IV**.

## Project Structure

```text
model/      # model definitions
train.py    # training script
test.py     # evaluation script
utils.py    # utility functions
```

## Usage

Before running the code, please first prepare a `config.yaml` file and set the required paths and hyperparameters.

Run:
```bash
python main_seed.py
```
