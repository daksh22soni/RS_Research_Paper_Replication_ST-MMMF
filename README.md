# ST-MMMF: Data Augmentation and Refinement for Recommender Systems

> Study of **"Data Augmentation and Refinement for Recommender System: A Semi-Supervised Approach Using Maximum Margin Matrix Factorization"**
> Shamal Shaikh, Venkateswara Rao Kagita, Vikas Kumar, Arun K Pujari

---

## Table of Contents

- [Overview](#overview)
- [Paper Summary](#paper-summary)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Overview

This repository contains the full replication of the **ST-MMMF** (Self-Training with Maximum Margin Matrix Factorization) algorithm proposed in Paper 22. The study implements a semi-supervised data augmentation and refinement pipeline for collaborative filtering-based recommender systems, evaluated on the MovieLens 100K and MovieLens 1M benchmark datasets.

---

## Paper Summary

Collaborative Filtering (CF) recommender systems suffer from **data sparsity** — most user-item ratings are unobserved. This paper addresses sparsity by proposing:

- A **self-training semi-supervised loop** that iteratively augments the sparse rating matrix with high-confidence predicted ratings
- A **data refinement step** that removes low-confidence known ratings from the training set
- Use of **Maximum Margin Matrix Factorization (MMMF)** as the base learner, exploiting its geometrical decision-boundary interpretation to define confidence regions

The key insight is that MMMF's learned threshold structure provides a natural confidence signal: predictions far from decision boundaries (deep inside a rating region) are considered high-confidence and eligible for augmentation, while known ratings close to boundaries are considered unreliable and are refined away.

---

## Methodology

### Maximum Margin Matrix Factorization (MMMF)

Given a sparse rating matrix **Y ∈ R^(N×M)**, MMMF learns factor matrices **U ∈ R^(N×d)** and **V ∈ R^(M×d)** and a per-user threshold matrix **Θ** by minimizing a regularized smooth hinge loss:

```
min J(U, V, Θ) = Σ h(T^r_ij · (θ_{i,r} - U_i V_j^T)) + (λ/2)(‖U‖²_F + ‖V‖²_F)
```

Ratings are predicted by mapping the real-valued score `x_ij = U_i V_j^T` to the discrete scale using the learned thresholds.

### ST-MMMF Algorithm

Each iteration consists of four stages:

| Stage | Description |
|-------|-------------|
| **1. Training** | Run MMMF on current rating matrix Y to obtain U, V, Θ |
| **2. Augmentation** | Add unobserved entries predicted with high confidence (inside the central region bounded by τ₁) |
| **3. Refinement** | Remove known entries predicted with low confidence (within τ₂ of a decision boundary) |
| **4. Replacement** | Update Y with the augmented and refined matrix for the next iteration |

**Confidence criterion** for augmentation (Equation 12):
```
θ_{i,r-1} + ϑ_i · τ₁ < U_i V_j^T < θ_{i,r} − ϑ_i · τ₁
```

**Confidence criterion** for refinement (Equation 13):
```
θ_{i,r} − ϑ_i · τ₂ < U_i V_j^T < θ_{i,r} + ϑ_i · τ₂
```

where `ϑ_i` is the average inter-threshold gap for user i, and τ₁, τ₂ are shifting parameters (τ₁ > τ₂).

### Handling Class Imbalance

To prevent augmentation from amplifying the dominant rating class, augmented samples are weighted inversely proportional to their current class frequency (Equation 14):

```
weight(r) = (1 − Z_r) / Σ_j (1 − Z_j)
```

where Z_r is the proportion of rating r in the current training set.

---

## Repository Structure

```
RS_Research_Paper_Replication_ST-MMMF/
│
├── README.md                      ← This file
├── requirements.txt               ← Python dependencies
├── .gitignore                     ← Files excluded from version control
│
├── src/
│   ├── movielens_100k_eda.py      ← EDA script for MovieLens 100K
│   ├── movielens_1m_eda.py        ← EDA script for MovieLens 1M
│   └── rsproject.ipynb            ← Recommendation System Replication Code
│
├── data/
│   ├── ml-100k/                   ← MovieLens 100K raw data
│   │   ├── u.data                 ← Full dataset (100,000 ratings)
│   │   ├── u.item                 ← Movie information
│   │   ├── u.user                 ← User demographics
│   │   ├── u.genre                ← Genre list
│   │   ├── u.occupation           ← Occupation list
│   │   ├── u1.base / u1.test      ← 80/20 split fold 1
│   │   ├── ...                    ← Folds 2–5 + ua/ub splits
│   │   └── README                 ← Official dataset documentation
│   │
│   └── ml-1m/                     ← MovieLens 1M raw data
│       ├── ratings.dat            ← 1,000,209 ratings
│       ├── movies.dat             ← Movie information
│       ├── users.dat              ← User demographics
│       └── README                 ← Official dataset documentation
│
├── results/
│   ├── ml-100k-eda-output/                  ← EDA results for MovieLens 100K dataset
│   │   ├── step4_rating_distribution.png    ← Rating distribution (histogram)
│   │   ├── step5_user_activity.png          ← User activity distribution
│   │   ├── step6_movie_popularity.png       ← Movie popularity analysis
│   │   ├── step7_genre_distribution.png     ← Genre frequency distribution
│   │   ├── step8_user_demographics.png      ← User demographics (age, gender)
│   │   ├── step9_rating_trends.png          ← Rating trends over time
│   │   └──  step10_join_integrity.png        ← Data consistency / join validation
│   │
│   ├── ml-1m-eda-output/                    ← EDA results for MovieLens 1M dataset
│   │   ├── 01_rating_distribution.png       ← Rating distribution
│   │   ├── 02_user_activity_distribution.png← User activity distribution
│   │   ├── 03_movie_popularity_distribution.png ← Movie popularity
│   │   ├── 04_genre_distribution.png        ← Genre distribution
│   │   ├── 05_age_analysis.png              ← Age group analysis
│   │   ├── 06_gender_analysis.png           ← Gender distribution
│   │   ├── 07_occupation_analysis.png       ← Occupation analysis
│   │   └──  EDA_SUMMARY.txt                  ← Summary insights from EDA
│   │
│   ├── fig5_rating_distribution.png   ← Rating distribution (Figure 5)
│   ├── fig6_performance_curves.png    ← MAE/RMSE curves (Figure 6)
│   ├── table_100K_train_iter1.png     ← Confusion matrix — 100K train, iter 1
│   ├── table_100K_train_iter50.png    ← Confusion matrix — 100K train, iter 50
│   ├── table_100K_test_iter1.png      ← Confusion matrix — 100K test, iter 1
│   ├── table_100K_test_iter50.png     ← Confusion matrix — 100K test, iter 50
│   ├── table_1M_train_iter1.png       ← Confusion matrix — 1M train, iter 1
│   ├── table_1M_train_iter50.png      ← Confusion matrix — 1M train, iter 50
│   ├── table_1M_test_iter1.png        ← Confusion matrix — 1M test, iter 1
│   ├── table_1M_test_iter50.png       ← Confusion matrix — 1M test, iter 50
│   ├── performance_results.csv        ← MAE/RMSE per iteration (all models)
│   └── output.txt                     ← Full console log from execution
│
└── document/
    ├── Paper 22                       ← Original Research Paper 
    └── replication_manual.docx        ← Detailed replication manual
```

---

## Datasets

Both datasets are the standard MovieLens benchmarks from [GroupLens](https://grouplens.org/datasets/movielens/).

### MovieLens 100K

| Property | Value |
|----------|-------|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 |
| Rating scale | 1–5 |
| Sparsity | ~94% |

### MovieLens 1M

| Property | Value |
|----------|-------|
| Users | 6,040 |
| Movies | 3,952 |
| Ratings | 1,000,209 |
| Rating scale | 1–5 |
| Sparsity | ~96% |

**Preprocessing** (as per paper Section 5.1):
- Users with fewer than 20 observed ratings are removed
- 80% of observed ratings used for training, 20% for testing

---

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/daksh22soni/RS_Research_Paper_Replication_ST-MMMF.git
cd RS_Research_Paper_Replication_ST-MMMF

# Install dependencies
pip install -r requirements.txt
```

### Data

The `data/` directory is included in this repository. Alternatively, download directly:

- [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

---

## Usage

### Exploratory Data Analysis

```bash
# EDA on MovieLens 100K
python src/movielens_100k_eda.py

# EDA on MovieLens 1M
python src/movielens_1m_eda.py
```

Both scripts perform a 10-step analysis covering:
1. File purpose understanding
2. Basic structure check
3. Missing value analysis
4. Rating distribution
5. User activity analysis
6. Movie popularity analysis
7. Genre distribution
8. Tag analysis (100K)
9. Links completeness
10. Join integrity check

All generated plots are saved to the working directory.

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d` | tuned | Latent dimension size |
| `λ` | {10^(i/16), i∈{1,5,...,40}} | Regularization parameter |
| `τ₁` | 49.99 | High-confidence shifting parameter |
| `τ₂` | 10 | Low-confidence shifting parameter |
| `s` | 100% | Sampling percentage per iteration |
| Max augmented/iter | 5,000 | Cap on new ratings per iteration |
| Iterations | 50 | Number of ST-MMMF rounds |
| Train/Test split | 80/20 | Standard split |

---

## Results

All results are pre-generated and available in the `outputs/` directory.

### Rating Distribution (Figure 5)
<img width="1634" height="551" alt="image" src="https://github.com/user-attachments/assets/6ed524c3-d37f-4403-86d5-46690a9df7d0" />

Both datasets exhibit highly imbalanced rating distributions, with ratings 3 and 4 dominating. The ST-MMMF augmentation strategy explicitly accounts for this imbalance using inverse-frequency weighting.

### Performance Over Augmentation Iterations (Figure 6)

<img width="1934" height="1331" alt="image" src="https://github.com/user-attachments/assets/7aba65b5-3f99-42dd-88e6-e1d33ed7317a" />


MAE and RMSE are tracked across 50 augmentation iterations for ST-MMMF and four baseline algorithms: SVD, NMF, SVD++, and Co-Clustering.

### Confusion Matrices — MovieLens 100K

| | Iteration 1 | Iteration 50 |
|---|---|---|
| **Training set** | <img width="1769" height="593" alt="image" src="https://github.com/user-attachments/assets/6dcb40bc-1605-485b-9b68-5fe5034c8272" /> | <img width="1769" height="593" alt="image" src="https://github.com/user-attachments/assets/3c6abeb3-068f-42ae-be12-2d85bc27c318" /> |
| **Test set** | <img width="1763" height="593" alt="image" src="https://github.com/user-attachments/assets/0df65ea5-aa38-484d-b55f-03f973d1784a" /> | <img width="1763" height="593" alt="image" src="https://github.com/user-attachments/assets/ffebd45d-02c6-4a27-851f-8322c39e0f8e" /> |

### Confusion Matrices — MovieLens 1M

| | Iteration 1 | Iteration 50 |
|---|---|---|
| **Training set** | <img width="1774" height="593" alt="image" src="https://github.com/user-attachments/assets/5ad43b2b-b715-4117-896c-a3952160d6ad" /> | <img width="1774" height="593" alt="image" src="https://github.com/user-attachments/assets/49269917-2685-4a06-883d-63347889f127" /> |
| **Test set** | <img width="1769" height="593" alt="image" src="https://github.com/user-attachments/assets/670f9cd4-4695-47bc-a23e-7169bb6ea716" /> | <img width="1769" height="593" alt="image" src="https://github.com/user-attachments/assets/4bf43bcc-d0eb-46a4-ad73-f5a5da27ca12" /> |

### Table 4 — Effect of Data Augmentation (MovieLens 100K)

| Iteration | # Observed | # Unobserved | # High-conf | # Augmented | # Overlap |
|-----------|------------|--------------|-------------|-------------|-----------|
|     1     |   74,296   |  1,511,830   |   571,554   |    4,998    |    N/A    |
|     2     |   74,440   |  1,511,686   |   915,251   |    4,997    |  571,554  |
|     3     |   76,911   |  1,509,215   |  1,145,425  |    4,997    |  915,251  |
|     4     |   80,524   |  1,505,602   |  1,275,230  |    4,998    | 1,145,425 |
|     5     |   84,717   |  1,501,409   |  1,346,426  |    4,998    | 1,275,230 |

The overlap column confirms **monotonicity**: entries predicted with high confidence in iteration *k* continue to be high-confidence in iteration *k+1*, satisfying the desirable property established by the paper.

### Baseline Performance — Iteration 50 (MAE / RMSE)

|  Dataset  | ST-MMMF |  SVD  |  NMF  | SVD++ | Co-Clustering |
|-----------|---------|-------|-------|-------|---------------|
|  100K MAE |  0.800  | 0.797 | 1.597 | 0.795 |     1.472     |
| 100K RMSE |  1.110  | 1.107 | 1.973 | 1.104 |     1.854     |
|   1M MAE  |  0.792  | 0.680 | 1.757 | 0.667 |     1.419     |
|  1M RMSE  |  1.096  | 0.963 | 2.138 | 0.952 |     1.806     |

Full per-iteration results are available in [`results/performance_results.csv`](results/performance_results.csv).

---

## References
[`ST-MMMF`]([document/Paper22.pdf](https://www.sciencedirect.com/science/article/abs/pii/S0957417423024697))
```bibtex
@article{shaikh2023stmmmf,
  title   = {Data augmentation and refinement for recommender system:
             A semi-supervised approach using maximum margin matrix factorization},
  author  = {Shaikh, Shamal and Kagita, Venkateswara Rao and
             Kumar, Vikas and Pujari, Arun K},
  journal = {arXiv preprint arXiv:2306.13050},
  year    = {2023}
}
```
---

## Acknowledgements

Datasets provided by [GroupLens Research](https://grouplens.org/datasets/movielens/) at the University of Minnesota.
