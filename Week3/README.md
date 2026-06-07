#  Customer Intelligence System
### End-to-End ML Pipeline: Classification, Ensemble Learning & Clustering

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Project Overview

This project is a **Week 3 Data Science Assignment** that builds a complete, end-to-end **Customer Intelligence System** using unsupervised and supervised machine learning on the [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) dataset.

**Business Context:**  
HELP International — a global humanitarian NGO — has raised **$10 million USD** to deploy aid to countries in direst need. As the data scientist, the goal is to:
- Segment 167 countries into development tiers using clustering
- Build ensemble classifiers to predict any country's tier
- Rank the most vulnerable countries for targeted aid allocation

---

##  Dataset

| Property | Details |
|---|---|
| Source | [Kaggle – rohan0301](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) |
| Rows | 167 countries |
| Features | 9 socio-economic & health indicators |

### Features Description

| Feature | Description |
|---|---|
| `child_mort` | Child mortality (under 5) per 1,000 live births |
| `exports` | Exports as % of GDP per capita |
| `health` | Total health spending as % of GDP per capita |
| `imports` | Imports as % of GDP per capita |
| `income` | Net income per person (USD) |
| `inflation` | Annual GDP growth rate (%) |
| `life_expec` | Average life expectancy (years) |
| `total_fer` | Total fertility rate |
| `gdpp` | GDP per capita (USD) |

---

##  ML Techniques Used

### 1. Unsupervised Clustering
| Algorithm | Purpose |
|---|---|
| **K-Means** (K=3) | Segment countries into 3 development tiers — Under-developed, Developing, Developed |
| **DBSCAN** | Density-based clustering to detect geopolitical outliers (e.g., Qatar, Singapore) |
| **PCA** | Reduce 9D features to 2D for visualisation |

### 2. Ensemble Classification
| Algorithm | Purpose |
|---|---|
| **Random Forest** | Bagging ensemble — 100–300 decision trees |
| **XGBoost** | Gradient boosting — sequential error correction |
| **GridSearchCV** | Automated hyperparameter tuning with 5-fold Stratified Cross-Validation |

---

##  Results

### Clustering Output (K-Means K=3)
| Tier | Countries | Avg Income | Avg Life Expectancy |
|---|---|---|---|
|  Under-developed | 47 | $3,942 | 59.2 yrs |
|  Developing | 84 | $12,306 | 72.8 yrs |
|  Developed | 36 | $45,672 | 80.1 yrs |

**DBSCAN Outliers:** Haiti, Luxembourg, Malta, Nigeria, Qatar, Singapore

### Classification Performance
| Model | Accuracy | F1-Macro (5-Fold CV) |
|---|---|---|
| Random Forest (Baseline) | 1.00 | 0.9640 |
| Random Forest (Tuned) | 1.00 | 0.9640 |
| XGBoost (Baseline) | 1.00 | 0.9489 |
| XGBoost (Tuned) | 1.00 | 0.9311 |

### Top 5 Countries — Humanitarian Aid Priority
| Rank | Country | Child Mortality | Life Expectancy | Needs Index |
|---|---|---|---|---|
| 1 | 🇭🇹 Haiti | 208 / 1000 | 32.1 yrs | **0.9895** |
| 2 | Central African Republic | 149 / 1000 | 47.5 yrs | 0.7764 |
| 3 | Sierra Leone | 160 / 1000 | 55.0 yrs | 0.7493 |
| 4 | Chad | 150 / 1000 | 56.5 yrs | 0.7091 |
| 5 | Mali | 137 / 1000 | 59.5 yrs | 0.6658 |

---

##  Repository Structure

```
Week3-Customer-Intelligence-System/
│
├── Week3_assignment.ipynb    ← Main Jupyter Notebook (submit this)
└── README.md                 ← This file
```

---

##  How to Run

### Option A — Run on Kaggle (Recommended)
1. Go to [Kaggle](https://www.kaggle.com) → **Create** → **New Notebook**
2. Import `Week3_assignment.ipynb`
3. Add dataset: search **"unsupervised-learning-on-country-data"** → Add
4. Click **Run All**

### Option B — Run Locally
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Open the notebook
jupyter notebook Week3_assignment.ipynb
```

> **Note:** The notebook auto-detects the environment. On Kaggle it loads the dataset from `/kaggle/input/`. Locally it fetches from a public GitHub mirror automatically.

---

## Notebook Structure (14 Sections, 49 Cells)

| # | Section | Description |
|---|---|---|
| 1 | Environment Setup | Library imports & version check |
| 2 | Styling | Dark-mode premium chart configuration |
| 3 | Data Loading | Auto-detect Kaggle / GitHub / synthetic fallback |
| 4 | EDA | Distributions, correlation heatmap, pair-plot |
| 5 | Preprocessing | StandardScaler normalisation |
| 6 | PCA | 2D dimensionality reduction + scree plot |
| 7 | K-Means | Elbow method, silhouette analysis, radar chart |
| 8 | DBSCAN | k-distance plot, outlier detection, comparison |
| 9 | Ensemble Setup | RF + XGBoost baseline training |
| 10 | Hyperparameter Tuning | GridSearchCV with StratifiedKFold |
| 11 | Model Evaluation | Confusion matrices, classification reports, feature importance |
| 12 | Aid Prioritisation | Composite Needs Index, Top-15 ranking |
| 13 | Cross-Validation | 5-Fold CV summary bar chart |
| 14 | Executive Summary | Strategic recommendations for HELP International |

---

## Libraries Used

```python
pandas          # Data manipulation
numpy           # Numerical computing
matplotlib      # Visualisation
seaborn         # Statistical plots
scikit-learn    # ML algorithms & preprocessing
xgboost         # Gradient boosting classifier
```

---

## Author

**Aadrika Sharma**  
Data Science Internship — Week 3 Assignment  
Dataset: [Kaggle – Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)
