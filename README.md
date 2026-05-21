# Week 1 Assignment — ML Foundations

**Author:** Aadrika Sharma  
**Submission:** `week1_Aadrika_Sharma.ipynb`

---




---

## Topics Covered

| Part | Topic |
|------|-------|
| Part 1 | Python Fundamentals |
| Part 2 | NumPy |
| Part 3 | Pandas |
| Part 4 | Linear Algebra |
| Part 5 | Statistics |
| Part 6 | Probability Theory |

---

##  Part Breakdown

### Part 1 — Python Fundamentals

- **1.1 Data Types & Control Flow** — `classify_number(n)` function using conditional branching
- **1.2 Data Structures** — Word frequency dict, unique words set, long words via list comprehension
- **1.3 Exceptions & Types** — `safe_divide(a, b)` with `ZeroDivisionError` and `TypeError` handling
- **1.4 Functions & Lambdas** — `apply_twice(f, x)` with lambda `triple`

---

### Part 2 — NumPy

- **2.1 Array Creation & Shapes** — 1D, 2D (3×4), and 3D (2×2×3) reshaping with `np.arange`
- **2.2 Indexing & Slicing** — Row, column, submatrix extraction; boolean indexing
- **2.3 Operations & Dot Product** — Element-wise vs matrix multiplication; scalar operations; dot product

---

### Part 3 — Pandas

Working with a 10-row employee dataset (`Engineering`, `Marketing`, `HR` departments) with intentional missing values.

- **3.1 DataFrames vs Series** — Column extraction as Series and DataFrame
- **3.2 `iloc` & `loc`** — Positional and label-based indexing
- **3.3 Filtering & Group By** — Senior engineers filter; department-wise mean salary & performance
- **3.4 Handling Missing Data** — Median imputation for salary, mean imputation for age, row drop for missing performance

---

### Part 4 — Linear Algebra

- **4.1 Vectors & Matrices** — L2 norm computation; vector arrow plot using `matplotlib`
- **4.2 Matrix Operations** — Addition, scalar multiplication, matrix multiplication; non-commutativity proof
- **4.3 Eigenvalues & Eigenvectors** — `np.linalg.eig`; verification of Av = λv; eigenvalue-scaled arrow plot
- **4.4 SVD & Dimensionality Reduction** — Full SVD decomposition; reconstruction verification; rank-1 approximation

---

### Part 5 — Statistics

- **5.1 Descriptive Statistics** — Mean, median, std, range, IQR for salary; histogram + KDE plot
- **5.2 Hypothesis Testing** — One-sample t-test; Pearson correlation between salary and experience
- **5.3 Error Metrics** — MAE, MSE, RMSE, R², Adjusted R² implemented from scratch (no sklearn)
- **5.4 Distribution Testing & Stationarity** — KS test; ADF test on non-stationary series; differencing for stationarity
- **5.5 Model Monitoring** — PSI computation; distribution shift simulation and plot

---

### Part 6 — Probability Theory

- **6.1 Core Concepts** — Marble bag probability; joint & conditional probability; independence check
- **6.3 Bayes' Theorem** — Spam filter posterior; `naive_bayes_predict` implementation
- **6.4 Central Limit Theorem** — Sampling from exponential distribution; CLT verification with KS test

---

