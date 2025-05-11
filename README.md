# Bayesian Optimisation 

## 📌 Project Overview

This project was developed for a **Bayesian Optimisation (BO)**, where the task was to **optimise eight unknown black-box functions** — each varying in **dimensionality and real-world analogy**, ranging from contamination detection to high-dimensional hyperparameter tuning.

Each function could only be queried once every few days, simulating real-world settings where evaluations are expensive or limited. The challenge was to design highly **informative, data-efficient optimisation strategies** using Bayesian methods.

---

## 🔍 Problem Breakdown

Each function reflected a specific industrial or scientific setting:


- [**Function 1:** 2D – *Radiation/Contamination Source Detection*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f1_optimisation.ipynb)
- [**Function 2:** 2D – *Noisy Machine Learning Simulator*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f2_optimisation.ipynb)
- [**Function 3:** 3D – *Drug Discovery Optimisation*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f3_optimisation.ipynb)
- [**Function 4:** 4D – *Fast but Inaccurate Business Model*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f4_optimisation.ipynb)
- [**Function 5:** 4D – *Chemical Reaction Yield*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f5_optimisation.ipynb)
- [**Function 6:** 5D – *Cake Recipe (Multi-Objective Composite)*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f6_optimisation.ipynb)
- [**Function 7:** 6D – *Hyperparameter Tuning for ML Models*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f7_optimisation.ipynb)
- [**Function 8:** 8D – *High-Dimensional Black-Box Search*](https://github.com/Alerdo/BlackBox-Bayesian_Optimisation/blob/master/initial_data/0.0%20notebooks/f8_optimisation.ipynb)


Initial datasets were provided in `.npy` format and used to seed initial model fitting.

---

## 📚 Research-Informed Strategy

This project was strongly informed by state-of-the-art literature:

- **HEBO** (Heteroscedastic Evolutionary Bayesian Optimisation) – Huawei, NeurIPS 2020  
- **Warped Gaussian Processes** – Snelson, Rasmussen & Ghahramani (NIPS 2003)  
- **Combining Transformations and Acquisitions** – arXiv:2206.03301

These guided:
- Output warping techniques for Gaussianity  
- Noise modelling  
- Kernel composition and parameterisation  
- Acquisition blending (UCB, EI, PI) using multi-objective optimisation  

---

## 🛠️ Tools & Technologies

- **Language:** Python 3.10+
- **Core Libraries:** `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pymoo`
- **Model:** `GaussianProcessRegressor` with:
  - RBF, Matern, RationalQuadratic, Constant, and White kernels
- **Optimisation:** NSGA-II via `pymoo` for acquisition function blending
- **Exploration Techniques:**
  - ε-greedy exploration  
  - Dynamic bound restriction
- **Diagnostics:**
  - PCA for feature space reduction  
  - Random Forests for feature importance  

---

## ⚙️ End-to-End Strategy

1. **Problem Understanding**  
   Reviewed function context, set bounds based on expected input ranges and units.

2. **Exploratory Data Analysis (EDA)**  
   Outlier detection, distribution plots, missing-value checks, and signal-to-noise estimation.

3. **Transformations**  
   Output: Yeo–Johnson, Box–Cox (shifted if needed)  
   Input: Kumaraswamy wrapping or MinMax scaling

4. **Kernel Design**  
   - RBF for smooth surfaces  
   - Matern 1.5 for moderate nonlinearity  
   - RationalQuadratic for multi-scale variation  
   - WhiteKernel for observational noise

5. **Composite Model Structure**  
   Final GP = Constant × (RBF + Matern) + White  
   Acquisition = MACE routine (EI + PI + UCB)  
   Search augmented with ε-random exploration

6. **Iteration Loop**  
   Fit GP → optimise acquisition → submit next point → update model  
   Log predictive mean, variance, and uncertainty

7. **Visual Monitoring**  
   Used PCA to reduce dimensionality and visualise GP mean, CI, and next query

---

## 🧠 Highlights & Key Insights

- **Modularity and discipline** were crucial — a reusable pipeline allowed fast iteration and consistent diagnostics across all eight functions.
- **Dynamic bound restriction** and **feature filtering** helped significantly in higher-dimensional functions (Functions 7 & 8).
- In noisy landscapes, **WhiteKernel tuning** and **output warping** prevented overfitting and improved model generalisation.
- **Transformation choice** (especially shift + Box–Cox vs. raw) directly impacted model gradient behaviour and acquisition effectiveness.
- **Acquisition blending** (via MACE/NSGA-II) was a powerful late-stage tool to balance global search with targeted refinement.

---

## ✅ Results

- Delivered strong performance across multiple functions  
- Achieved the **best score in one function** (Function 6)  
- Applied cutting-edge BO techniques in a real-world simulation environment

---

I'm always available to discuss, collaborate, and connect. Feel free to reach out through my Portfolio or LinkedIn.

<p align="center">
    <a href="https://alerdo-ballabani.co.uk/" target="_blank">
        <img src="https://img.shields.io/badge/Portfolio-Visit_My_Website-blue?style=for-the-badge" alt="Portfolio">
    </a>
    <a href="https://www.linkedin.com/in/alerdo-ballabani-450a85283/" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-Connect_With_Me-blue?style=for-the-badge" alt="LinkedIn">
    </a>
</p>

---

Feel free to connect!

