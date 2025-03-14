# hebo_overfitting.py

import torch
import gpytorch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import yeojohnson

# Import your custom GP model here if needed:
# from your_module import CustomKernelGPModel

def check_overfitting(hebo_instance, test_x, test_y):
    """
    Performs overfitting checks for small datasets. This function:
      1. Compares Train vs Test LML
      2. Performs Leave-One-Out (LOO) cross-validation on a small dataset
      3. Checks average predictive variance (stddev)
      4. Computes Log Predictive Density (LPD)
      5. Computes MSE on the test set
      6. Approximates BIC

    :param hebo_instance: An instance of the HEBOOptimizer class, which has:
       - hebo_instance.gp (trained GPyTorch model)
       - hebo_instance.X, hebo_instance.y_raw, hebo_instance.y_transformed
       - A method: hebo_instance.gp.get_lml() for Train LML
    :param test_x: NumPy array of shape (N_test, D) for test inputs
    :param test_y: NumPy array of shape (N_test,) or (N_test, 1) for test targets
    """

    print("\n=== Overfitting Checks ===")

    # Convert test data to tensors
    test_x_t = torch.tensor(test_x, dtype=torch.float32)
    test_y_t = torch.tensor(test_y, dtype=torch.float32)

    # 1) Train vs. Test LML
    train_lml = hebo_instance.gp.get_lml()  # calls gp.get_lml()
    hebo_instance.gp.eval()
    hebo_instance.gp.likelihood.eval()

    with torch.no_grad():
        test_output = hebo_instance.gp(test_x_t)
        test_lml_val = hebo_instance.gp.likelihood(test_output).log_prob(test_y_t).sum()

    print(f"Train LML: {train_lml:.4f} | Test LML: {test_lml_val.item():.4f}")

    # 2) Leave-One-Out (LOO) Cross-Validation
    # Feasible only for very small data; it re-trains a mini-GP (slow for bigger sets).
    n = len(hebo_instance.y_raw)
    if n <= 20:  # arbitrary small threshold
        loo_lmls = []
        for i in range(n):
            # Mask out the i-th point
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            x_loo = torch.tensor(hebo_instance.X[mask], dtype=torch.float32)
            y_loo = torch.tensor(hebo_instance.y_transformed[mask], dtype=torch.float32)

            # (Re)create a minimal GP from scratch
            mini_gp = hebo_instance.__class__.gp.__new__(hebo_instance.gp.__class__)  # or directly instantiate your GP
            mini_gp = hebo_instance.gp.__class__(x_loo, y_loo, hebo_instance.likelihood)

            mini_gp.train()
            opt = torch.optim.Adam(mini_gp.parameters(), lr=0.05)
            mll_loo = gpytorch.mlls.ExactMarginalLogLikelihood(hebo_instance.likelihood, mini_gp)

            # Quick short training loop
            for _ in range(40):
                opt.zero_grad()
                out_loo = mini_gp(x_loo)
                loss_loo = -mll_loo(out_loo, y_loo)
                loss_loo.backward()
                opt.step()

            # Evaluate log probability for the left-out point
            mini_gp.eval()
            with torch.no_grad():
                x_i = torch.tensor(hebo_instance.X[i:i+1], dtype=torch.float32)
                y_i = torch.tensor([hebo_instance.y_transformed[i]], dtype=torch.float32)
                out_i = mini_gp(x_i)
                lml_i = mini_gp.likelihood(out_i).log_prob(y_i)
                loo_lmls.append(lml_i.item())

        avg_loo_lml = np.mean(loo_lmls)
        print(f"LOO Average LML (demo): {avg_loo_lml:.4f}")
    else:
        print("Skipping LOO-CV (dataset bigger than 20).")

    # 3) Predictive Variance Check on test set
    with torch.no_grad():
        test_pred = hebo_instance.gp(test_x_t)
        test_std = test_pred.stddev.numpy()
    avg_std = np.mean(test_std)
    print(f"Avg Predictive Std (test set): {avg_std:.4f}")

    # 4) Log Predictive Density (LPD) on test set
    lpd = test_lml_val.item() / len(test_y)
    print(f"Log Predictive Density (LPD) on test set: {lpd:.4f}")

    # 5) Mean Squared Error (MSE) on test set
    mse_test = mean_squared_error(test_y, test_pred.mean.detach().numpy())
    print(f"Test MSE: {mse_test:.4f}")

    # 6) Bayesian Information Criterion (BIC) ~ k*log(n) - 2*logLik
    # Here we use train_lml for logLik and approximate the number of parameters
    n_params = sum(p.numel() for p in hebo_instance.gp.parameters() if p.requires_grad)
    bic = n_params * np.log(n) - 2 * train_lml
    print(f"BIC (approx): {bic:.2f}")

    print("=== End Overfitting Checks ===\n")
