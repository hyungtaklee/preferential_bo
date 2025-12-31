# -*- coding: utf-8 -*-
# Original code Copyright (c) 2023 CyberAgent AI Lab
# Modifications Copyright (c) 2025 Hyungtak Lee
# Licensed under the MIT License.
import copy
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
from typing_extensions import TypedDict
from collections.abc import Callable
import numpy.typing as npt

import torch
import numpy as np
from scipy import optimize, special
from scipy.stats import norm, multivariate_normal, qmc
from tqdm import tqdm

import gpytorch

maxfeval = 3 * 1e2
jitter = 1e-8

class PrefGPTemplate:
    def __init__(self,
                 X: npt.ArrayLike,
                 kernel: Callable[[...], npt.ArrayLike],
                 kernel_bounds,
                 noise_std: float = 1e-2,
                ):
        self.input_dim: npt.ArrayLike = np.shape(X)[1] // 2
        self.num_duels: npt.ArrayLike = np.shape(X)[0]
        self.noise_std: float = noise_std
        self.kernel: Callable[[...], npt.ArrayLike] = kernel
        self.kernel_bounds = kernel_bounds

        self.X: npt.ArrayLike = X
        self.flatten_X: npt.ArrayLike = np.r_[X[:, :self.input_dim], X[:, self.input_dim:]]
        self.winner_idx: npt.ArrayLike = np.arange(self.num_duels)
        self.looser_idx: npt.ArrayLike = np.arange(self.num_duels, 2 * self.num_duels)


class PrefGPLaplace(PrefGPTemplate):
    r"""
    X_sort in $\RR^{#duels \times 2 input_dim}$: left side x is winner, right side x is looser

    Wei Chu and Zoubin Ghahramani. Preference learning with Gaussian processes. In Proceedings of the 22nd international conference on Machine learning, pages 137–144, 2005b.

    """
    def __init__(self,
                 X: npt.ArrayLike,
                 kernel: Callable[[...], npt.ArrayLike],
                 kernel_bounds,
                 noise_std: float = 1e-2,
                 seed: Optional[int] = None,
                 la_iteration: int = 100,
                ):
        super().__init__(X, kernel, kernel_bounds, noise_std)

        if np.unique(self.flatten_X, axis=1).shape[0] != (2 * self.num_duels):
            raise ValueError(
                "Input has duplicate duel points. The current gradient implementation "
                "requires unique points."
            )

        self.flatten_K_inv: Optional[npt.ArrayLike] = None
        self.hessian_indicator: npt.ArrayLike = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]
        self.initial_points_sampler: qmc.Sobol = qmc.Sobol(d=self.input_dim, seed=seed)
        self.la_iteration = la_iteration

    def inference(self) -> None:
        # For numerical stability, we add a small value at diagonal elements
        self.flatten_K = self.kernel(torch.tensor(self.flatten_X)).numpy() + jitter * np.eye(
            2 * self.num_duels
        )
        self.flatten_K_inv = np.linalg.inv(self.flatten_K)
        self.f_map, self.covariance_map, lambda_ = self.laplace_inference()
        self.lambda_inv = np.linalg.inv(lambda_ + jitter * np.eye(2 * self.num_duels))

    def inverse_mills_ratio(self, z: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Compute the inverse Mills ratio which is $r(z) = \text{pdf}(z) / \text{cdf}(z)$.
        """
        z_cdf = normcdf(z)
        cdf_nonzero_idx = z_cdf > 0
        # inverse mills ratio \approx -z if z <<<0
        inverse_mills_ratio = -z
        inverse_mills_ratio[cdf_nonzero_idx] = normpdf(z[cdf_nonzero_idx]) / (
            z_cdf[cdf_nonzero_idx]
        )
        return inverse_mills_ratio

    def laplace_inference(self) -> Tuple[npt.ArrayLike, npt.ArrayLike, Callable[[npt.ArrayLike, npt.ArrayLike], Tuple[npt.ArrayLike]]]:
        """
        Find the posterior's Laplace approximation using the Newton's method.
        """
        new_f_map: npt.ArrayLike = np.zeros(2 * self.num_duels) # new MAP estimate
        z: npt.ArrayLike = np.zeros(self.num_duels) # z_k's in the paper
        new_obj_val: npt.ArrayLike = np.inf

        for i in range(self.la_iteration):
            old_f_map = np.copy(new_f_map)
            old_obj_val = np.copy(new_obj_val)
            
            # Constructing dueling results
            z = (old_f_map[self.winner_idx] - old_f_map[self.looser_idx]) \
                / (np.sqrt(2) * self.noise_std)
            inverse_mills_ratio = self.inverse_mills_ratio(z)

            # Compute the gradient (the Newton-Raphson method)
            gradient = self._objective_gradient(old_f_map, inverse_mills_ratio)
            hessian = self._objective_hessian(z, inverse_mills_ratio)
            update = np.linalg.solve(hessian, np.c_[gradient]).ravel()

            # Update
            new_f_map = old_f_map - update
            new_obj_val = self._objective(new_f_map, z)
            if np.max(np.abs(new_f_map - old_f_map)) <= 1e-5:
                break

            # Needs to add error tolerance variable instead of int(5)
            if np.any(np.abs(new_f_map) > 5):
                return np.nan, np.nan, np.nan

            z_tmp = (new_f_map[self.winner_idx] - new_f_map[self.looser_idx]) \
                    / (np.sqrt(2) * self.noise_std)

        if i < 100:
            raise RuntimeError("Error in Laplace approximation may be large")
        f_map = np._c[new_f_map]
        # beta = self.flatten_K_inv @ f_map
        z = (f_map[self.winner_idx] - f_map[looser_idx]) \
            / (np.sqrt(2) * self.noise_std)
        inverse_mills_ratio: npt.ArrayLike = self.inverse_mills_ratio(z)
        
        lambda_ = self._lambda(z, inverse_mills_ratio)
        covariance_map = np.linalg.inv(self.flatten_K_inv + lambda_)

        return f_map, covariance_map, lambda_

    def _objective(self, f: npt.ArrayLike, z: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Objective for MAP $S(f) = -\sum_{k=1}^m \Phi(z_k) + \frac{1}{2} f^\top \Sigma^{-1} f$.
        """
        s_f = -np.sum(norm.logcdf(z)) + (np.c_[f].T @ self.flatten_K_inv @ np.c_[f] / 2.0)

        return s_f.ravel()

    def _objective_gradient(self, f: npt.ArrayLike, inverse_mills_ratio: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Compute the gradient of $S(f)$. $\frac{\partial S(f)}{\partial f}$.
        """
        tmp_grad = inverse_mills_ratio / (np.sqrt(2) * self.noise_std)
        first_term_grad = np.r_[-1 * tmp_grad, tmp_grad]
        second_term_grad = self.flatten_K_inv @ np.c_[f]

        return first_term_grad + second_term_grad.ravel()
    
    def _objective_hessian(self, z: npt.ArrayLike, inverse_mills_ratio: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Compute Hessian matrix of the posterior, $\Sigma^-1 + \Lambda$,
        where $\Sigma$ is the covariance of the prior.
        """
        lambda_: npt.ArrayLike = self._lambda(z, inverse_mills_ratio)

        return self.flatten_K_inv + lambda_

    def _lambda(self, z: npt.ArrayLike, inverse_mills_ratio: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Compute $\Lambda$ matrix which is coming from the probit likelihood.
        """
        tmp_hessian = (inverse_mills_ratio ** 2 + z * inverse_mills_ratio) \
                      / (2 * self.noise_std ** 2)
        hessian_first_term = (
            self.hessian_indicator * np.c_[np.r_[tmp_hessian, tmp_hessian]]
        )
        return hessian_first_term

    def predict(self, X: npt.ArrayLike, full_cov=False) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        cov_train_test = self.kernel(self.flatten_X, X)
        posterior_mean = cov_train_test.T @ self.flatten_K_inv @ self.f_map

        if full_cov:
            cov_test = self.kernel(X, X)
            posterior_cov = (
                cov_test
                - cov_train_test.T
                @ np.linalg.inv(self.flatten_K + self.lambda_inv)
                @ cov_train_test
            )
            return posterior_mean, posterior_cov
        else:
            var_test = self.kernel.Kdiag(X)
            posterior_var = var_test - np.einsum(
                "ij,jk,ki->i",
                cov_train_test.T,
                np.linalg.inv(self.flatten_K + self.lambda_inv),
                cov_train_test,
            )

            return posterior_mean, np.c_[posterior_var]

    def negative_log_likelihood(self, kernel_params: Optional[npt.ArrayLike] = None) -> npt.ArrayLike:
        if kernel_params is None:
            kernel = copy.copy(self.kernel)
        else:
            kernel_params = np.atleast_2d(kernel_params)
            kernel = GPy.kern.RBF(
                input_dim=self.input_dim, lengthscale=kernel_params, ARD=True
            )
        # if self.flatten_K_inv is not None:
        #     copy_flatten_K_inv = copy.copy(self.flatten_K_inv)
        # else:
        #     copy_flatten_K_inv = None

        flatten_K = kernel.K(self.flatten_X) + jitter * np.eye(2 * self.num_duels)
        # flatten_K = kernel.K(self.flatten_X)
        self.flatten_K_inv = np.linalg.inv(flatten_K)
        f_map, _, Lambda = self.Laplace_inference()

        if np.any(np.isnan(f_map)):
            return np.nan

        z = (f_map[self.winner_idx] - f_map[self.looser_idx]) / (
            np.sqrt(2) * self.noise_std
        )
        # inverse_mills_ratio = self.inverse_mills_ratio(z)
        flatten_K_Lambda = flatten_K @ Lambda
        minus_log_likelihood = (
            self._objective(f_map, z)
            + 0.5 * np.log(np.linalg.det(flatten_K_Lambda + np.eye(2 * self.num_duels)))
        )[0]

        return minus_log_likelihood

    def model_selection(self, num_start_points: int = 10):
        """
        Select model hyperparameters and update self.kernel
        """
        x = self.initial_points_sampler.random(n=num_start_points - 1) * (
            self.kernel_bounds[1] - self.kernel_bounds[0]
        )
        x = np.r_[x, np.atleast_2d(self.kernel.lengthscale.values)]

        def wrapper_negative_log_likelihood(kernel_params) -> None:
            return self.negative_log_likelihood(kernel_params=kernel_params)

        func_values = list()
        for i in range(np.shape(x)[0]):
            res = optimize.minimize(
                wrapper_negative_log_likelihood,
                x0=x[i],
                bounds=np.array(self.kernel_bounds).T.tolist(),
                method="L-BFGS-B",
                options={"ftol": 0.1},
                jac="2-point",
            )
            func_values.append(res["fun"])
            x[i] = res["x"]

        min_index = np.argmin(func_values)
        print("Selected kernel lengthscales: {}".format(x[min_index]))
        self.kernel = GPy.kern.RBF(
            input_dim=self.input_dim, lengthscale=x[min_index], ARD=True
        )
        # self.inference()

    def sample(self, sample_size: int = 1) -> None:
        self.RFF = RFF_RBF(
            input_dim=self.input_dim, lengthscales=self.kernel.lengthscale.values
        )
        self.coefficient = np.random.randn(self.RFF.basis_dim, sample_size) # noise

        f_sample = (
            np.linalg.cholesky(self.covariance_map)
            @ np.random.randn(2 * self.num_duels, sample_size)
            + self.f_map
        )
        flatten_X_transform = self.RFF.transform(self.flatten_X)
        f_prior_flattenX = flatten_X_transform @ self.coefficient

        self.K_inv_f_sample = self.flatten_K_inv @ (f_sample - f_prior_flattenX)
        pass

    def evaluate_sample(self, X: npt.ArrayLike) -> npt.ArrayLike:
        K_X_flattenX = self.kernel.K(X, self.flatten_X)

        X_transform = self.RFF.transform(X)
        f_X_samples_prior = X_transform @ self.coefficient
        f_X_samples = f_X_samples_prior + K_X_flattenX @ self.K_inv_f_sample
        return f_X_samples

    def add_data(self, X_win: npt.ArrayLike, X_loo: npt.ArrayLike) -> None:
        X_win = np.atleast_2d(X_win)
        X_loo = np.atleast_2d(X_loo)
        assert np.shape(X_win) == np.shape(
            X_loo
        ), "Shapes of winner and looser in added data do not match"
        self.X = np.r_[self.X, np.c_[X_win, X_loo]]
        self.num_duels = self.num_duels + np.shape(X_win)[0]
        self.flatten_X = np.r_[self.X[:, : self.input_dim], self.X[:, self.input_dim :]]
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]

        if np.shape(np.unique(self.flatten_X, axis=1))[0] == 2 * self.num_duels:
            raise RumtimeError("Input has same duel points, so the current implementation for gradient of objective cannot be considered correctly")
 

# Utility functions
SQRT_TWO = np.sqrt(2)

def normal_cdf_inverse(z: npt.ArrayLike) -> npt.ArrayLike:
    return special.erfinv(2 * z - 1) * SQRT_TWO


def normcdf(x: npt.ArrayLike) -> npt.ArrayLike:
    return 0.5 * (1 + special.erf(x / SQRT_TWO))


def normpdf(x: npt.ArrayLike) -> npt.ArrayLike:
    pdf = np.zeros(np.shape(x))
    small_x_idx = np.abs(x) < 50
    pdf[small_x_idx] = np.exp(-x[small_x_idx] ** 2 / 2) / (np.sqrt(2 * np.pi))
    return pdf


def RBF_ARD_kernel_gradient(K: npt.ArrayLike, X: npt.ArrayLike, lengthscale: npt.ArrayLike) -> npt.ArrayLike:
    """
    ARD RBF kernel in Chu and Gharhamani 2005.
    """
    N = np.shape(K)[0]
    d = np.shape(X)[1]
    X1 = X.reshape(N, 1, d)
    X2 = X.reshape(1, N, d)
    # N times N times d
    one_dimensional_squared_scaled_distance = (X1 - X2) ** 2 / lengthscale.reshape(
        1, 1, d
    ) ** 3
    return K.reshape(N, N, 1) * one_dimensional_squared_scaled_distance


if __name__ == "__main__":
    pass
