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

class PrefGP:
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


class PrefGPLaplace(PrefGP):
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
        r"""
        Find the posterior's MAP using the Newton's method.
        Since $S(f)$ is shown to be a convex programming problem, $f_\text{MAP}$ is $S'(f_\text{MAP}) = 0$
        Hence we compute:
        $f_{n + 1} = f_{n} - f'(f_{n})/f''(f_{n}),
        where we need a gradient and Hessian of S() wrt f.
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
            z_new = (new_f_map[self.winner_idx] - new_f_map[self.looser_idx]) \
                     / (np.sqrt(2) * self.noise_std)

            new_obj_val = self._objective(new_f_map, z_new)
            if np.max(np.abs(new_f_map - old_f_map)) <= 1e-5:
                break

            # Needs to add error tolerance variable instead of int(5)
            if np.any(np.abs(new_f_map) > 5):
                return np.nan, np.nan, np.nan

        if i >= self.la_iteration:
            raise RuntimeError("Error in Laplace approximation may be large")
        f_map = np.c_[new_f_map]
        # beta = self.flatten_K_inv @ f_map
        z = (f_map[self.winner_idx] - f_map[self.looser_idx]) \
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
        cov_train_test = self.kernel(torch.tensor(self.flatten_X), torch.tensor(X)).detach().numpy()
        posterior_mean = cov_train_test.T @ self.flatten_K_inv @ self.f_map

        if full_cov:
            cov_test = self.kernel(torch.tensor(X), torch.tensor(X)).detach().numpy()
            posterior_cov = (
                cov_test
                - cov_train_test.T
                @ np.linalg.inv(self.flatten_K + self.lambda_inv)
                @ cov_train_test
            )
            return posterior_mean, posterior_cov
        else:
            var_test = self.kernel(X, diag=True).detach().numpy()
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
            input_dim=self.input_dim, lengthscales=self.kernel.lengthscale.numpy()
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
 

class PrefGPEP(PrepGP):
    """
    X_sort in \mathbb{R}^{# duels \times 2 input_dim}: left side x is winner, right side x is looser.
    """
    def __init__(self, 
                X: npt.ArrayLike,
                kernel,
                kernel_bounds,
                noise_std=1e-2,
                seed: Optional[int] = None,
                ):
        super().__init__(X, kernel, kernel_bounds, noise_std)

        self.input_dim = np.shape(X)[1] // 2
        self.num_duels = np.shape(X)[0]
        self.noise_std = noise_std
        self.kernel = kernel
        self.kernel_bounds = kernel_bounds

        # for LP inference
        self.winner_idx = np.arange(self.num_duels)
        self.looser_idx = np.arange(self.num_duels, 2 * self.num_duels)
        self.hessian_indicator = np.r_[
            np.c_[np.eye(self.num_duels), -1 * np.eye(self.num_duels)],
            np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)],
        ]
        self.flatten_K_inv = None

        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]
        self.X = X
        self.flatten_X = np.r_[X[:, : self.input_dim], X[:, self.input_dim :]]
        self.initial_points_sampler = qmc.Sobol(d=self.input_dim, seed=0)

    def inference(self, sample_size=None):
        self.flatten_K = self.kernel(torch.tensor(self.flatten_X)).detach().numpy() + self.noise_std ** 2 + np.eye(2 * self.num_duels)

        # Define the prior for the duels v ~ GP(0, K_v)
        self.K_v = self.A @ self.flatten_K @ self.A.T
        self.K_v_inv = np.linalg.inv(self.K_v)

        # Expectation propagation loop
        # Approximated posterior with Gaussian N(mu_TN, sigma_TN)
        # Approximated likelihood with Gaussian N(mu_tilde, sigma_tilde)
        self.mu_TN, self.sigma_TN, self.mu_tilde, self.sigma_tilde = ep_orthants_tmvn(
            upper=np.zeros(self.num_duels),
            mu=np.zeros(self.num_duels),
            sigma=self.K_v,
            L=self.num_duels,
        )
        # Updated posterior for prediction (posterior prediction distribution)
        self.sigma_plus_sigma_tilde = self.K_v + np.diag(self.sigma_tilde)
        self.sigma_plus_sigma_tilde_inv = np.linalg.inv(self.sigma_plus_sigma_tilde)
        self.sigma_plus_sigma_tilde_inv_mu_tilde = (
            self.sigma_plus_sigma_tilde_inv.T @ self.mu_tilde
        )

    def predict(self, X, full_cov=False):
        X = np.atleast_2d(X)
        test_point_size = np.shape(X)[0]
        transform_matrix = np.r_[
            np.c_[np.eye(test_point_size), np.zeros((test_point_size, self.num_duels))],
            np.c_[np.zeros((2 * self.num_duels, test_point_size)), self.A.T],
        ]

        cov_X_flattenX = self.kernel(torch.tensor(X), torch.tensor(self.flatten_X)).detach().numpy()
        K_X_flattenX = np.c_[
            np.r_[
                self.kernel(torch.tensor(X)).detach().numpy() + jitter * np.eye(test_point_size), cov_X_flattenX.T
            ],
            np.r_[cov_X_flattenX, self.flatten_K],
        ]

        K_X_v = (
            transform_matrix.T[test_point_size:, :]
            @ K_X_flattenX
            @ transform_matrix[:, :test_point_size]
        )

        tmp = K_X_v.T @ self.sigma_plus_sigma_tilde_inv
        mean = K_X_v.T @ self.sigma_plus_sigma_tilde_inv_mu_tilde
        if full_cov:
            cov = self.kernel(torch.tensor(X)).detach().numpy() - tmp @ K_X_v
            
            return mean, cov
        
        else:
            var = self.kernel(torch.tensor(X)).diagonal(dim1=-1, dim2=-2).detach().numpy()- np.einsum("ij,ji->i", tmp, K_X_v)

            return mean, var

    import numpy as np

    def minus_log_likelihood_ep(self, kernel_params):
        dtype = torch.float64

        X = torch.as_tensor(self.flatten_X, dtype=dtype, device=device)         # (2*num_duels, input_dim)
        A = torch.as_tensor(self.A, dtype=dtype, device=device)                 # (num_duels, 2*num_duels)

        # ---- GPyTorch RBF kernel (ARD == per-dimension lengthscales) ----
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim).to(device=device, dtype=dtype)

        # kernel_params can be scalar or vector; shape to [1,1,D] for ARD
        ls = torch.as_tensor(kernel_params, dtype=dtype, device=device)
        if ls.ndim == 0:
            ls = ls.view(1, 1, 1)                     # scalar lengthscale
        else:
            ls = ls.view(1, 1, -1)                    # ARD lengthscales, D == input_dim
        kernel.lengthscale = ls                       # setter handles raw parameter transform

        # ---- kernel matrix (dense) + noise ----
        K = kernel(X, X).evaluate()                   # (2*num_duels, 2*num_duels) torch tensor
        flatten_K = K + (self.noise_std ** 2) * torch.eye(2 * self.num_duels, dtype=dtype, device=device)

        K_v = A @ flatten_K @ A.T                     # torch tensor

        K_v = K_v.detach().cpu().numpy()
        # K_v_inv = np.linalg.inv(K_v)

        _, sigma_TN, mu_tilde, sigma_tilde = ep_orthants_tmvn(
            upper=np.zeros(self.num_duels),
            mu=np.zeros(self.num_duels),
            sigma=K_v_np,
            L=self.num_duels,
        )
        # sigma_plus_sigma_tilde = K_v + np.diag(sigma_tilde)
        # sigma_plus_sigma_tilde_inv = np.linalg.inv(sigma_plus_sigma_tilde)
        # sigma_plus_sigma_tilde_inv_mu_tilde = sigma_plus_sigma_tilde_inv.T @ mu_tilde
        ##########################################################
        tmp_vec = mu_tilde / sigma_tilde

        _, original_logdet = np.linalg.slogdet(K_v_np)
        _, ep_logdet = np.linalg.slogdet(sigma_TN)
        log_likelihood = 0.5 * (
            -original_logdet
            + ep_logdet
            + np.c_[tmp_vec].T @ sigma_TN @ np.c_[tmp_vec]
            - np.c_[tmp_vec].T @ np.c_[mu_tilde]
        )

        return -log_likelihood

    def sample(self, sample_size=1):
        self.RFF = RFF_RBF(
            input_dim=self.input_dim, lengthscales=self.kernel.lengthscale.numpy()
        )
        self.coefficient = np.random.randn(self.RFF.basis_dim, sample_size)

        v_sample = (
            np.linalg.cholesky(self.sigma_TN)
            @ np.random.randn(self.num_duels, sample_size)
            + self.mu_TN
        )
        flattenX_transform = self.RFF.transform(self.flatten_X)
        f_prior_flattenX = flattenX_transform @ self.coefficient
        f_prior_v = self.A @ f_prior_flattenX

        self.K_inv_f_sample = self.K_v_inv @ (v_sample - f_prior_v)
        pass

    def evaluate_sample(self, X):
        X = np.atleast_2d(X)
        test_point_size = np.shape(X)[0]
        transform_matrix = np.r_[
            np.c_[np.eye(test_point_size), np.zeros((test_point_size, self.num_duels))],
            np.c_[np.zeros((2 * self.num_duels, test_point_size)), self.A.T],
        ]

        cov_X_flattenX = self.kernel.K(X, self.flatten_X)
        K_X_flattenX = np.c_[
            np.r_[
                self.kernel.K(X) + jitter * np.eye(test_point_size), cov_X_flattenX.T
            ],
            np.r_[cov_X_flattenX, self.flatten_K],
        ]

        K_X_v = (
            transform_matrix.T[test_point_size:, :]
            @ K_X_flattenX
            @ transform_matrix[:, :test_point_size]
        )

        X_transform = self.RFF.transform(X)
        f_X_samples_prior = X_transform @ self.coefficient
        f_X_samples = f_X_samples_prior + K_X_v @ self.K_v_inv_v_sample

        return f_X_samples

    def add_data(self, X_win, X_loo):
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
        self.A = np.c_[-1 * np.eye(self.num_duels), np.eye(self.num_duels)]

        assert (
            np.shape(np.unique(self.flatten_X, axis=1))[0] == 2 * self.num_duels
        ), "Input has same duel points, so the current implementation for gradient of objective cannot be considered correctly"


    def ep_orthants_tmvn(self, upper, mu, sigma, L):
        """
        return means and covariance matrices of truncated multi-variate normal distribution truncated with truncation < upper.

        Parameters
        ----------
        mu : numpy array
            mean of original L-dimentional multi-variate nomal (L)
        sigma : numpy array
            covariance matrix of L-dimentional original multi-variate normal (L \times L)
        upper : numpy array
            upper position of M cells in region truncated by pareto frontier (L)

        Returns
        -------
        mu_TN : numpy array
            means of truncated multi-variate normal approximated by EP (L)
        sigma_TN : numpy array
            covariance matrix of truncated multi-variate normal approximated by EP (L \times L)
        mu_tilde:
        sigma_tilde: 
        """
        mu_tilde = np.zeros(L)
        sigma_tilde = np.inf * np.ones(L)
        mu_TN = mu
        sigma_TN = sigma
        mu_TN_before = mu
        sigma_TN_before = sigma

        sigma_inv = np.linalg.inv(sigma)
        sigma_inv_mu = sigma_inv.dot(mu)

        for i in range(1000):
            for j in range(L):
                # Remove site distribution (sigma_tilde) from the marginal distribution
                # producing cavity distribtion (mu_bar, sigma_bar)
                sigma_bar = 1.0 / (1.0 / sigma_TN[j, j] - 1.0 / sigma_tilde[j])
                mu_bar = sigma_bar * (
                    mu_TN[j] / sigma_TN[j, j] - mu_tilde[j] / sigma_tilde[j]
                )

                # Tilted distribution
                beta = (upper[j] - mu_bar) / np.sqrt(sigma_bar) # standardization
                Z = normcdf(beta) # 
                beta_pdf = normpdf(beta)
                diff_pdf = -beta_pdf
                diff_pdf_product = -beta * beta_pdf

                gamma = diff_pdf_product / Z - (diff_pdf / Z) ** 2
                if gamma == 0:
                    # If gamma = 0, we can interprete that there is no condition
                    sigma_tilde[j] = np.inf
                    mu_tilde[j] = 0
                else:
                    sigma_tilde[j] = -(1.0 / gamma + 1) * sigma_bar
                    mu_tilde[j] = mu_bar - 1.0 / gamma * (diff_pdf / Z) * np.sqrt(sigma_bar)

                sigma_tilde_inv = np.diag(1.0 / sigma_tilde)
                sigma_TN = np.linalg.inv(sigma_tilde_inv + sigma_inv)
                mu_TN = sigma_TN.dot(sigma_tilde_inv.dot(mu_tilde) + sigma_inv_mu)

            change = np.max(
                [
                    np.max(np.abs(sigma_TN - sigma_TN_before)),
                    np.max(np.abs(mu_TN - mu_TN_before)),
                ]
            )
            sigma_TN_before = sigma_TN
            mu_TN_before = mu_TN

            if np.isnan(change):
                print("iteration :", i)
                print("mu", mu)
                print("sigma", sigma)
                print("upper", upper)
                print("mu_TN", mu_TN)
                print("sigma_TN", sigma_TN)
                print(gamma)
                exit()

            if change < 1e-8:
                # print('iteration :', i)
                break

        return mu_TN, sigma_TN, mu_tilde, sigma_tilde


################################################################################
# Utility functions and classes
################################################################################
# Constants
SQRT_TWO = np.sqrt(2)
A_ZERO = 0.2570


def orthants_mvn_gibbs_sampling(
    dim, cov_inv, burn_in=500, thinning=1, sample_size=1000, initial_sample=None
):
    if initial_sample is None:
        sample_chain = np.zeros((dim, 1))
    else:
        if initial_sample.shape != (dim, 1):
            raise ValueError(f"Expected shape ({(dim, 1)}), but got {initial_sample.shape}")
        current_sample = initial_sample.copy()

    conditional_std = 1 / np.sqrt(np.diag(cov_inv))
    scaled_cov_inv = cov_inv / np.c_[np.diag(cov_inv)]
    sample_list = []
    for i in range((burn_in + thinning * (sample_size - 1)) * dim):
        j = i % dim
        conditional_mean = sample_chain[j] - scaled_cov_inv[j] @ sample_chain
        sample_chain[j] = (
            -1
            * one_side_trunc_norm_sampling(
                lower=conditional_mean[0] / conditional_std[j]
            )
            * conditional_std[j]
            + conditional_mean[0]
        )

        if ((i + 1) - burn_in * dim) % (
            dim * thinning
        ) == 0 and i + 1 - burn_in * dim >= 0:
            sample_list.append(sample_chain.copy())

    samples = np.hstack(sample_list)

    return samples


def trunc_norm_sampling(lower=None, upper=None, mean=0, std=1):
    """
    See Sec.2.1 in YIFANG LI AND SUJIT K. GHOSH Efficient Sampling Methods for Truncated Multivariate Normal and Student-t Distributions Subject to Linear Inequality Constraints, Journal of Statistical Theory and Practice, 9:712–732, 2015
            Christian P Robert. Simulation of truncated normal variables. Statistics and computing, 5(2):121–125, 1995.
    """
    if lower is None and upper is None:
        return np.random.randn(1) * std + mean
    elif lower is None:
        upper = (upper - mean) / std
        return -1 * one_side_trunc_norm_sampling(lower=-upper) * std + mean
    elif upper is None:
        lower = (lower - mean) / std
        return one_side_trunc_norm_sampling(lower=lower) * std + mean
    elif lower <= 0 and 0 < upper:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            two_sided_trunc_norm_sampling_zero_containing(lower=lower, upper=upper)
            * std
            + mean
        )
    elif 0 <= lower:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            two_sided_trunc_norm_sampling_positive_lower(lower=lower, upper=upper) * std
            + mean
        )
    elif upper <= 0:
        lower = (lower - mean) / std
        upper = (upper - mean) / std
        return (
            -1
            * two_sided_trunc_norm_sampling_positive_lower(lower=-upper, upper=-lower)
            * std
            + mean
        )


def one_side_trunc_norm_sampling(lower=None):
    if lower > A_ZERO:
        alpha = (lower + np.sqrt(lower**2 + 4)) / 2.0
        while True:
            z = np.random.exponential(alpha) + lower
            rho_z = np.exp(-((z - alpha) ** 2) / 2.0)
            u = np.random.rand(1)
            if u <= rho_z:
                return z
    elif lower >= 0:
        while True:
            z = np.abs(np.random.randn(1))
            if lower <= z:
                return z
    else:
        while True:
            z = np.random.randn(1)
            if lower <= z:
                return z


def two_sided_trunc_norm_sampling_zero_containing(lower, upper):
    if upper <= lower * np.sqrt(2 * np.pi):
        M = 1.0 / np.sqrt(
            2 * np.pi
        )  # / (normcdf(upper) - normcdf(lower)) * (upper - lower)
        while True:
            z = np.random.rand(1) * (upper - lower) + lower
            u = np.random.rand(1)
            if u <= normpdf(z) / M:
                return z
    else:
        while True:
            z = np.random.randn(1)
            if lower <= z and z <= upper:
                return z


def two_sided_trunc_norm_sampling_positive_lower(lower, upper):
    if lower < A_ZERO:
        b_1_a = lower + np.sqrt(np.pi / 2.0) * np.exp(lower**2 / 2.0)
        if upper <= b_1_a:
            M = normpdf(
                lower
            )  # / (normcdf(upper) - normcdf(lower)) # * (upper - lower)
            while True:
                z = np.random.rand(1) * (upper - lower) + lower
                u = np.random.rand(1)
                if u <= normpdf(z) / M:
                    return z
        else:
            while True:
                z = np.abs(np.random.randn(1))
                if lower <= z and z <= upper:
                    return z
    else:
        tmp = np.sqrt(lower**2 + 4)
        b_2_a = lower + 2 / (lower + tmp) * np.exp(
            (lower**2 - lower * tmp) / 4.0 + 0.5
        )
        if upper <= b_2_a:
            M = normpdf(
                lower
            )  # / (normcdf(upper) - normcdf(lower)) # * (upper - lower)
            while True:
                z = np.random.rand(1) * (upper - lower) + lower
                u = np.random.rand(1)
                if u <= normpdf(z) / M:
                    return z
        else:
            alpha = (lower + np.sqrt(lower**2 + 4)) / 2.0
            while True:
                z = np.random.exponential(alpha) + lower
                if z <= upper:
                    rho_z = np.exp(-((z - alpha) ** 2) / 2.0)
                    u = np.random.rand(1)
                    if u <= rho_z:
                        return z


def normal_cdf_inverse(z: npt.ArrayLike) -> npt.ArrayLike:
    return special.erfinv(2 * z - 1) * SQRT_TWO


def normcdf(x: npt.ArrayLike) -> npt.ArrayLike:
    return 0.5 * (1 + special.erf(x / SQRT_TWO))


def normpdf(x: npt.ArrayLike) -> npt.ArrayLike:
    pdf = np.zeros(np.shape(x))
    small_x_idx = np.abs(x) < 50
    pdf[small_x_idx] = np.exp(-x[small_x_idx] ** 2 / 2) / (np.sqrt(2 * np.pi))
    return pdf


class RFF_RBF:
    """
    rbf(gaussian) kernel of GPy k(x, y) = variance * exp(- 0.5 * ||x - y||_2^2 / lengthscale**2)
    """

    def __init__(self, lengthscales, input_dim, variance=1, basis_dim=1000):
        self.basis_dim = basis_dim
        self.std = np.sqrt(variance)
        self.random_weights = (1 / np.atleast_2d(lengthscales)) * np.random.normal(
            size=(basis_dim, input_dim)
        )
        self.random_offset = np.random.uniform(0, 2 * np.pi, size=basis_dim)

    def transform(self, X):
        X = np.atleast_2d(X)
        X_transform = X.dot(self.random_weights.T) + self.random_offset
        X_transform = self.std * np.sqrt(2 / self.basis_dim) * np.cos(X_transform)
        return X_transform

    """
    Only for one dimensional X
    """

    def transform_grad(self, X):
        X = np.atleast_2d(X)
        X_transform_grad = X.dot(self.random_weights.T) + self.random_offset
        X_transform_grad = (
            -self.std
            * np.sqrt(2 / self.basis_dim)
            * np.sin(X_transform_grad)
            * self.random_weights.T
        )
        return X_transform_grad


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
