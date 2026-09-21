import math

import torch
import gpytorch
import scipy

from utils import utility

# Pytorch's default tensor type is float32. Change for the preformance of operations such as Cholesky
torch.set_default_dtype(torch.float64)

class Kernel:
    def __init__(
        self,
        kernel_name: str = "RBFKernel",
        use_scale: bool = False,
        scale: float = 1.0,
        lengthscale: float = 1.0,
    ) -> None:
        # Retrieve class type from gpytorch.kernels
        kern_cls = getattr(gpytorch.kernels, kernel_name, None)

        # Check kernel class
        if kern_cls is None or not isinstance(kern_cls, type):
            raise ValueError(
                f"A kernel '{kernel_name}' cannot be found from 'gpytorch.kernels'."
            )   

        lengthscale = float(lengthscale)

        if not math.isfinite(lengthscale) or lengthscale <= 0:
            raise ValueError(
                f"Lengthscale must be finite and > 0. Got: {lengthscale}"
            )

        # Create a base kernel
        base_kernel = kern_cls()

        # If the kernel has lengthscale (RBF/Matern) set lengthscale
        # RBF/Matern 등 lengthscale을 가진 kernel이면 설정
        if getattr(base_kernel, "has_lengthscale", False):
            base_kernel.lengthscale = lengthscale
        else:
            raise ValueError(
                f"{kernel_name} does not support lengthscale."
            )
        
        self.kernel_name = kernel_name
        self.lengthscale = lengthscale

        if use_scale:
            if scale <= 0:
                raise ValueError(
                    f"Covariance scale must be > 0. Got: {scale}"
                )

            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            self.covar_module.outputscale = scale
            self.scale = scale

        else:
            self.covar_module = base_kernel
            self.scale = None

    @torch.inference_mode()
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if x2 is None:
            return self.covar_module(x1).to_dense()

        return self.covar_module(x1, x2).to_dense()


class PrefGP:
    def __init__(self,
                 kernel: Kernel | None = None,
                 kernel_name: str = "RBFKernel",
                 kernel_length_scale: float | None = 5e-2,
                 kernel_scale: float | None = None,
                 noise_std: float = 1e-2,
                 input_dim: int | None = None,
        ):
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer. Got: {input_dim}")
        
        noise_std = float(noise_std)
        if not math.isfinite(noise_std) or noise_std <= 0.0:
            raise ValueError(
                "noise_std must be positive for probit likelihood. "
                "Use a small value such as 1e-6 for near-deterministic feedback."
            )

        if kernel is None:
            # Create a kernel object
            if kernel_length_scale is None:
                raise ValueError("kernel_length_scale must be provided.")

            self.kernel = Kernel(
                kernel_name=kernel_name,
                lengthscale=kernel_length_scale,
                use_scale=kernel_scale is not None,
                scale=kernel_scale if kernel_scale is not None else 1.0,
            )

        else: # If a kernel is provided
            self.kernel = kernel

        self.kernel_length_scale = self.kernel.lengthscale
        self.kernel_scale = self.kernel.scale

        self.noise_std = noise_std

        self.num_duels = -1
        self.input_dim = input_dim

        self._X_train = None
        self._Y_train = None
        self.duel = None

        # Constructed from self._X_train and self._Y_train
        self.X_unique = None
        self.W = None # (n_duel, n_unique)
        self.K_unique = None
        self.K_unique_inv = None

        # Internal matrices for prediction
        self._L = None
        self._B = None

        # internal constants
        self._SQRT2 = math.sqrt(2)
        self._SQRT2_OVER_PI = math.sqrt(2.0 / math.pi)

    @torch.inference_mode()
    def update_observations(self, 
                            X_train: torch.Tensor,
                            Y_train: torch.Tensor, 
                            duel: torch.Tensor | None = None) -> None:
        """Update X_train and Y_train.

        Args:
            X_train (torch.Tensor): Points of duel observations 
            (num_duels, 2, n_dim).
            Y_train (torch.Tensor): Utility observations for both candidates,
                with shape (num_duels, 2).
            duel (torch.Tensor | None): Optional preference signs with one +1
                winner and one -1 loser per row. Derived from Y_train if omitted.

        Raises:
            ValueError: if the numbers of duels do not match.
        """
        # Warn if their are data set
        if (self._X_train is not None) or (self._Y_train is not None):
            print("Overwrite the train data")

        self._validate_X_train(X_train)
        self._validate_Y_train(Y_train)

        # Check the nubmer of duel observations
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("The number of duels between X_train and Y_train" \
            " are different:" \
            f" Got X_train: {X_train.shape}, Y_train: {Y_train.shape}"
            )

        if X_train.shape[0] == 0:
            raise ValueError("At least one duel observation is required.")
        if X_train.device != Y_train.device:
            raise ValueError("X_train and Y_train must be on the same device.")
        if not torch.isfinite(X_train).all() or not torch.isfinite(Y_train).all():
            raise ValueError("X_train and Y_train must contain only finite values.")

        if duel is None:
            if torch.any(Y_train[:, 0] == Y_train[:, 1]):
                raise ValueError(
                    "Tied utility observations require an explicit duel winner."
                )
            # Construct duel observation (from ../utils/utility.py)
            max_idx = torch.argmax(Y_train, dim=1)
            duel = -torch.ones(
                Y_train.shape,
                dtype=X_train.dtype,
                device=X_train.device,
            )
            duel.scatter_(1, max_idx.unsqueeze(1), 1.0) # Place 1 to the winning matrix
        else:
            if not isinstance(duel, torch.Tensor):
                raise TypeError("duel must be a torch.Tensor.")
            if duel.device != X_train.device:
                raise ValueError("duel and X_train must be on the same device.")
            duel = duel.to(dtype=X_train.dtype)

        X_unique, W = utility.construct_duel_matrix(X_train, duel)

        self.kernel.covar_module.to(device=X_train.device, dtype=X_train.dtype)
        K_unique = self.kernel(X_unique)
        K_unique, chol = self._stable_cholesky(K_unique)
        K_unique_inv = torch.cholesky_inverse(chol)

        self._X_train = X_train
        self._Y_train = Y_train
        self.duel = duel
        self.num_duels = X_train.shape[0]
        self.X_unique = X_unique
        self.W = W
        self.K_unique = K_unique
        self.K_unique_inv = K_unique_inv

    @torch.inference_mode()
    def add_observations(self, X_new_obs, Y_new_obs) -> None:
        """Add a new duel to the existing duel dataset (function for BayesOpt).

        Args:
            X_new_obs (torch.Tensor): (2, n_dim)
            Y_new_obs (torch.Tensor): (2,)

        """
        if not isinstance(X_new_obs, torch.Tensor) or not isinstance(Y_new_obs, torch.Tensor):
            raise TypeError("X_new_obs and Y_new_obs must be torch.Tensor objects.")
        
        if X_new_obs.shape != (2, self.input_dim):
            raise ValueError(
                f"X_new_obs must have shape (2, {self.input_dim}). "
                f"Got: {tuple(X_new_obs.shape)}"
            )
        if Y_new_obs.shape != (2,):
            raise ValueError(f"Y_new_obs must have shape (2,). Got: {tuple(Y_new_obs.shape)}")
        if X_new_obs.device != self.X_train.device or Y_new_obs.device != self.Y_train.device:
            raise ValueError("New observations must be on the same device as the training data.")

        X_train = self.X_train # (n_duels, 2, n_dim)
        Y_train = self.Y_train # (n_duels, 2)

        # Check duplicated duels, raise error if there's a duplicated one
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        new_obs_flat = X_new_obs.reshape(1, -1)
        new_obs_rev_flat = X_new_obs.flip(0).reshape(1, -1)

        duplicate = (
            X_train_flat.eq(new_obs_flat).all(dim=1)
            | X_train_flat.eq(new_obs_rev_flat).all(dim=1)
        )
        if torch.any(duplicate):
            dup_idx = torch.nonzero(duplicate, as_tuple=True)[0][0].item()
            raise ValueError(f"Duplicated duel observation at index {dup_idx}.")
        
        # Add X_new_obs and Y_new_obs to current self.X_train and self.Y_train
        X_train = torch.cat([X_train, X_new_obs.unsqueeze(0)], dim=0) # (n_duels + 1, 2, n_dim)
        Y_train = torch.cat([Y_train, Y_new_obs.unsqueeze(0)], dim=0) # (n_duels + 1, 2)

        if Y_new_obs[0] == Y_new_obs[1]:
            raise ValueError("Tied utility observations require an explicit duel winner.")
        
         # Update duel as well (from ../utils/utility.py)
        max_idx = torch.argmax(Y_new_obs).reshape(1, 1)
        new_duel = -torch.ones(
            (1, 2),
            dtype=self.duel.dtype,
            device=self.duel.device,
        )
        new_duel.scatter_(1, max_idx, 1.0)
        duel = torch.cat([self.duel, new_duel], dim=0)

        # Update observation with the updated X_train and Y_train
        self.update_observations(X_train, Y_train, duel=duel)

    @torch.inference_mode()
    def _inverse_mills_ratio(self, z: torch.Tensor) -> torch.Tensor:
        """Compute inverse Mills ratio for a given tensor z.
        lambda(z) = phi(z) / Phi(z)
        
        Args:
            z (torch.Tensor): the input 

        Returns:
            torch.Tensor: inverse Mills ratio of z.
        """
        return self._SQRT2_OVER_PI / torch.special.erfcx(-z / self._SQRT2)
    
    @torch.inference_mode()
    def _standard_normal_logcdf(self, z: torch.Tensor) -> torch.Tensor:
        r"""Safe implementation of \log \Phi(z).
        Args:
            z (torch.Tensor): a value to evaluate
        Returns:
            torch.Tensor: a CDF value at z
        """
        # \log \Phi(z)
        if hasattr(torch.special, "log_ndtr"):
            return torch.special.log_ndtr(z)
        
        # Fallback
        tiny = torch.finfo(z.dtype).tiny
        return torch.log(
            (0.5 * torch.special.erfc(-z / self._SQRT2)).clamp_min(tiny)
        )
    
    @torch.inference_mode()
    def _pairwise_scale(self) -> float:
        """Compute a scale factor of the affine probit likelihood.
        Assumptions:
            self.noise_std (float): a noise strength (observation noise std).
        Returns:
            float: a computed a noise strength 
        """
        return math.sqrt(2.0) * self.noise_std
    
    @torch.inference_mode()
    def _stable_cholesky(
            self,
            K: torch.Tensor,
            initial_jitter: float | None = None,
            max_tries: int = 8,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stable Cholesky decomposition to prevent non-positive definite error.

        Args:
            K (torch.Tensor): A kernel matrix to decompose
            initial_jitter (float | None): initial jitter amount (will be x10 for each iteration).
            max_tries (int): the maximum tries

        Returns:
            torch.Tensor: Symmetrized and jittered K
            torch.Tensor: Cholesky decomposed matrix

        """
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError(f"K must be square matrix. Got {tuple(K.shape)}.")

        # Symmetrize the kernel--remove tiny numerical asymmetry (safe Cholesky)
        K = 0.5 * (K + K.mT)

        if initial_jitter is None:
            initial_jitter = 1e-6 if K.dtype == torch.float32 else 1e-8

        identity = torch.eye(
            K.shape[0],
            dtype=K.dtype,
            device=K.device,
        )

        jitter = initial_jitter

        for _ in range(max_tries):
            try:
                chol = torch.linalg.cholesky(K + jitter * identity)
                return K + jitter * identity, chol
            except RuntimeError:
                jitter *= 10.0

        eigvals = torch.linalg.eigvalsh(K.detach().cpu().double())

        raise RuntimeError(
            "Cholesky failed even after adaptive jitter. "
            f"min_eig={eigvals.min().item()}, "
            f"max_eig={eigvals.max().item()}, "
            f"final_jitter={jitter}"
        )

    @property
    def X_train(self):
        if self._X_train is None:
            raise ValueError("X_train has not been set.")
        
        return self._X_train
    
    @property
    def Y_train(self):
        if self._Y_train is None:
            raise ValueError("Y_train has not been set.")
        
        return self._Y_train

    @X_train.setter
    def X_train(self, X_train):
        self._validate_X_train(X_train)
        self._X_train = X_train

    def _validate_X_train(self, X_train: torch.Tensor) -> None:
        if not isinstance(X_train, torch.Tensor):
            raise TypeError(f"X_train should be torch.Tensor. (Input type: {type(X_train)})")
        
        # Check dimension (num_duel, 2, self.input_dim)
        if (X_train.ndim != 3) or (X_train.shape[2] != self.input_dim) or (X_train.shape[1] != 2):
            raise ValueError(
                f"Incorrect shape for X_train\n"
                f"Expect: (num_duel, 2, {self.input_dim})\n"
                f"Got: {tuple(X_train.shape)}"
            )

        if not X_train.is_floating_point():
            raise TypeError("X_train must use a floating-point dtype.")

    @Y_train.setter
    def Y_train(self, Y_train):
        self._validate_Y_train(Y_train)
        self._Y_train = Y_train

    def _validate_Y_train(self, Y_train: torch.Tensor) -> None:
        if not isinstance(Y_train, torch.Tensor):
            raise TypeError(f"Y_train should be torch.Tensor. (Input type: {type(Y_train)})")
        
        # Check dimension (num_duel, 2)
        if Y_train.ndim != 2 or Y_train.shape[1] != 2:
            raise ValueError(
                f"Incorrect shape for Y_train\n"
                f"Expect: (num_duel, 2)\n"
                f"Got: {tuple(Y_train.shape)}"
            )


class PrefGP_LA(PrefGP):
    def __init__(
        self,
        kernel: Kernel | None = None,
        kernel_name: str = "RBFKernel",
        kernel_length_scale: float | None = 5e-2,
        kernel_scale: float | None = None,
        noise_std: float = 1e-2,
        input_dim: int | None = None,
        newton_iter_num: int = 100,
        newton_threshold: float = 1e-5,
    ):
        super().__init__(
            kernel=kernel,
            kernel_name=kernel_name,
            kernel_length_scale=kernel_length_scale,
            kernel_scale=kernel_scale,
            noise_std=noise_std,
            input_dim=input_dim,
        )

        # Check parameters related to Newton's method
        if not isinstance(newton_iter_num, int) or newton_iter_num <= 0:
            raise ValueError("newton_iter_num must be a positive integer.")
        newton_threshold = float(newton_threshold)
        if not math.isfinite(newton_threshold) or newton_threshold <= 0.0:
            raise ValueError("newton_threshold must be positive.")

        # Hyperparameters
        self._newton_iter_num = newton_iter_num
        self._newton_threshold = newton_threshold

    @torch.inference_mode()
    def predict(self,
                X_test: torch.Tensor,
                mean_train: torch.Tensor,
                cov_train: torch.Tensor,
                is_full_cov: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make a prediction at X_test, based on (X_train, Y_train) 
        with a given approximation point, X_test.
        Since the posterior is approximated with Laplace approximation,
        prediction is a form of a Gausssian distribution.
        
        Args:
            X_test (torch.Tensor): points to compute the posterior predictive. (n_pred, n_dim)
            mean_train (torch.Tensor): 
            cov_train (torch.Tensor): 
            is_full_cov (bool): whether the full covariance matrix is computed (True)
                or variance on each point is computed (False).
        
        Returns:
            mean (torch.Tensor): a predictive mean (n_pred, )
            
            covariance_or_variance:
                If is_full_cov is True:
                    A covariance tensor (matrix) (n_pred, n_pred).
                If is_full_cov is False:
                    A variance vector (n_pred, ).
        """

        if self.X_unique is None or self.K_unique_inv is None:
            raise RuntimeError("Call update_observations before prediction.")
        if not all(isinstance(value, torch.Tensor) for value in (X_test, mean_train, cov_train)):
            raise TypeError("X_test, mean_train, and cov_train must be torch.Tensor objects.")

        # Accept a single point with shape (n_dim, )
        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(0)

        if X_test.ndim != 2 or X_test.shape[1] != self.input_dim:
            raise ValueError(
                f"X_test must have shape (n_pred, {self.input_dim}). "
                f"Got: {tuple(X_test.shape)}"
            )
        if X_test.device != self.X_unique.device:
            raise ValueError("X_test must be on the same device as the training data.")

        f_mean = mean_train
        f_cov = cov_train

        if f_mean.ndim == 2 and f_mean.shape[-1] == 1:
            f_mean = f_mean.squeeze(-1)

        n_unique = self.X_unique.shape[0]
        if f_mean.shape != (n_unique,):
            raise ValueError(f"mean_train must have shape ({n_unique},).")
        if f_cov.shape != (n_unique, n_unique):
            raise ValueError(f"cov_train must have shape ({n_unique}, {n_unique}).")
        if f_mean.device != self.X_unique.device or f_cov.device != self.X_unique.device:
            raise ValueError("Posterior moments must be on the same device as the training data.")

        f_cov = f_cov.to(dtype=f_mean.dtype)

        K_inv = self.K_unique_inv.to(dtype=f_mean.dtype)

        # K_u_test := K(X_unique, X_test)
        # shape (n_unique, n_pred)
        K_u_test = self.kernel(self.X_unique, X_test).to(dtype=f_mean.dtype)

        # Predictive mean
        # mean = K_u_test K_uu^{-1} f_mean
        #      = K_u_test^\top k_inv f_mean
        alpha = K_inv @ f_mean
        pred_mean = K_u_test.T @ alpha

        # Predictive covariance / variance.
        # A = K_inv - K_inv @ f_cov @ K_inv
        A = K_inv - K_inv @ f_cov @ K_inv

        if is_full_cov:
            K_test_test = self.kernel(X_test).to(dtype=f_mean.dtype)

            pred_cov = K_test_test - K_u_test.T @ A @ K_u_test

            # Numerical symmetrization
            pred_cov = 0.5 * (pred_cov + pred_cov.mT)

            return pred_mean, pred_cov
        
        else:
            # Diag K(X_test, X_test)
            pred_var_prior = self.kernel(X_test).to(dtype=f_mean.dtype).diagonal()

            # diag(K_u_test^\top @ A @ K_u_test)
            correction = torch.sum(K_u_test * (A @ K_u_test), dim=0)

            pred_var = pred_var_prior - correction

            # Tiny negative values can appear due to numerical error
            pred_var = pred_var.clamp_min(0.0)

            return pred_mean, pred_var

    @torch.inference_mode()
    def _find_probe(self,
                    f_map: torch.Tensor,
                    probing_direction: torch.Tensor,
                    probing_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Construct a local Laplace approximation at a probing point.

        The probing point is

        $$f_{probe} = f_{map} + \alpha d,$$

        where ``alpha`` is ``probing_strength`` and ``d`` is
        ``probing_direction``. A quadratic expansion of the negative log
        posterior around this point has mean

        $$f_{probe} - H_{probe}^{-1} g_{probe}$$

        and covariance ``H_probe^{-1}``.

        Args:
            f_map: MAP utility values, shape ``(n_unique,)``.
            probing_direction: Direction from the MAP, shape ``(n_unique,)``.
            probing_strength: Scalar multiplier for the probing direction.

        Returns:
            A tuple containing the locally corrected mean, covariance, and
            stabilized Hessian at the probing point.
        """
        if self.W is None or self.K_unique_inv is None:
            raise RuntimeError("Call update_observations before finding a probe.")
        if not isinstance(f_map, torch.Tensor) or not isinstance(
            probing_direction, torch.Tensor
        ):
            raise TypeError("f_map and probing_direction must be torch.Tensor objects.")

        n_unique = self.K_unique_inv.shape[0]
        expected_shape = (n_unique,)
        if f_map.shape != expected_shape or probing_direction.shape != expected_shape:
            raise ValueError(
                "f_map and probing_direction must both have shape "
                f"{expected_shape}."
            )
        if f_map.device != self.K_unique_inv.device or probing_direction.device != f_map.device:
            raise ValueError(
                "f_map, probing_direction, and the fitted GP must be on the same device."
            )
        if not f_map.is_floating_point() or not probing_direction.is_floating_point():
            raise TypeError("f_map and probing_direction must use floating-point dtypes.")
        if not torch.isfinite(f_map).all() or not torch.isfinite(probing_direction).all():
            raise ValueError("f_map and probing_direction must contain only finite values.")

        probing_strength = float(probing_strength)
        if not math.isfinite(probing_strength):
            raise ValueError("probing_strength must be finite.")

        device = f_map.device
        work_dtype = torch.promote_types(f_map.dtype, self.K_unique_inv.dtype)
        if work_dtype in (torch.float16, torch.bfloat16, torch.float32) and device.type != "mps":
            work_dtype = torch.float64

        f_probe = f_map.to(dtype=work_dtype) + probing_strength * probing_direction.to(
            dtype=work_dtype
        )
        W = self.W.to(device=device, dtype=work_dtype)
        scale = self._pairwise_scale()

        with torch.no_grad():
            z_probe = (W @ f_probe) / scale
            inverse_mills_ratio = self._inverse_mills_ratio(z_probe)
            gradient_probe = self._objective_gradient(
                f=f_probe,
                W=W,
                inverse_mills_ratio=inverse_mills_ratio,
            )
            hessian_probe = self._objective_hessian(
                z=z_probe,
                inverse_mills_ratio=inverse_mills_ratio,
                W=W,
            )
            hessian_probe, chol = self._stable_cholesky(hessian_probe)
            correction = torch.cholesky_solve(
                gradient_probe.unsqueeze(-1),
                chol,
            ).squeeze(-1)
            mean_probe = f_probe - correction
            covariance_probe = torch.cholesky_inverse(chol)

        return mean_probe, covariance_probe, hessian_probe
    
    @torch.inference_mode()
    def _find_proving_direction(self, X_test: torch.Tensor) -> torch.Tensor:
        r"""Find predictive-mean probing directions for test points.

        For a test-point covariance vector ``k_j`` and the MAP Hessian
        ``H_0``, the unnormalized direction is

        $$d_j' = H_0^{-1} K^{-1} k_j.$$

        Its transpose is the Laplace posterior cross-covariance between
        ``f(x_j)`` and the latent utilities at ``X_unique``. To obtain a unit
        direction under the ``H_0`` metric, divide ``d_j'`` by
        ``sqrt((K^{-1} k_j)^T d_j')``.

        Args:
            X_test: Predictive points with shape ``(n_test, input_dim)`` or a
                single point with shape ``(input_dim,)``.

        Returns:
            One unnormalized probing direction per predictive point, with
            shape ``(n_test, n_unique)``.
        """
        if self.W is None or self.X_unique is None or self.K_unique_inv is None:
            raise RuntimeError(
                "Call update_observations before finding probing directions."
            )
        if not isinstance(X_test, torch.Tensor):
            raise TypeError("X_test must be a torch.Tensor.")
        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(0)
        if X_test.ndim != 2 or X_test.shape[1] != self.input_dim:
            raise ValueError(
                f"X_test must have shape (n_test, {self.input_dim}). "
                f"Got: {tuple(X_test.shape)}"
            )
        if X_test.shape[0] == 0:
            raise ValueError("X_test must contain at least one predictive point.")
        if X_test.device != self.X_unique.device:
            raise ValueError("X_test must be on the same device as the training data.")
        if not X_test.is_floating_point() or not torch.isfinite(X_test).all():
            raise ValueError("X_test must contain finite floating-point values.")

        _, covariance_map, _ = self.inference()
        work_dtype = covariance_map.dtype
        X_test = X_test.to(dtype=self.X_unique.dtype)
        K_unique_test = self.kernel(self.X_unique, X_test).to(dtype=work_dtype)
        precision = self.K_unique_inv.to(dtype=work_dtype)

        precision_cross_covariance = precision @ K_unique_test
        probing_directions = covariance_map @ precision_cross_covariance

        return probing_directions.T.contiguous()

    @torch.inference_mode()
    def __find_proving_direction_pca(self, k: int = 1) -> torch.Tensor:
        r"""Find the leading likelihood-active directions using PCA of ``W``.

        Args:
            k (int): the nubmer of proving points. 

        Assumptions:
            self.W (torch.Tensor): A duel observation data matrix (n_duel, n_unique).
            self.X_unique: A vector of unique observation points (n_unique, n_dim).

        Returns:
            torch.Tensor: a collection of k probing points (k, n_unique).

        Raises:
            RuntimeError:
                If observations have not been initialized or W is not a correct
                duel observation matrix.
            TypeError:
                If k is not an integer.
            ValueError:
                If k is invalid or exceeds the numerical rank of W.
        """
        if self.W is None or self.X_unique is None:
            raise RuntimeError(
                "Call update_observations before finding probing directions."
            )
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError(f"k must be an integer. Got {type(k).__name__}.")
        if k <= 0:
            raise ValueError(f"k must be positive. Got {k}.")

        W = self.W
        if W.ndim != 2 or W.shape[1] != self.X_unique.shape[0]:
            raise ValueError(
                "W must have shape (n_duels, n_unique), with one column per "
                "unique observation point."
            )
        if not W.is_floating_point():
            W = W.to(dtype=torch.get_default_dtype())
        if not torch.isfinite(W).all():
            raise ValueError("W must contain only finite values.")

        _, singular_values, Vh = torch.linalg.svd(W, full_matrices=False)
        if singular_values.numel() == 0:
            raise RuntimeError("W has no singular vectors.")

        tolerance = (
            max(W.shape)
            * torch.finfo(singular_values.dtype).eps
            * singular_values[0]
        )
        numerical_rank = int(
            torch.count_nonzero(singular_values > tolerance).item()
        )
        if numerical_rank == 0:
            raise RuntimeError("W has no nonzero likelihood-active directions.")
        if k > numerical_rank:
            raise ValueError(
                f"k={k} exceeds the numerical rank of W, which is "
                f"{numerical_rank}. Directions beyond this rank lie in the "
                "null space of W and do not change any duel margin."
            )

        probing_directions = Vh[:k].clone()
        largest_indices = torch.argmax(torch.abs(probing_directions), dim=1)
        row_indices = torch.arange(k, device=probing_directions.device)
        signs = torch.sign(probing_directions[row_indices, largest_indices])
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)

        return probing_directions * signs.unsqueeze(1)

    @torch.inference_mode()
    def inference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the MAP of the posterior given (self._X_train, self._Y_train)
        and the posterior objective $S(f)$.

        Assumptions:
            self.X_unique (torch.Tensor):
            self.K_unique (torch.Tensor):
            self.W (torch.Tensor):

        Returns:
            torch.Tensor: a Maximum a priori point of the posterior (f_map)
            torch.Tensor: Covariance at the MAP f_map (covariance_map)
            torch.Tensor: Lambda 
        
        Raises:

        """
        # Check assumptions
        if self.X_unique is None or self.W is None or self.K_unique_inv is None:
            raise RuntimeError("Call update_observations before finding the MAP.")

        # Types and devices
        device = self.K_unique_inv.device
        source_dtype = self.K_unique_inv.dtype
        work_dtype = source_dtype
        if source_dtype in (torch.float16, torch.bfloat16, torch.float32) and device.type != "mps":
            work_dtype = torch.float64

        # Setting variables and parameters
        W = self.W.to(device=device, dtype=work_dtype)
        precision = self.K_unique_inv.to(dtype=work_dtype)
        scale = self._pairwise_scale()
        f_map = torch.zeros(W.shape[1], dtype=work_dtype, device=device)
        armijo = 1e-4

        def objective(f: torch.Tensor) -> torch.Tensor:
            z = (W @ f) / scale
            return 0.5 * torch.dot(f, precision @ f) - self._standard_normal_logcdf(z).sum()

        with torch.no_grad():
            converged = False
            for _ in range(self._newton_iter_num):
                z = (W @ f_map) / scale
                
                # inverse Mills rator for gradient and Hessian
                inverse_mills_ratio = self._inverse_mills_ratio(z)

                # Compute gradient
                likelihood_gradient = -(W.T @ inverse_mills_ratio) / scale
                gradient = precision @ f_map + likelihood_gradient

                # Compute Hessian
                Lambda = self._objective_lambda(z, inverse_mills_ratio, W)
                hessian = precision + Lambda

                # Sove for hessian * update = gradient <=> update = hessian^{-1} * gradient
                _, chol = self._stable_cholesky(hessian)
                update = torch.cholesky_solve(
                    gradient.unsqueeze(-1),
                    chol,
                ).squeeze(-1)

                # For safety check g^\top H^{-1} g > 0, decrement := g^\top H^{-1} g as update H^{-1} g
                # if decrement < 0, that means it moves to the opposite direction of the gradient g.
                decrement = torch.dot(gradient, update) 

                if not torch.isfinite(decrement) or decrement < 0.0:
                    raise RuntimeError("Newton solve produced a non-finite descent direction.")
                
                # Convergence test
                if 0.5 * decrement <= self._newton_threshold:
                    converged = True
                    break
                
                # Backtracking line search
                current_objective = objective(f_map)
                step_size = 1.0
                for _ in range(30):
                    candidate = f_map - step_size * update
                    candidate_objective = objective(candidate)
                    if (
                        torch.isfinite(candidate_objective)
                        and candidate_objective
                        <= current_objective - armijo * step_size * decrement
                    ):
                        f_map = candidate
                        break
                    step_size *= 0.5
                else:
                    raise RuntimeError("Newton line search failed to decrease the MAP objective.")

            if not converged:
                raise RuntimeError(
                    "MAP optimization did not converge within "
                    f"{self._newton_iter_num} iterations."
                )

            z = (W @ f_map) / scale
            inverse_mills_ratio = self._inverse_mills_ratio(z)
            Lambda = self._objective_lambda(z, inverse_mills_ratio, W)
            hessian = precision + Lambda
            _, chol = self._stable_cholesky(hessian)
            covariance_map = torch.cholesky_inverse(chol)

        return f_map, covariance_map, Lambda

    @torch.inference_mode()
    def inference_orig(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the original Laplace Newton solver on the current observations.

        Uses full Newton steps and checks the maximum coordinate change after
        applying each update. Unlike ``inference``, this method has no line
        search, Newton-decrement stopping test, or additional Hessian jitter.
        The configured iteration limit and threshold are used (defaults: 100
        and 1e-5, as in the original).

        The current unique-input representation, prior stabilization, and
        derivative helpers are retained. Returns the same ``(f_map,
        covariance_map, Lambda)`` shapes, device, and working dtype as
        ``inference``. Invalid/divergent updates and iteration exhaustion raise
        ``RuntimeError`` rather than returning the original scalar NaNs or
        silently accepting an unconverged result.
        """
        if self.X_unique is None or self.W is None or self.K_unique_inv is None:
            raise RuntimeError("Call update_observations before finding the MAP.")

        device = self.K_unique_inv.device
        source_dtype = self.K_unique_inv.dtype
        work_dtype = source_dtype
        if source_dtype in (torch.float16, torch.bfloat16, torch.float32) and device.type != "mps":
            work_dtype = torch.float64

        W = self.W.to(device=device, dtype=work_dtype)
        precision = self.K_unique_inv.to(dtype=work_dtype)
        scale = self._pairwise_scale()
        f_map = torch.zeros(W.shape[1], dtype=work_dtype, device=device)

        for _ in range(self._newton_iter_num):
            z = (W @ f_map) / scale
            inverse_mills_ratio = self._inverse_mills_ratio(z)
            gradient = self._objective_gradient(f_map, W, inverse_mills_ratio)
            Lambda = self._objective_lambda(z, inverse_mills_ratio, W)
            hessian = precision + Lambda

            update = torch.linalg.solve(hessian, gradient)
            candidate = f_map - update
            if not torch.isfinite(candidate).all():
                raise RuntimeError("Original Newton solver produced non-finite utilities.")

            converged = (
                torch.max(torch.abs(candidate - f_map)).item()
                <= self._newton_threshold
            )
            f_map = candidate
            if converged:
                break

            # Preserve the original solver's divergence guard, with API-safe
            # error handling instead of a tuple of scalar NaNs.
            if torch.any(torch.abs(f_map) > 5.0).item():
                raise RuntimeError("Original Newton solver exceeded the utility bound of 5.")
        else:
            raise RuntimeError(
                "MAP optimization did not converge within "
                f"{self._newton_iter_num} iterations."
            )

        z = (W @ f_map) / scale
        inverse_mills_ratio = self._inverse_mills_ratio(z)
        Lambda = self._objective_lambda(z, inverse_mills_ratio, W)
        hessian = precision + Lambda
        covariance_map = torch.linalg.inv(hessian)

        return f_map, covariance_map, Lambda

    @torch.inference_mode()
    def affine_probit_likelihood(self, 
                                 f_unique: torch.Tensor,
                                 W: torch.Tensor) -> torch.Tensor:
        """An affine probit likelihood implementation with
        the given true utility f

        Args:
            f_unique (torch.Tensor):
            W (torch.Tensor):
        
        Returns:
            likelihood (torch.Tensor):
        """

        # Compute z := \frac{f(v) - f(u)}{\sqrt{2}\sigma} and the likelihood
        scale = self._pairwise_scale()
        W = W.to(device=f_unique.device, dtype=f_unique.dtype)
        z = (W @ f_unique) / scale

        return -self._standard_normal_logcdf(z).sum()

    @torch.inference_mode()
    def _map_objective(self, f: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        r"""Define the objective for optimization to find the MAP.

        The objective is evaluated at the give f, i.e., $S(f)$.
        
        $$S(f) = -\sum_{i=1}^m \log \Phi(z_i)
        + \frac{1}{2}f^\top K^{-1} f$$

        Args:
            f (torch.Tensor): A point to evaluate the objective $S$.
                This is unnormalized posterior probability at given $f$. (X_unique)
            W (torch.Tensor): A duel observation matrix corresponds to f.

        Returns:
            torch.Tensor: A tensor of MAP f_MAP.
        
        Raises: if self._X_train and self._Y_train are not set.
        """

        # Compute z := \frac{f(v) - f(u)}{\sqrt{2}\sigma} and the likelihood
        W = W.to(device=f.device, dtype=f.dtype)
        likelihood = self.affine_probit_likelihood(f, W)

        # Compute prior probability
        precision = self.K_unique_inv.to(device=f.device, dtype=f.dtype)
        prior = 0.5 * torch.dot(f, precision @ f)

        # Object value
        s_value = prior + likelihood # Should be (1, )

        return s_value

    @torch.inference_mode()
    def _objective_gradient(self,
                            f: torch.Tensor,
                            W: torch.Tensor,
                            inverse_mills_ratio: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute a gradient of the objective at a given f
        Args:
            f (torch.Tensor): a point to compute gradient.
            inverse_mills_rator (torch.Tensor): an inverse Mills ratio computed
                from f and its related duel matrix W.
            W (torch.Tensor): a duel matrix
            inverse_mills_ratio (torch.Tensor): \phi(z) / \Phi(z) where 
                z := W @ f / scale
        Returns:
            torch.Tensor: A gradient matrix
        """
        scale = self._pairwise_scale()

        # Prior gradient:
        #
        # d/df [0.5 f^T K^{-1} f] = K^{-1} f
        W = W.to(device=f.device, dtype=f.dtype)
        inverse_mills_ratio = inverse_mills_ratio.to(device=f.device, dtype=f.dtype)
        precision = self.K_unique_inv.to(device=f.device, dtype=f.dtype)
        prior_gradient = precision @ f

        # Likelihood gradient:
        #
        # d/df [-sum log Phi(Wf / scale)]
        #   = - W^T * (\phi(Wf) / \Phi(Wf) / scale
        likelihood_gradient = -(W.T @ inverse_mills_ratio) / scale

        return prior_gradient + likelihood_gradient

    @torch.inference_mode()
    def _objective_hessian(self,
                           z: torch.Tensor,
                           inverse_mills_ratio: torch.Tensor,
                           W: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute a Hessian of the objective at a given f
        Args:
            z (torch.Tensor): a point to compute Hessian.
            inverse_mills_rator (torch.Tensor): an inverse Mills ratio computed
                from f and its related duel matrix W.
        Returns:
            torch.Tensor: A Hessian matrix

        """
        Lambda = self._objective_lambda(
            z=z,
            inverse_mills_ratio=inverse_mills_ratio,
            W=W
        )

        precision = self.K_unique_inv.to(device=z.device, dtype=z.dtype)
        return precision + Lambda

    @torch.inference_mode()
    def _objective_lambda(self,
                          z: torch.Tensor,
                          inverse_mills_ratio: torch.Tensor,
                          W: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute Hessian contribution from the affine probit likelihood.

        This returns Lambda such that
        \nabla^2_f S(f) = K^{-1} (prior) + Lambda (Likelihood)

        For likelihood part
            -\sum_i \Phi (W_i * f) / scale

        The second derivative wrt z is:
            - W.T (-z * inv_mills_ratio(z) - inv_mills_ratio(z)^2) W / scale^2
            = W.T (z + inverse_mills_ratio(z)) * inverse_mills_ratio(z) W / scale^2.
        """
        scale = self._pairwise_scale()
        W = W.to(device=z.device, dtype=z.dtype)
        inverse_mills_ratio = inverse_mills_ratio.to(device=z.device, dtype=z.dtype)

        lambda_diag = inverse_mills_ratio * (inverse_mills_ratio + z)
        # This can be ill-conditioned since (noise_std ** 2) can be very small when noise_std == 1e-6.
        lambda_diag = (lambda_diag / (scale ** 2)).clamp_min(0.0)

        Lambda = W.T @ (lambda_diag.unsqueeze(-1) * W)

        return Lambda

    @torch.inference_mode()
    def _objective_third_derivative(
        self,
        z: torch.Tensor,
        inverse_mills_ratio: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the third derivative tensor of the MAP objective.

        For ``r(z) = phi(z) / Phi(z)`` and ``c(z) = r(z) (r(z) + z)``,
        ``r'(z) = -c(z)`` and

        $$c'(z) = r(z) - c(z) (2r(z) + z).$$

        Since ``z_i = W_i f / scale``, the likelihood contribution is

        $$\nabla^3 S(f) = \sum_i \frac{c'(z_i)}{scale^3}
        W_i \otimes W_i \otimes W_i.$$

        The Gaussian-prior term is quadratic and therefore contributes zero.

        Args:
            z: Scaled duel differences, shape ``(n_duels,)``.
            inverse_mills_ratio: ``phi(z) / Phi(z)``, shape ``(n_duels,)``.
            W: Duel matrix, shape ``(n_duels, n_unique)``.

        Returns:
            A symmetric tensor with shape
            ``(n_unique, n_unique, n_unique)``.
        """
        if z.ndim != 1 or inverse_mills_ratio.shape != z.shape:
            raise ValueError("z and inverse_mills_ratio must be vectors with equal shape.")
        if W.ndim != 2 or W.shape[0] != z.shape[0]:
            raise ValueError(
                "W must have shape (n_duels, n_unique) with one row per z value."
            )

        W = W.to(device=z.device, dtype=z.dtype)
        ratio = inverse_mills_ratio.to(device=z.device, dtype=z.dtype)
        curvature = ratio * (ratio + z)
        curvature_derivative = ratio - curvature * (2.0 * ratio + z)
        third_diag = curvature_derivative / (self._pairwise_scale() ** 3)

        return torch.einsum(
            "i,ij,ik,il->jkl",
            third_diag,
            W,
            W,
            W,
        )
    
    @torch.inference_mode()
    def _construct_correction_features(self,
                                        pos_dir: torch.Tensor,
                                        neg_dir: torch.Tensor,
                                        pos_mean: torch.Tensor,
                                        neg_mean: torch.Tensor,
                                        pos_cov: torch.Tensor,
                                        neg_cov: torch.Tensor,
                                        strength: float | None,
                                        X_test: torch.Tensor,
        ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        r"""Construct finite-difference features for the correction regressor.

        Let ``L = K(X_test, X_unique) K_unique^{-1}``. For positive and
        negative probes at ``f_0 +/- strength * direction``, this returns

        ``(strength * L @ pos_dir, strength * L @ neg_dir)``,

        ``(pos_mean + neg_mean - 2 * L @ f_0) / strength**2``, and

        ``(pos_cov - neg_cov) / (2 * strength)``.

        When ``strength`` is ``None``, the corresponding unscaled finite-
        difference numerators are returned.

        Args:
            pos_dir: Positive probing direction, shape ``(n_unique,)``.
            neg_dir: Negative probing direction, shape ``(n_unique,)``.
            pos_mean: Predictive mean at the positive probe, shape
                ``(n_test,)``.
            neg_mean: Predictive mean at the negative probe, shape
                ``(n_test,)``.
            pos_cov: Predictive covariance at the positive probe, shape
                ``(n_test, n_test)``.
            neg_cov: Predictive covariance at the negative probe, shape
                ``(n_test, n_test)``.
            strength: Probe strength, or ``None`` to omit finite-difference
                scaling.
            X_test: Prediction points, shape ``(n_test, input_dim)``.

        Returns:
            The positive/negative predictive direction features, centered
            mean feature, and odd covariance-difference feature.
        """
        if self.X_unique is None or self.K_unique_inv is None:
            raise RuntimeError(
                "Call update_observations before constructing correction features."
            )

        tensors = {
            "pos_dir": pos_dir,
            "neg_dir": neg_dir,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "pos_cov": pos_cov,
            "neg_cov": neg_cov,
            "X_test": X_test,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")

        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(0)
        if X_test.ndim != 2 or X_test.shape[1] != self.input_dim:
            raise ValueError(
                f"X_test must have shape (n_test, {self.input_dim}). "
                f"Got: {tuple(X_test.shape)}"
            )

        n_test = X_test.shape[0]
        n_unique = self.X_unique.shape[0]
        expected_shapes = {
            "pos_dir": (n_unique,),
            "neg_dir": (n_unique,),
            "pos_mean": (n_test,),
            "neg_mean": (n_test,),
            "pos_cov": (n_test, n_test),
            "neg_cov": (n_test, n_test),
        }
        for name, expected_shape in expected_shapes.items():
            if tensors[name].shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}. "
                    f"Got: {tuple(tensors[name].shape)}"
                )

        device = self.X_unique.device
        for name, value in tensors.items():
            if value.device != device:
                raise ValueError(
                    f"{name} must be on the same device as the training data."
                )
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(
                    f"{name} must contain finite floating-point values."
                )

        if strength is not None:
            strength = float(strength)
            if not math.isfinite(strength) or strength <= 0.0:
                raise ValueError("strength must be positive and finite, or None.")

        f_map, _, _ = self.inference()
        work_dtype = f_map.dtype
        K_unique_test = self.kernel(
            self.X_unique,
            X_test.to(dtype=self.X_unique.dtype),
        ).to(dtype=work_dtype)
        precision = self.K_unique_inv.to(dtype=work_dtype)
        projection = K_unique_test.T @ precision

        pos_direction_feature = projection @ pos_dir.to(dtype=work_dtype)
        neg_direction_feature = projection @ neg_dir.to(dtype=work_dtype)
        map_predictive_mean = projection @ f_map
        mean_feature = (
            pos_mean.to(dtype=work_dtype)
            + neg_mean.to(dtype=work_dtype)
            - 2.0 * map_predictive_mean
        )
        covariance_feature = (
            pos_cov.to(dtype=work_dtype) - neg_cov.to(dtype=work_dtype)
        )

        if strength is not None:
            pos_direction_feature = strength * pos_direction_feature
            neg_direction_feature = strength * neg_direction_feature
            mean_feature = mean_feature / (strength ** 2)
            covariance_feature = covariance_feature / (2.0 * strength)

        return (
            (pos_direction_feature, neg_direction_feature),
            mean_feature,
            covariance_feature,
        )


# class PrefGP_Gibbs(PrefGP):
#     def __init__(self, 
#                  kernel_name: str = "RBFKernel",
#                  kernel_scale: float | None = None,
#                  noise_std: float = 1e-2,
#                  input_dim: int | None = None,
#                  sampling_method: str = "Gibbs",
#                  sample_size: int = 1000,
#                  burn_in: int = 1000,
#                  thinning: int = 1,
#                  seed: int | None = None,
#         ):
        
#         super().__init__(
#             kernel_name=kernel_name,
#             kernel_scale=kernel_scale,
#             noise_std=noise_std,
#             input_dim=input_dim
#         )

#         self.sampling_method = sampling_method
#         self.sampling_size = sample_size
#         self.burn_in = burn_in
#         self.thinning = thinning

#         # MC varianbles
#         self.initial_sample = None
#         self.v_sample = None
#         self.initial_points_sampler = scipy.stats.qmc.Sobol(d=self.input_dim, seed=seed)


class PrefGP_Gibbs(PrefGP):
    def __init__(
        self,
        kernel: Kernel | None = None,
        kernel_name: str = "RBFKernel",
        kernel_length_scale: float | None = 5e-2,
        kernel_scale: float | None = None,
        noise_std: float = 1e-2,
        input_dim: int | None = None,
        sample_size: int = 1000,
        burn_in: int = 1000,
        thinning: int = 1,
    ):
        super().__init__(
            kernel=kernel,
            kernel_name=kernel_name,
            kernel_length_scale=kernel_length_scale,
            kernel_scale=kernel_scale,
            noise_std=noise_std,
            input_dim=input_dim,
        )

        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
        if burn_in < 0:
            raise ValueError("burn_in must be nonnegative.")
        if thinning <= 0:
            raise ValueError("thinning must be positive.")

        self.sample_size = sample_size
        self.burn_in = burn_in
        self.thinning = thinning

        # Latent comparison covariance and factorization.
        # g: f_w - f_l, g ~ N(0, W K_unique W^\top + 2\sigma^2 I)
        self.K_g = None
        self._K_g_chol = None
        self._K_g_precision = None

        # Shape: (num_duels, sample_size)
        self.g_samples = None

        # Posterior moments at X_unique.
        self.f_mean = None
        self.f_cov = None

    @staticmethod
    @torch.inference_mode()
    def _sample_std_normal_lower(
        lower: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample Z ~ N(0, 1), subject to Z >= lower.

        Uses rejection samplers similar to Robert (1995).
        """
        if lower.ndim != 0:
            raise ValueError("lower must be a scalar tensor.")

        lower_value = lower.item()

        if lower_value > 0.257:
            # Exponential-tail proposal.
            alpha = 0.5 * (
                lower + torch.sqrt(lower.square() + 4.0)
            )

            # Important: Exponential expects a rate, not a scale.
            exponential = torch.distributions.Exponential(rate=alpha)

            while True:
                z = lower + exponential.sample()
                acceptance = torch.exp(
                    -0.5 * (z - alpha).square()
                )

                if torch.rand(
                    (),
                    dtype=lower.dtype,
                    device=lower.device,
                ) <= acceptance:
                    return z

        if lower_value >= 0.0:
            # Half-normal proposal.
            while True:
                z = torch.randn(
                    (),
                    dtype=lower.dtype,
                    device=lower.device,
                ).abs()

                if z >= lower:
                    return z

        # Standard-normal rejection is efficient for lower < 0.
        while True:
            z = torch.randn(
                (),
                dtype=lower.dtype,
                device=lower.device,
            )

            if z >= lower:
                return z

    @classmethod
    @torch.inference_mode()
    def _positive_orthant_gibbs(
        cls,
        precision: torch.Tensor,
        sample_size: int,
        burn_in: int,
        thinning: int,
        initial_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample from N(0, precision^{-1}) subject to g > 0.

        Returns:
            Tensor with shape (dimension, sample_size).
        """
        if precision.ndim != 2:
            raise ValueError("precision must be a matrix.")
        if precision.shape[0] != precision.shape[1]:
            raise ValueError("precision must be square.")

        dimension = precision.shape[0]

        if initial_sample is None:
            current = torch.zeros(
                dimension,
                dtype=precision.dtype,
                device=precision.device,
            )
        else:
            if initial_sample.shape != (dimension,):
                raise ValueError(
                    f"initial_sample must have shape ({dimension},)."
                )
            if torch.any(initial_sample < 0):
                raise ValueError(
                    "initial_sample must lie in the positive orthant."
                )
            current = initial_sample.clone()

        precision_diag = precision.diagonal()

        if torch.any(precision_diag <= 0):
            raise RuntimeError(
                "Precision matrix must have a positive diagonal."
            )

        conditional_std = torch.rsqrt(precision_diag)

        # Row j is Q[j, :] / Q[j, j].
        scaled_precision = (
            precision / precision_diag.unsqueeze(1)
        )

        samples = []

        total_sweeps = burn_in + thinning * sample_size

        for sweep in range(1, total_sweeps + 1):
            for j in range(dimension):
                # Equivalent to
                #
                #   -sum_{k != j} Q[j,k] g[k] / Q[j,j].
                #
                conditional_mean = (
                    current[j]
                    - scaled_precision[j] @ current
                )

                std = conditional_std[j]

                # Need g_j >= 0:
                #
                # g_j = mean + std * z
                # z >= -mean / std.
                standardized_lower = -conditional_mean / std

                z = cls._sample_std_normal_lower(
                    standardized_lower
                )

                current[j] = conditional_mean + std * z

            if (
                sweep > burn_in
                and (sweep - burn_in) % thinning == 0
            ):
                samples.append(current.clone())

        return torch.stack(samples, dim=1)

    @torch.inference_mode()
    def inference(
        self,
        sample_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run Gibbs inference and return posterior moments of
        f at X_unique.

        Args:
            sample_size (int | None): 

        Returns:
            torch.Tensor: Posterior mean of f_unique
            torch.Tensor: Posterior covariance of f_unique
        """
        if (
            self.W is None
            or self.K_unique is None
            or self.X_unique is None
        ):
            raise RuntimeError(
                "Call update_observations before inference."
            )

        if sample_size is None:
            sample_size = self.sample_size

        W = self.W.to(
            device=self.K_unique.device,
            dtype=self.K_unique.dtype,
        )
        K_unique = self.K_unique

        num_duels = W.shape[0]

        identity = torch.eye(
            num_duels,
            dtype=K_unique.dtype,
            device=K_unique.device,
        )

        # Affine-probit comparison scale:
        #
        # scale^2 = (sqrt(2) * noise_std)^2
        #         = 2 * noise_std^2.
        comparison_variance = self._pairwise_scale() ** 2

        # f_unique ~ N(0, K_unique)
        # g = W @ f_unique + comparision_noise
        # comparison_noise ~ N(0, 2 * noise_std ** 2 * I)
        # observations imply g > 0
        # Then,
        # g ~ N(0, K_g), where K_g = W @ K_unique @ W.T + 2 * noise_std ** 2 * I
        self.K_g = (
            W @ K_unique @ W.T
            + comparison_variance * identity
        )

        self.K_g, self._K_g_chol = (
            self._stable_cholesky(self.K_g)
        )

        # Gibbs coordinate updates use the precision matrix.
        self._K_g_precision = torch.cholesky_inverse(
            self._K_g_chol
        )

        # The Gibbs sampler targets
        # p(g | g > 0)
        self.g_samples = self._positive_orthant_gibbs(
            precision=self._K_g_precision,
            sample_size=sample_size,
            burn_in=self.burn_in,
            thinning=self.thinning,
        )

        # Compute the Gaussian conditional distribution
        # f_unique | g^{(r)} ~ N(
        #                        K_unique @ W.T @ K^{-1}_g @ g^{(r)},
        #                        K_unique - K_unique @ W.T @ K^{-1}_g @ W @ K_unique
        #                        )

        # Cross-covariance Cov(f_unique, g).
        C_fg = K_unique @ W.T  # (n_unique, num_duels)

        # K_g^{-1} g_samples.
        solved_samples = torch.cholesky_solve(
            self.g_samples,
            self._K_g_chol,
        )

        # Conditional means for every Gibbs sample.
        #
        # Shape: (n_unique, sample_size)
        conditional_means = C_fg @ solved_samples

        # Cov(f_unique | g), independent of the sampled g.
        solved_cross = torch.cholesky_solve(
            C_fg.T,
            self._K_g_chol,
        )

        conditional_cov = K_unique - C_fg @ solved_cross
        conditional_cov = 0.5 * (
            conditional_cov + conditional_cov.T
        )

        # Exact posterior mean estimated by Monte Carlo.
        self.f_mean = conditional_means.mean(dim=1)

        # Total covariance:
        #
        # Cov(f | g > 0)
        # = E[Cov(f | g)] + Cov(E[f | g]).
        centered_means = (
            conditional_means - self.f_mean.unsqueeze(1)
        )

        mixture_cov = (
            centered_means @ centered_means.T
        ) / sample_size

        self.f_cov = conditional_cov + mixture_cov
        self.f_cov = 0.5 * (
            self.f_cov + self.f_cov.T
        )

        return self.f_mean, self.f_cov

    @torch.inference_mode()
    def conditional_predict(
        self,
        X_test: torch.Tensor,
        is_full_cov: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        For prediction points f_t, predict p(f_t | g^{r}), given g^{r}, which
          is a Gaussian.
        Note that the posterior predictive distribution p(f_t | g > 0) is not
        a Gaussian distribution. Hence, the confidence interval can't be, e.g.,
        mean +- 1.96 * std.

        Args:
            X_test (torch.Tensor): the points to be evaluated, f(X_test)
            is_full_cov (torch.Tensor): if False, return only variances of f(X_test).
        Returns:
            torch.Tensor: conditional means, [ E[f(X_test) | g^{(1)], ... E[f(X_test) | g^{(R)}],
              given g_samples := [g^{(1)}, .... g^{(R)}].
            torch.Tensor: conditional variance/covariance, it isn't affected by g_samples.
        """
        if self.g_samples is None or self._K_g_chol is None:
            raise RuntimeError("Call inference before prediction.")

        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(0)

        X_test = X_test.to(
            device=self.X_unique.device,
            dtype=self.X_unique.dtype,
        )

        W = self.W.to(
            device=X_test.device,
            dtype=X_test.dtype,
        )

        # Cov(f_test, g).
        K_test_unique = self.kernel(
            X_test,
            self.X_unique,
        )
        C_test_g = K_test_unique @ W.T

        solved_samples = torch.cholesky_solve(
            self.g_samples,
            self._K_g_chol,
        )

        # One conditional mean per Gibbs sample.
        conditional_means = C_test_g @ solved_samples

        # K_g X = C_test_g -> X (solved_cross) = K_g^{-1} C_test_g.
        solved_cross = torch.cholesky_solve(
            C_test_g.T,
            self._K_g_chol,
        )

        if is_full_cov:
            K_test = self.kernel(X_test)
            # Compute the covariance
            conditional_cov = (
                K_test - C_test_g @ solved_cross
            )
            # Symmetrization
            conditional_cov = 0.5 * (
                conditional_cov + conditional_cov.T
            )

            return conditional_means, conditional_cov

        prior_var = self.kernel(X_test).diagonal()

        conditional_var = (
            prior_var
            - torch.sum(
                C_test_g * solved_cross.T,
                dim=1,
            )
        ).clamp_min(0.0) # To prevent very small negative numbers (e.g., -1e-15)

        return conditional_means, conditional_var


    @torch.inference_mode()
    def predictive_mean_and_cov(
        self,
        X_test: torch.Tensor,
        is_full_cov: bool | None = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditional_means, conditional_cov_or_var = (
            self.conditional_predict(
                X_test,
                is_full_cov=is_full_cov,
            )
        )

        # conditional_means:
        # (n_test, sample_size)
        pred_mean = conditional_means.mean(dim=1)

        centered_means = (
            conditional_means
            - pred_mean.unsqueeze(1)
        )

        if is_full_cov:
            # Cov_g(E[f_test | g])
            mixture_cov = (
                centered_means @ centered_means.T
            ) / conditional_means.shape[1]

            pred_cov = (
                conditional_cov_or_var
                + mixture_cov
            )

            pred_cov = 0.5 * (
                pred_cov + pred_cov.T
            )

            return pred_mean, pred_cov

        # Only diagonal:
        # Var_g(E[f_test | g])
        mixture_var = centered_means.square().mean(dim=1)

        pred_var = (
            conditional_cov_or_var
            + mixture_var
        )

        return pred_mean, pred_var

    @torch.inference_mode()
    def win_probability(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        include_observation_noise: bool = True,
    ) -> torch.Tensor:
        r"""
        Compute duel win probability 
        
        \Pr(x_1 \succ x_2 \vert \mathcal{D})

        For each Gibbs sample f(x1) - f(x2) | g^{(r)} ~ (\mu^{(r)}_\Delta, \simga^_\Delta)

        P(f(x1) - f(x2) | \mathcal{D}) \approx \frac{1}{R} \sim_{r=1}^R \Phi(\frac{\mu^{r}_\Delta}{\sigma_\Delta})

        For noisy duel outcome,

        P(f(x1) - f(x2) | g^{(r)}) = \Phi(\frac{\mu^{r}_\Delta}{\sqrt{ \sigma_\Delta^2 + 2\sigma^2 }})

        Args:
            x1 (torch.Tensor): A point to win
            x2 (torch.Tensor): A point to lose
            include_observation_noise (bool):

        Returns:
            torch.Tensor: 
        """
        X_pair = torch.stack([x1, x2], dim=0)

        conditional_means, conditional_cov = (
            self.conditional_predict(
                X_pair,
                is_full_cov=True,
            )
        )

        # Positive means x1 has greater utility.
        difference_mean = (
            conditional_means[0] - conditional_means[1]
        )

        difference_var = (
            conditional_cov[0, 0]
            + conditional_cov[1, 1]
            - 2.0 * conditional_cov[0, 1]
        )

        if include_observation_noise:
            difference_var = (
                difference_var
                + self._pairwise_scale() ** 2
            )

        difference_std = difference_var.clamp_min(
            torch.finfo(difference_var.dtype).eps
        ).sqrt()

        probabilities = torch.special.ndtr(
            difference_mean / difference_std
        )

        return probabilities.mean()

    # TODO: Port the original code to check the equivalence.
    # def predictive_quantile(
    #         self,
    #         X_test: torch.Tensor,
    #         prob = 0.5
    # ):
    #     n_test = X_test.shape[0]

    #     pred_mean, pred_var = self.conditional_predict(X_test, False)
    #     pred_std = torch.sqrt(pred_var)

    #     sigma = torch.normal.icdf(prob)
    #     interval = sigma * pred_std

    #     search_lower = torch.min(pred_mean) + interval
    #     search_upper = torch.max(pred_mean) + interval

    #     evaluate_f = torch.ones(n_test)
    #     center_cdf = torch.ones(n_test)
    #     uncoveraged_idx = torch.arange(n_test)

    #     while torch.any(uncoveraged_idx):
    #         evaluate_f[uncoveraged_idx] = (


    #         ) / 2.0 + search_lower[uncoveraged_idx]

    #     return evaluate_f

    @torch.inference_mode()
    def predictive_quantile(
        self,
        X_test: torch.Tensor,
        prob: float = 0.5,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> torch.Tensor:
        r"""
        Compute pointwise posterior predictive quantiles.

        Finds q_p(x) satisfying

            P(f(x) <= q_p(x) | D) = prob

        where the posterior predictive distribution is approximated by

            p(f(x) | D)
            ≈ (1 / R) sum_r
            N(f(x); mu_r(x), sigma_cond^2(x)).

        Args:
            X_test:
                Test points with shape (n_test, input_dim).

            prob:
                Target quantile probability in (0, 1).

            tol:
                Bisection tolerance.

            max_iter:
                Maximum number of bisection iterations.

        Returns:
            Tensor with shape (n_test,), containing the posterior
            predictive quantile at each test point.
        """
        prob = float(prob)

        if not (0.0 < prob < 1.0):
            raise ValueError(
                f"prob must satisfy 0 < prob < 1. Got {prob}."
            )

        # conditional_means:
        #   shape (n_test, sample_size)
        #
        # conditional_var:
        #   shape (n_test,)
        conditional_means, conditional_var = self.conditional_predict(
            X_test,
            is_full_cov=False,
        )

        conditional_std = conditional_var.clamp_min(
            torch.finfo(conditional_var.dtype).eps
        ).sqrt()

        p = torch.tensor(
            prob,
            dtype=conditional_means.dtype,
            device=conditional_means.device,
        )

        # z_p = Phi^{-1}(p)
        z_p = torch.special.ndtri(p)

        # Each mixture component has p-quantile
        #
        # q_r(x) = mu_r(x) + z_p * sigma_cond(x)
        component_quantiles = (
            conditional_means
            + z_p * conditional_std.unsqueeze(1)
        )

        # The mixture p-quantile is bracketed by the smallest
        # and largest component p-quantiles.
        search_lower = component_quantiles.min(dim=1).values
        search_upper = component_quantiles.max(dim=1).values

        for _ in range(max_iter):
            midpoint = 0.5 * (
                search_lower + search_upper
            )

            # Mixture CDF evaluated at midpoint:
            #
            # F_x(q)
            # = (1/R) sum_r
            #   Phi((q - mu_r(x)) / sigma_cond(x))
            standardized = (
                midpoint.unsqueeze(1) - conditional_means
            ) / conditional_std.unsqueeze(1)

            center_cdf = torch.special.ndtr(
                standardized
            ).mean(dim=1)

            # If F(midpoint) >= prob,
            # the desired quantile lies to the left.
            move_upper = center_cdf >= p

            search_upper = torch.where(
                move_upper,
                midpoint,
                search_upper,
            )

            search_lower = torch.where(
                move_upper,
                search_lower,
                midpoint,
            )

            if torch.max(
                search_upper - search_lower
            ) <= tol:
                break

        return 0.5 * (
            search_lower + search_upper
        )