
import torch

import numpy as np
import numpy.typing as npt

from . import func

class UtilityOracle:
    def __init__(self, f: str, fidelity: int | None = None,
                 bounds: npt.NDArray | None = None) -> None:

        # Get utility class object
        f_cls = getattr(func, f, None)

        # If the name cannot be found in the library, raise error
        if f_cls is None or not isinstance(f_cls, type):
            raise ValueError(f"A utility '{f}' cannot be found.")

        self.utility_func_obj = f_cls()
        
        self.input_dim = self.utility_func_obj.d
        self.num_fidelities = self.utility_func_obj.M

        if fidelity is None:
            fidelity = self.num_fidelities - 1
        if (
            not isinstance(fidelity, int)
            or isinstance(fidelity, bool)
            or not 0 <= fidelity < self.num_fidelities
        ):
            raise ValueError(
                f"fidelity must be an integer in [0, {self.num_fidelities - 1}]. "
                f"Got: {fidelity}"
            )
        self.fidelity = fidelity

        if bounds is None:
            self.bounds = np.asarray(self.utility_func_obj.bounds)
        else:
            bounds = np.asarray(bounds)
            expected_shape = (2, self.input_dim)
            if bounds.shape != expected_shape:
                raise ValueError(
                    f"bounds must have shape {expected_shape}. Got: {bounds.shape}"
                )
            if not np.isfinite(bounds).all():
                raise ValueError("bounds must contain only finite values.")
            if np.any(bounds[0] >= bounds[1]):
                raise ValueError("Each lower bound must be smaller than its upper bound.")
            self.bounds = bounds

    def observe_duel(self, X: torch.Tensor, noise=0.0):
        """Observe preferece feedback on X, where 
        X.shape == (n_duel, 2, n_dim) given a utility function f with noise.
        
        Args:
            X (torch.Tensor): a tensor of the points to observe (n_duel, 2, n_dim)
            noise (int): the standard deviation of the zero mean Gaussian noise

        Returns:
            torch.Tensor: A duel observation tensor (n_duel, 2)
            torch.Tensor: A noisy utility observation tensor (n_duel, 2)
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor.")
        if X.ndim != 3 or X.shape[1:] != (2, self.input_dim):
            raise ValueError(
                f"X must have shape (n_duel, 2, {self.input_dim}). "
                f"Got: {tuple(X.shape)}"
            )
        if not X.is_floating_point() or not torch.isfinite(X).all():
            raise ValueError("X must contain finite floating-point values.")

        noise = float(noise)
        if not np.isfinite(noise) or noise < 0.0:
            raise ValueError("noise must be non-negative.")

        flat_X = X.reshape(-1, self.input_dim)
        f_np = self.utility_func_obj.values(
            flat_X.detach().cpu().numpy(),
            fidelity=self.fidelity,
        )
        f = torch.as_tensor(f_np, dtype=X.dtype, device=X.device)
        expected_values = X.shape[0] * 2
        if f.numel() != expected_values:
            raise RuntimeError(
                f"Utility {type(self.utility_func_obj).__name__} returned "
                f"{f.numel()} values for {expected_values} points."
            )
        f = f.reshape(X.shape[0], 2)
        if not torch.isfinite(f).all():
            raise RuntimeError(
                f"Utility {type(self.utility_func_obj).__name__} returned non-finite values."
            )

        e = torch.randn_like(f) * noise

        Y = f + e

        # Construct duel observation
        if torch.any(Y[:, 0] == Y[:, 1]):
            raise ValueError(
                "The utility oracle produced a tie; add observation noise or "
                "provide a tie-breaking policy."
            )
        max_idx = torch.argmax(Y, dim=1)

        duel = torch.zeros_like(Y)
        duel.scatter_(1, max_idx.unsqueeze(1), 1.0) # Place 1 to the winning matrix
        duel = duel * 2.0 - 1.0

        return (duel, Y)


def construct_duel_matrix(X: torch.Tensor, 
                            duel: torch.Tensor, 
                            is_drop_self_duels: bool = False):
    """Construct the duel matrix by flattening X and
    providing the duel matrix W

    Args:
        X (torch.Tensor): A set of duel points (n_duel, 2, n_dim)
        duel (torch.Tensor): A duel observation matrix on X (n_duel, 2)
        is_drop_self_duels (bool): If True, drop the self duels silently,
            Otherwise, raise a ValueError.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Unique points and their duel matrix.
    """

    if not isinstance(X, torch.Tensor) or not isinstance(duel, torch.Tensor):
        raise TypeError("X and duel must be torch.Tensor objects.")
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (n_duel, 2, input_dim). Got: {tuple(X.shape)}")
    if not X.is_floating_point() or not torch.isfinite(X).all():
        raise ValueError("X must contain finite floating-point values.")

    n_duel, _, input_dim = X.shape
    if duel.shape != (n_duel, 2):
        raise ValueError(f"duel must have shape ({n_duel}, 2). Got: {tuple(duel.shape)}")
    if duel.device != X.device:
        raise ValueError("X and duel must be on the same device.")
    if not torch.all((duel == 1) | (duel == -1)):
        raise ValueError("Each duel must contain exactly one +1 and one -1.")

    # Check duel observation
    if not torch.all(duel.sum(dim=1) == 0):
        raise ValueError("Each duel must contain one +1 and one -1.")

    # Flattening X
    X_flatten = X.reshape(-1, input_dim) 
    
    # Remove duplicated X rows.
    # inverse_X[j] tells which unique row X_flatten[j] maps to.
    X_unique, inverse_X = torch.unique(
        X_flatten,
        dim=0,
        sorted=True,
        return_inverse=True,
    )

    n_unique = X_unique.shape[0]

    # Shape: (n_duel, 2)
    col_idx = inverse_X.reshape(n_duel, 2)

    # Construct W entries
    W = torch.zeros(n_duel, n_unique, dtype=X.dtype, device=X.device)
    W.scatter_add_(dim=1, index=col_idx, src=duel.to(dtype=X.dtype))

    # Check if the same point appears on both sides of the duel
    valid = W.abs().sum(dim=1) > 0

    if not torch.all(valid):
        if is_drop_self_duels:
            W = W[valid]
        else:
            bad = torch.where(~valid)[0].detach().cpu().tolist()
            raise ValueError(
                "Some duels compare a point with itself. "
                f"Degenerate duel indices: {bad}. "
                "Set is_drop_self_duels=True to remove them."
            )

    return (X_unique, W)
