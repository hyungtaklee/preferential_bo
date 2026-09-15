import unittest

import numpy as np
import torch

from core.gp import PrefGP, PrefGP_Gibbs, PrefGP_LA
from utils.func import Levy, Powell
from utils.utility import UtilityOracle


class UtilityOracleTests(unittest.TestCase):
    def test_each_candidate_is_evaluated_as_an_independent_point(self) -> None:
        X = torch.tensor(
            [
                [[0.0, 0.0], [-1.0, -1.0]],
                [[1.0, 0.5], [-0.5, 0.25]],
            ],
            dtype=torch.float64,
        )
        oracle = UtilityOracle("Beale")

        duel, observed = oracle.observe_duel(X)
        expected = oracle.utility_func_obj.values(X.reshape(-1, 2).numpy()).reshape(2, 2)

        torch.testing.assert_close(observed, torch.from_numpy(expected))
        self.assertEqual(duel.shape, (2, 2))

    def test_corrected_benchmark_formulas(self) -> None:
        point = np.array([[1.0, 2.0, 3.0, 4.0]])
        self.assertAlmostEqual(Powell().values(point).item(), -1512.0)

        levy_point = np.array([[0.0, 2.0, -1.0, 3.0]])
        w = 1 + (levy_point - 1) / 4.0
        expected = -(
            np.sin(np.pi * w[:, 0]) ** 2
            + np.sum(
                (w[:, :-1] - 1) ** 2
                * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2),
                axis=1,
            )
            + (w[:, -1] - 1) ** 2
            * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
        )
        np.testing.assert_allclose(Levy().values(levy_point).ravel(), expected)


class PrefGPLaplaceTests(unittest.TestCase):
    # PCA probing-direction test retained for comparison.
    # def test_proving_directions_are_leading_principal_directions(self) -> None:
    #     X = torch.tensor(
    #         [
    #             [[0.0], [1.0]],
    #             [[0.0], [1.0]],
    #             [[1.0], [2.0]],
    #             [[0.0], [2.0]],
    #         ],
    #         dtype=torch.float64,
    #     )
    #     Y = torch.tensor(
    #         [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    #         dtype=torch.float64,
    #     )
    #     gp = PrefGP(input_dim=1)
    #     gp.update_observations(X, Y)
    #
    #     directions = gp._find_proving_direction_pca(k=2)
    #     singular_values = torch.linalg.svdvals(gp.W)
    #     expected_energies = singular_values[:2].square()
    #     duel_gram = gp.W.T @ gp.W
    #
    #     self.assertEqual(directions.shape, (2, 3))
    #     torch.testing.assert_close(
    #         directions @ directions.T,
    #         torch.eye(2, dtype=directions.dtype),
    #     )
    #     torch.testing.assert_close(
    #         duel_gram @ directions.T,
    #         directions.T * expected_energies,
    #     )
    #     torch.testing.assert_close(
    #         torch.sum((gp.W @ directions.T).square(), dim=0),
    #         expected_energies,
    #     )
    #
    #     largest_indices = torch.argmax(torch.abs(directions), dim=1)
    #     largest_components = directions.gather(
    #         1,
    #         largest_indices.unsqueeze(1),
    #     ).squeeze(1)
    #     self.assertTrue(torch.all(largest_components > 0.0))
    #
    #     with self.assertRaisesRegex(ValueError, "numerical rank"):
    #         gp._find_proving_direction_pca(k=3)

    def test_proving_directions_match_posterior_cross_covariance(self) -> None:
        X = torch.tensor(
            [
                [[0.0], [1.0]],
                [[1.0], [2.0]],
                [[0.0], [2.0]],
            ],
            dtype=torch.float64,
        )
        Y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        gp = PrefGP_LA(input_dim=1, noise_std=0.2)
        gp.update_observations(X, Y)

        X_test = torch.tensor([[0.25], [1.5]], dtype=torch.float64)
        directions = gp._find_proving_direction(X_test)
        _, _, Lambda = gp._find_map()
        precision = gp.K_unique_inv.to(dtype=directions.dtype)
        K_unique_test = gp.kernel(gp.X_unique, X_test).to(dtype=directions.dtype)
        right_hand_side = precision @ K_unique_test
        hessian, chol = gp._stable_cholesky(
            precision + Lambda.to(dtype=directions.dtype)
        )
        expected = torch.cholesky_solve(right_hand_side, chol).T

        self.assertEqual(directions.shape, (2, 3))
        torch.testing.assert_close(directions, expected)
        torch.testing.assert_close(hessian @ directions.T, right_hand_side)

    def test_objective_derivatives_match_autograd(self) -> None:
        X = torch.tensor(
            [[[0.0], [1.0]], [[1.0], [2.0]], [[0.0], [2.0]]],
            dtype=torch.float64,
        )
        Y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        gp = PrefGP_LA(input_dim=1, noise_std=0.2)
        gp.update_observations(X, Y)

        f = torch.tensor([-0.3, 0.1, 0.7], dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(gp._map_objective(f, gp.W), f)[0]
        hessian = torch.autograd.functional.hessian(
            lambda value: gp._map_objective(value, gp.W),
            f,
        )
        third_derivative = torch.autograd.functional.jacobian(
            lambda value: torch.autograd.functional.hessian(
                lambda inner: gp._map_objective(inner, gp.W),
                value,
                create_graph=True,
            ),
            f,
        )
        z = gp.W @ f.detach() / gp._pairwise_scale()
        ratio = gp._inverse_mills_ratio(z)

        torch.testing.assert_close(
            gp._objective_gradient(f.detach(), gp.W, ratio),
            gradient,
        )
        torch.testing.assert_close(
            gp._objective_hessian(z, ratio, gp.W),
            hessian,
        )
        torch.testing.assert_close(
            gp._objective_third_derivative(z, ratio, gp.W),
            third_derivative,
        )

    def test_probe_matches_local_quadratic_approximation(self) -> None:
        X = torch.tensor(
            [[[0.0], [1.0]], [[1.0], [2.0]], [[0.0], [2.0]]],
            dtype=torch.float64,
        )
        Y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        gp = PrefGP_LA(input_dim=1, noise_std=0.2)
        gp.update_observations(X, Y)
        f_map, _, _ = gp._find_map()
        X_test = torch.tensor([[1.5]], dtype=f_map.dtype)
        direction = gp._find_proving_direction(X_test)[0]
        strength = 1e-3

        mean, covariance, hessian = gp._find_probe(
            f_map,
            direction,
            strength,
        )

        f_probe = f_map + strength * direction
        z_probe = gp.W.to(f_probe) @ f_probe / gp._pairwise_scale()
        ratio = gp._inverse_mills_ratio(z_probe)
        gradient = gp._objective_gradient(f_probe, gp.W, ratio)
        identity = torch.eye(f_map.numel(), dtype=f_map.dtype)

        torch.testing.assert_close(hessian @ covariance, identity)
        torch.testing.assert_close(hessian @ (f_probe - mean), gradient)

    def test_correction_features_match_predictive_finite_differences(self) -> None:
        X = torch.tensor(
            [[[0.0], [1.0]], [[1.0], [2.0]], [[0.0], [2.0]]],
            dtype=torch.float64,
        )
        Y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        gp = PrefGP_LA(input_dim=1, noise_std=0.2)
        gp.update_observations(X, Y)

        X_test = torch.tensor([[0.25], [1.5]], dtype=torch.float64)
        f_map, _, _ = gp._find_map()
        positive_direction = gp._find_proving_direction(X_test)[0]
        negative_direction = -positive_direction
        strength = 5e-2

        positive_probe = gp._find_probe(
            f_map,
            positive_direction,
            strength,
        )
        negative_probe = gp._find_probe(
            f_map,
            negative_direction,
            strength,
        )
        positive_mean, positive_covariance = gp.predict(
            X_test,
            positive_probe[0],
            positive_probe[1],
            is_full_cov=True,
        )
        negative_mean, negative_covariance = gp.predict(
            X_test,
            negative_probe[0],
            negative_probe[1],
            is_full_cov=True,
        )

        direction_features, mean_feature, covariance_feature = (
            gp._construct_correction_features(
                pos_dir=positive_direction,
                neg_dir=negative_direction,
                pos_mean=positive_mean,
                neg_mean=negative_mean,
                pos_cov=positive_covariance,
                neg_cov=negative_covariance,
                strength=strength,
                X_test=X_test,
            )
        )

        K_unique_test = gp.kernel(gp.X_unique, X_test).to(dtype=f_map.dtype)
        projection = K_unique_test.T @ gp.K_unique_inv.to(dtype=f_map.dtype)
        baseline_mean = projection @ f_map
        expected_direction_features = (
            strength * (projection @ positive_direction),
            strength * (projection @ negative_direction),
        )
        expected_mean_feature = (
            positive_mean + negative_mean - 2.0 * baseline_mean
        ) / (strength ** 2)
        expected_covariance_feature = (
            positive_covariance - negative_covariance
        ) / (2.0 * strength)

        torch.testing.assert_close(
            direction_features[0],
            expected_direction_features[0],
        )
        torch.testing.assert_close(
            direction_features[1],
            expected_direction_features[1],
        )
        torch.testing.assert_close(mean_feature, expected_mean_feature)
        torch.testing.assert_close(
            covariance_feature,
            expected_covariance_feature,
        )

        unscaled_features = gp._construct_correction_features(
            pos_dir=positive_direction,
            neg_dir=negative_direction,
            pos_mean=positive_mean,
            neg_mean=negative_mean,
            pos_cov=positive_covariance,
            neg_cov=negative_covariance,
            strength=None,
            X_test=X_test,
        )
        torch.testing.assert_close(
            unscaled_features[0][0],
            projection @ positive_direction,
        )
        torch.testing.assert_close(
            unscaled_features[1],
            positive_mean + negative_mean - 2.0 * baseline_mean,
        )
        torch.testing.assert_close(
            unscaled_features[2],
            positive_covariance - negative_covariance,
        )

    def test_small_noise_map_is_finite_and_decreases_objective(self) -> None:
        grid = torch.linspace(-1.0, 1.0, 7)
        X = torch.stack((grid[1:], grid[:-1]), dim=1).unsqueeze(-1)
        Y = X.squeeze(-1)
        gp = PrefGP_LA(input_dim=1, noise_std=1e-6)
        gp.update_observations(X, Y)

        mean, covariance, Lambda = gp._find_map()
        zero = torch.zeros_like(mean)

        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(covariance).all())
        self.assertTrue(torch.isfinite(Lambda).all())
        self.assertLess(
            gp._map_objective(mean, gp.W).item(),
            gp._map_objective(zero, gp.W).item(),
        )

        pred_mean, pred_var = gp.predict(grid.unsqueeze(-1), mean, covariance)
        self.assertEqual(pred_mean.shape, (7,))
        self.assertTrue(torch.all(pred_var >= 0.0))

    def test_add_observation_updates_count_and_rejects_reverse_duplicate(self) -> None:
        X = torch.tensor([[[0.0], [1.0]]], dtype=torch.float64)
        Y = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        gp = PrefGP(input_dim=1)
        gp.update_observations(X, Y)

        gp.add_observations(
            torch.tensor([[1.0], [2.0]], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
        self.assertEqual(gp.num_duels, 2)

        with self.assertRaisesRegex(ValueError, "Duplicated duel"):
            gp.add_observations(
                torch.tensor([[2.0], [1.0]], dtype=torch.float64),
                torch.tensor([1.0, 0.0], dtype=torch.float64),
            )


class PrefGPGibbsTests(unittest.TestCase):
    @staticmethod
    def _fitted_gp() -> PrefGP_Gibbs:
        X = torch.tensor(
            [
                [[0.0], [1.0]],
                [[1.0], [2.0]],
                [[0.0], [2.0]],
            ],
            dtype=torch.float64,
        )
        Y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
        )
        gp = PrefGP_Gibbs(
            input_dim=1,
            noise_std=0.2,
            sample_size=400,
            burn_in=150,
            thinning=1,
        )
        gp.update_observations(X, Y)
        return gp

    def test_inference_returns_finite_symmetric_posterior_moments(self) -> None:
        torch.manual_seed(7)
        gp = self._fitted_gp()

        mean, covariance = gp.inference()

        self.assertEqual(mean.shape, (3,))
        self.assertEqual(covariance.shape, (3, 3))
        self.assertEqual(gp.g_samples.shape, (3, 400))
        self.assertTrue(torch.all(gp.g_samples >= 0.0))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(covariance).all())
        torch.testing.assert_close(covariance, covariance.T)
        self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-10)

    def test_predictive_mixture_at_training_points_matches_inference(self) -> None:
        torch.manual_seed(11)
        gp = self._fitted_gp()
        posterior_mean, posterior_covariance = gp.inference()

        conditional_means, conditional_covariance = gp.conditional_predict(
            gp.X_unique,
            is_full_cov=True,
        )
        predictive_mean = conditional_means.mean(dim=1)
        centered = conditional_means - predictive_mean.unsqueeze(1)
        predictive_covariance = (
            conditional_covariance
            + centered @ centered.T / conditional_means.shape[1]
        )

        torch.testing.assert_close(predictive_mean, posterior_mean)
        torch.testing.assert_close(predictive_covariance, posterior_covariance)

    def test_win_probabilities_are_complementary_when_pair_is_reversed(self) -> None:
        torch.manual_seed(19)
        gp = self._fitted_gp()
        gp.inference()
        x1 = torch.tensor([0.25], dtype=torch.float64)
        x2 = torch.tensor([1.75], dtype=torch.float64)

        forward = gp.win_probability(x1, x2)
        reverse = gp.win_probability(x2, x1)

        self.assertGreaterEqual(forward.item(), 0.0)
        self.assertLessEqual(forward.item(), 1.0)
        torch.testing.assert_close(
            forward + reverse,
            torch.ones((), dtype=forward.dtype),
        )


if __name__ == "__main__":
    unittest.main()
