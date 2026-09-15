import argparse
import math
from pathlib import Path

import torch

from core.gp import PrefGP_Gibbs, PrefGP_LA
from utils.utility import UtilityOracle


def _true_utility(oracle: UtilityOracle, points: torch.Tensor) -> torch.Tensor:
    """Evaluate the noiseless utility while preserving tensor dtype/device."""
    values = oracle.utility_func_obj.values(
        points.detach().cpu().numpy(),
        fidelity=oracle.fidelity,
    )
    return torch.as_tensor(values, dtype=points.dtype, device=points.device).reshape(-1)


def _normalize_true_utility(true_values: torch.Tensor) -> torch.Tensor:
    """Standardize utility to the zero-mean, unit-variance GP latent scale."""
    centered = true_values - true_values.mean()
    standard_deviation = centered.square().mean().sqrt()
    if standard_deviation <= torch.finfo(true_values.dtype).eps:
        return torch.zeros_like(true_values)
    return centered / standard_deviation


def _descending_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return average ranks with rank 1 assigned to the greatest utility."""
    from scipy.stats import rankdata

    ranks = rankdata(-values.detach().cpu().double().numpy(), method="average")
    return torch.as_tensor(ranks, dtype=torch.float64)


def _mixture_moments(
    conditional_means: torch.Tensor,
    conditional_covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine equally weighted Gaussian conditional distributions."""
    mean = conditional_means.mean(dim=1)
    centered = conditional_means - mean.unsqueeze(1)
    covariance = (
        conditional_covariance
        + centered @ centered.T / conditional_means.shape[1]
    )
    return mean, 0.5 * (covariance + covariance.T)


def _normal_win_probability(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    observation_noise_std: float,
) -> torch.Tensor:
    difference_mean = mean[0] - mean[1]
    difference_variance = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2.0 * covariance[0, 1]
        + 2.0 * observation_noise_std**2
    )
    difference_std = difference_variance.clamp_min(
        torch.finfo(covariance.dtype).eps
    ).sqrt()
    return torch.special.ndtr(difference_mean / difference_std)


def _pairwise_preference_accuracy(
    true_values: torch.Tensor,
    estimated_values: torch.Tensor,
) -> torch.Tensor:
    """Return concordant-pair accuracy without allocating an n-by-n matrix."""
    correct = torch.zeros((), dtype=torch.float64, device=true_values.device)
    comparable = 0
    for index in range(true_values.numel() - 1):
        true_differences = true_values[index] - true_values[index + 1 :]
        estimated_differences = (
            estimated_values[index] - estimated_values[index + 1 :]
        )
        non_ties = true_differences != 0.0
        estimated_signs = torch.sign(estimated_differences[non_ties])
        true_signs = torch.sign(true_differences[non_ties])
        correct += (estimated_signs == true_signs).sum(dtype=torch.float64)
        # A predicted tie contains half the information of a strict ordering.
        correct += 0.5 * (estimated_signs == 0.0).sum(dtype=torch.float64)
        comparable += int(non_ties.sum().item())
    if comparable == 0:
        return torch.full((), float("nan"), dtype=torch.float64)
    return correct / comparable


def _pairwise_difference_rmse(
    true_values: torch.Tensor,
    estimated_values: torch.Tensor,
) -> torch.Tensor:
    squared_error_sum = torch.zeros(
        (), dtype=torch.float64, device=true_values.device
    )
    pair_count = 0
    for index in range(true_values.numel() - 1):
        true_differences = true_values[index] - true_values[index + 1 :]
        estimated_differences = (
            estimated_values[index] - estimated_values[index + 1 :]
        )
        squared_error_sum += (
            estimated_differences - true_differences
        ).double().square().sum()
        pair_count += true_differences.numel()
    return (squared_error_sum / pair_count).sqrt()


def _ranking_metrics(
    true_values: torch.Tensor,
    estimated_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    from scipy.stats import kendalltau, spearmanr

    truth_numpy = true_values.detach().cpu().double().numpy()
    estimate_numpy = estimated_values.detach().cpu().double().numpy()
    spearman = spearmanr(truth_numpy, estimate_numpy).statistic
    kendall = kendalltau(truth_numpy, estimate_numpy).statistic
    predicted_best = int(torch.argmax(estimated_values).item())
    true_best_value = true_values.max()
    simple_regret = true_best_value - true_values[predicted_best]
    return {
        "Spearman rank correlation": torch.as_tensor(spearman),
        "Kendall rank correlation": torch.as_tensor(kendall),
        "pairwise preference accuracy": _pairwise_preference_accuracy(
            true_values, estimated_values
        ).cpu(),
        "simple regret": simple_regret.detach().cpu(),
        "predicted-best index": torch.as_tensor(predicted_best),
    }


def _print_moment_comparison(
    title: str,
    labels: list[str],
    true_values: torch.Tensor,
    la_mean: torch.Tensor,
    la_covariance: torch.Tensor,
    gibbs_mean: torch.Tensor,
    gibbs_covariance: torch.Tensor,
    max_table_points: int,
) -> None:
    mean_difference = gibbs_mean - la_mean
    covariance_difference = gibbs_covariance - la_covariance
    la_variance = la_covariance.diagonal()
    gibbs_variance = gibbs_covariance.diagonal()
    la_error = la_mean - true_values
    gibbs_error = gibbs_mean - true_values

    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'point':<18} {'true':>12} {'LA mean':>12} {'Gibbs mean':>12} "
        f"{'|LA error|':>12} {'|Gibbs error|':>14} {'|LA-Gibbs|':>12}"
    )
    displayed_count = min(len(labels), max_table_points)
    for label, truth, la_m, gibbs_m in zip(
        labels[:displayed_count],
        true_values[:displayed_count],
        la_mean[:displayed_count],
        gibbs_mean[:displayed_count],
    ):
        print(
            f"{label:<18} {truth.item():12.5g} {la_m.item():12.5g} "
            f"{gibbs_m.item():12.5g} {abs(la_m - truth).item():12.5g} "
            f"{abs(gibbs_m - truth).item():14.5g} "
            f"{abs(gibbs_m - la_m).item():12.5g}"
        )
    if displayed_count < len(labels):
        print(f"... {len(labels) - displayed_count} additional points omitted")

    print("\nMarginal posterior variances")
    print(
        f"{'point':<18} {'LA var':>12} {'Gibbs var':>12} {'|var diff|':>12}"
    )
    for label, la_v, gibbs_v in zip(
        labels[:displayed_count],
        la_variance[:displayed_count],
        gibbs_variance[:displayed_count],
    ):
        print(
            f"{label:<18} {la_v.item():12.5g} {gibbs_v.item():12.5g} "
            f"{abs(gibbs_v - la_v).item():12.5g}"
        )
    if displayed_count < len(labels):
        print(f"... {len(labels) - displayed_count} additional points omitted")

    def print_matrix(name: str, matrix: torch.Tensor) -> None:
        print(f"\n{name}")
        print(f"{'row':>6}" + "".join(f"{index:>11}" for index in range(matrix.shape[1])))
        for index, row in enumerate(matrix):
            print(f"{index:>6}" + "".join(f"{value.item():11.5g}" for value in row))

    if len(labels) <= max_table_points:
        print("\nCovariance matrices (rows/columns follow the point order above)")
        print_matrix("LA covariance", la_covariance)
        print_matrix("Gibbs covariance", gibbs_covariance)
        print_matrix("absolute covariance difference", covariance_difference.abs())
    else:
        print(
            "\nFull covariance tables omitted because "
            f"{len(labels)} points exceed --max_table_points={max_table_points}. "
            "Full-matrix difference metrics are still reported below."
        )

    print("\nAggregate differences")
    print(f"{'metric':<42} {'value':>14}")
    centered_truth = true_values - true_values.mean()
    centered_la = la_mean - la_mean.mean()
    centered_gibbs = gibbs_mean - gibbs_mean.mean()
    la_ranking = _ranking_metrics(true_values, la_mean)
    gibbs_ranking = _ranking_metrics(true_values, gibbs_mean)
    metrics = {
        "LA RMSE versus true utility": la_error.square().mean().sqrt(),
        "Gibbs RMSE versus true utility": gibbs_error.square().mean().sqrt(),
        "LA maximum absolute true error": la_error.abs().max(),
        "Gibbs maximum absolute true error": gibbs_error.abs().max(),
        "LA centered RMSE versus truth": (
            centered_la - centered_truth
        ).square().mean().sqrt(),
        "Gibbs centered RMSE versus truth": (
            centered_gibbs - centered_truth
        ).square().mean().sqrt(),
        "LA pairwise-difference RMSE": _pairwise_difference_rmse(
            true_values, la_mean
        ),
        "Gibbs pairwise-difference RMSE": _pairwise_difference_rmse(
            true_values, gibbs_mean
        ),
        "LA Spearman rank correlation": la_ranking[
            "Spearman rank correlation"
        ],
        "Gibbs Spearman rank correlation": gibbs_ranking[
            "Spearman rank correlation"
        ],
        "LA Kendall rank correlation": la_ranking["Kendall rank correlation"],
        "Gibbs Kendall rank correlation": gibbs_ranking[
            "Kendall rank correlation"
        ],
        "LA pairwise preference accuracy": la_ranking[
            "pairwise preference accuracy"
        ],
        "Gibbs pairwise preference accuracy": gibbs_ranking[
            "pairwise preference accuracy"
        ],
        "LA simple regret": la_ranking["simple regret"],
        "Gibbs simple regret": gibbs_ranking["simple regret"],
        "LA predicted-best index": la_ranking["predicted-best index"],
        "Gibbs predicted-best index": gibbs_ranking["predicted-best index"],
        "LA-Gibbs mean RMSE": mean_difference.square().mean().sqrt(),
        "LA-Gibbs mean maximum difference": mean_difference.abs().max(),
        "covariance Frobenius error": torch.linalg.matrix_norm(
            covariance_difference
        ),
        "covariance relative Frobenius error": torch.linalg.matrix_norm(
            covariance_difference
        )
        / torch.linalg.matrix_norm(la_covariance).clamp_min(
            torch.finfo(la_covariance.dtype).eps
        ),
        "covariance maximum absolute error": covariance_difference.abs().max(),
        "LA covariance minimum eigenvalue": torch.linalg.eigvalsh(
            la_covariance
        ).min(),
        "Gibbs covariance minimum eigenvalue": torch.linalg.eigvalsh(
            gibbs_covariance
        ).min(),
    }
    for name, value in metrics.items():
        print(f"{name:<42} {value.item():14.6g}")


def _add_covariance_ellipse(
    axis,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    color: str,
    label: str,
) -> None:
    from matplotlib.patches import Ellipse

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = math.degrees(
        math.atan2(eigenvectors[1, 0].item(), eigenvectors[0, 0].item())
    )
    center = mean.detach().cpu().double()
    for standard_deviations, alpha in ((1.0, 0.25), (2.0, 0.10)):
        radii = standard_deviations * eigenvalues.sqrt()
        axis.add_patch(
            Ellipse(
                xy=center.tolist(),
                width=2.0 * radii[0].item(),
                height=2.0 * radii[1].item(),
                angle=angle,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                label=label if standard_deviations == 1.0 else None,
            )
        )
    axis.scatter(center[0], center[1], color=color, marker="x", s=60)


def _visualize_comparison(
    true_latent_values: torch.Tensor,
    la_latent_mean: torch.Tensor,
    la_latent_covariance: torch.Tensor,
    gibbs_latent_mean: torch.Tensor,
    gibbs_latent_covariance: torch.Tensor,
    la_predictive_mean: torch.Tensor,
    la_predictive_covariance: torch.Tensor,
    gibbs_predictive_mean: torch.Tensor,
    gibbs_predictive_covariance: torch.Tensor,
    true_predictive_values: torch.Tensor,
    la_win_probability: torch.Tensor,
    gibbs_win_probability: torch.Tensor,
    save_path: Path | None,
    show: bool,
    max_table_points: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, (latent_axis, predictive_axis, rank_axis) = plt.subplots(
        1, 3, figsize=(18, 6.2), constrained_layout=True
    )

    point_indices = torch.arange(la_latent_mean.numel()).cpu().numpy()
    normalized_true_latent = _normalize_true_utility(true_latent_values)
    latent_axis.plot(
        point_indices,
        normalized_true_latent.detach().cpu(),
        color="black",
        linestyle="--",
        marker="*",
        markersize=8,
        linewidth=1.5,
        label="true utility (z-score)",
        zorder=4,
    )
    for mean, covariance, color, label, offset in (
        (la_latent_mean, la_latent_covariance, "tab:blue", "LA", -0.05),
        (
            gibbs_latent_mean,
            gibbs_latent_covariance,
            "tab:orange",
            "Gibbs",
            0.05,
        ),
    ):
        standard_deviation = covariance.diagonal().clamp_min(0.0).sqrt()
        latent_axis.errorbar(
            point_indices + offset,
            mean.detach().cpu(),
            yerr=standard_deviation.detach().cpu(),
            color=color,
            marker="o",
            capsize=3,
            label=f"{label} mean +/- posterior SD",
        )
    variance_axis = latent_axis.twinx()
    variance_axis.plot(
        point_indices,
        la_latent_covariance.diagonal().detach().cpu(),
        color="tab:blue",
        linestyle=":",
        alpha=0.7,
        label="LA variance",
    )
    variance_axis.plot(
        point_indices,
        gibbs_latent_covariance.diagonal().detach().cpu(),
        color="tab:orange",
        linestyle=":",
        alpha=0.7,
        label="Gibbs variance",
    )
    latent_axis.set(
        title="Posterior at X_unique",
        xlabel="unique-point index",
        ylabel="posterior utility / normalized true utility",
        xticks=point_indices,
    )
    variance_axis.set_ylabel("posterior variance")
    handles, labels = latent_axis.get_legend_handles_labels()
    variance_handles, variance_labels = variance_axis.get_legend_handles_labels()
    latent_axis.legend(handles + variance_handles, labels + variance_labels, fontsize=8)
    latent_axis.grid(alpha=0.2)

    _add_covariance_ellipse(
        predictive_axis,
        la_predictive_mean,
        la_predictive_covariance,
        "tab:blue",
        f"LA: P(x1 wins)={la_win_probability.item():.3f}",
    )
    _add_covariance_ellipse(
        predictive_axis,
        gibbs_predictive_mean,
        gibbs_predictive_covariance,
        "tab:orange",
        f"Gibbs: P(x1 wins)={gibbs_win_probability.item():.3f}",
    )
    all_means = torch.cat([la_predictive_mean, gibbs_predictive_mean])
    all_stds = torch.cat(
        [
            la_predictive_covariance.diagonal().clamp_min(0.0).sqrt(),
            gibbs_predictive_covariance.diagonal().clamp_min(0.0).sqrt(),
        ]
    )
    plot_min = (all_means.min() - 3.0 * all_stds.max()).item()
    plot_max = (all_means.max() + 3.0 * all_stds.max()).item()
    predictive_axis.plot(
        [plot_min, plot_max], [plot_min, plot_max], "k--", label="equal utility"
    )
    predictive_axis.set(
        title="Joint posterior predictive",
        xlabel="f(x1)",
        ylabel="f(x2)",
        xlim=(plot_min, plot_max),
        ylim=(plot_min, plot_max),
        aspect="equal",
    )
    predictive_axis.grid(alpha=0.2)
    predictive_axis.legend(fontsize=8)
    predictive_axis.text(
        0.02,
        0.02,
        "true f_t = "
        f"({true_predictive_values[0].item():.5g}, "
        f"{true_predictive_values[1].item():.5g})",
        transform=predictive_axis.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    true_ranks = _descending_ranks(true_latent_values)
    la_ranks = _descending_ranks(la_latent_mean)
    gibbs_ranks = _descending_ranks(gibbs_latent_mean)
    rank_table_count = min(
        true_latent_values.numel(), max_table_points, 20
    )
    top_true_indices = torch.argsort(true_latent_values, descending=True)[
        :rank_table_count
    ].detach().cpu()
    rank_rows = [
        [
            f"f_unique[{index.item()}]",
            f"{true_ranks[index].item():g}",
            f"{la_ranks[index].item():g}",
            f"{gibbs_ranks[index].item():g}",
        ]
        for index in top_true_indices
    ]
    rank_axis.axis("off")
    rank_axis.set_title(
        "Utility ranks at X_unique\n"
        f"1 = best; top {rank_table_count} by true utility",
        pad=12,
    )
    rank_table = rank_axis.table(
        cellText=rank_rows,
        colLabels=["point", "true", "LA", "Gibbs"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.31, 1.0, 0.62],
    )
    rank_table.auto_set_font_size(False)
    rank_table.set_fontsize(8)
    rank_table.scale(1.0, min(1.4, 18.0 / rank_table_count))
    for column in range(4):
        rank_table[(0, column)].set_facecolor("0.9")
        rank_table[(0, column)].set_text_props(weight="bold")

    la_rank_metrics = _ranking_metrics(true_latent_values, la_latent_mean)
    gibbs_rank_metrics = _ranking_metrics(
        true_latent_values, gibbs_latent_mean
    )
    metric_rows = [
        [
            "Spearman correlation (higher better)",
            f"{la_rank_metrics['Spearman rank correlation'].item():.4f}",
            f"{gibbs_rank_metrics['Spearman rank correlation'].item():.4f}",
        ],
        [
            "Kendall correlation (higher better)",
            f"{la_rank_metrics['Kendall rank correlation'].item():.4f}",
            f"{gibbs_rank_metrics['Kendall rank correlation'].item():.4f}",
        ],
        [
            "Pairwise accuracy (higher better)",
            f"{la_rank_metrics['pairwise preference accuracy'].item():.4f}",
            f"{gibbs_rank_metrics['pairwise preference accuracy'].item():.4f}",
        ],
        [
            "Simple regret (lower better)",
            f"{la_rank_metrics['simple regret'].item():.5g}",
            f"{gibbs_rank_metrics['simple regret'].item():.5g}",
        ],
    ]
    rank_axis.text(
        0.5,
        0.265,
        "Rank-quality metrics",
        transform=rank_axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    metric_table = rank_axis.table(
        cellText=metric_rows,
        colLabels=["metric", "LA", "Gibbs"],
        colWidths=[0.55, 0.225, 0.225],
        cellLoc="center",
        colLoc="center",
        bbox=[0.05, 0.01, 0.90, 0.22],
    )
    metric_table.auto_set_font_size(False)
    metric_table.set_fontsize(8)
    for column in range(3):
        metric_table[(0, column)].set_facecolor("0.9")
        metric_table[(0, column)].set_text_props(weight="bold")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=160)
        print(f"\nSaved comparison figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=50)
    parser.add_argument(
        "--num_unique",
        type=int,
        default=6,
        metavar="N",
        help=(
            "number of unique training inputs in f_unique (N >= 2; no hard "
            "upper limit, but exact GP time/memory scale cubically/quadratically)"
        ),
    )
    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--obj_func", type=str, default="Beale")
    parser.add_argument("--kernel", type=str, default="RBFKernel")
    parser.add_argument("--kernel_length_scale", type=float, default=1e-1)
    parser.add_argument("--fidelity", type=int, default=None)
    parser.add_argument("--noise_std", type=float, default=5e-1)
    parser.add_argument("--observation_noise", type=float, default=0.0)
    parser.add_argument("--gibbs_samples", type=int, default=1000)
    parser.add_argument("--gibbs_burn_in", type=int, default=500)
    parser.add_argument("--gibbs_thinning", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_table_points",
        type=int,
        default=20,
        help="maximum points shown in detailed tables (full metrics use all points)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show the interactive matplotlib comparison window",
    )
    parser.add_argument(
        "--save_plot",
        type=Path,
        default=None,
        help="write the comparison figure to this path",
    )
    args = parser.parse_args()

    if args.num_unique < 2:
        parser.error("--num_unique must be at least 2")
    if args.max_table_points <= 0:
        parser.error("--max_table_points must be positive")
    if args.num_train < math.ceil(args.num_unique / 2):
        parser.error(
            "--num_train must be at least ceil(--num_unique / 2) so every "
            "unique point appears in the observations"
        )
    return args


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    oracle = UtilityOracle(f=args.obj_func, fidelity=args.fidelity)
    if args.input_dim is not None and args.input_dim != oracle.input_dim:
        raise ValueError(
            f"--input_dim={args.input_dim} does not match "
            f"{args.obj_func}'s dimension ({oracle.input_dim})"
        )

    bounds = torch.as_tensor(oracle.bounds, dtype=torch.float64)
    lower, upper = bounds
    unique_inputs = lower + (upper - lower) * torch.rand(
        args.num_unique,
        oracle.input_dim,
        dtype=torch.float64,
        generator=generator,
    )

    # Cover every unique input before adding random, non-self comparisons.
    pair_indices = []
    for index in range(0, args.num_unique, 2):
        pair_indices.append((index, (index + 1) % args.num_unique))
    while len(pair_indices) < args.num_train:
        pair = torch.randperm(args.num_unique, generator=generator)[:2]
        pair_indices.append((pair[0].item(), pair[1].item()))
    train_x = unique_inputs[torch.tensor(pair_indices, dtype=torch.long)]

    duel, observations = oracle.observe_duel(
        train_x, noise=args.observation_noise
    )

    la_gp = PrefGP_LA(
        kernel_name=args.kernel,
        kernel_length_scale=args.kernel_length_scale,
        input_dim=oracle.input_dim,
        noise_std=args.noise_std,
    )
    la_gp.update_observations(train_x, observations, duel=duel)
    la_mean, la_covariance, la_lambda = la_gp.inference()

    predictive_points = lower + (upper - lower) * torch.rand(
        2,
        oracle.input_dim,
        dtype=train_x.dtype,
        generator=generator,
    )

    # Laplace probing smoke tests use the signatures declared by PrefGP_LA.
    proving_directions = la_gp._find_proving_direction(predictive_points)
    positive_probe = la_gp._find_probe(la_mean, proving_directions[0], 5e-2)
    negative_probe = la_gp._find_probe(la_mean, -proving_directions[0], 5e-2)
    positive_prediction = la_gp.predict(
        predictive_points, *positive_probe[:2], is_full_cov=True
    )
    negative_prediction = la_gp.predict(
        predictive_points, *negative_probe[:2], is_full_cov=True
    )
    direction_features, mean_feature, covariance_feature = (
        la_gp._construct_correction_features(
            pos_dir=proving_directions[0],
            neg_dir=-proving_directions[0],
            pos_mean=positive_prediction[0],
            neg_mean=negative_prediction[0],
            pos_cov=positive_prediction[1],
            neg_cov=negative_prediction[1],
            strength=5e-2,
            X_test=predictive_points,
        )
    )
    assert positive_probe[2].shape == la_covariance.shape
    assert len(direction_features) == 2
    assert mean_feature.shape == (2,)
    assert covariance_feature.shape == (2, 2)

    # Test suite for Gibbs sampling-based Monte Carlo estimators.
    gibbs_gp = PrefGP_Gibbs(
        kernel_name=args.kernel,
        kernel_length_scale=args.kernel_length_scale,
        input_dim=oracle.input_dim,
        noise_std=args.noise_std,
        sample_size=args.gibbs_samples,
        burn_in=args.gibbs_burn_in,
        thinning=args.gibbs_thinning,
    )
    gibbs_gp.update_observations(train_x, observations, duel=duel)
    # Both inference methods must operate on the exact same observations.
    torch.testing.assert_close(la_gp.X_train, gibbs_gp.X_train)
    torch.testing.assert_close(la_gp.Y_train, gibbs_gp.Y_train)
    torch.testing.assert_close(la_gp.duel, gibbs_gp.duel)
    torch.testing.assert_close(la_gp.X_unique, gibbs_gp.X_unique)
    torch.testing.assert_close(la_gp.W, gibbs_gp.W)
    gibbs_mean, gibbs_covariance = gibbs_gp.inference()

    n_unique = la_gp.X_unique.shape[0]
    assert n_unique == args.num_unique
    assert gibbs_mean.shape == (n_unique,)
    assert gibbs_covariance.shape == (n_unique, n_unique)
    assert gibbs_gp.g_samples.shape == (args.num_train, args.gibbs_samples)
    assert torch.isfinite(gibbs_mean).all()
    assert torch.isfinite(gibbs_covariance).all()
    torch.testing.assert_close(gibbs_covariance, gibbs_covariance.T)
    torch.testing.assert_close(gibbs_mean, gibbs_gp.f_mean)
    torch.testing.assert_close(gibbs_covariance, gibbs_gp.f_cov)
    assert torch.linalg.eigvalsh(gibbs_covariance).min().item() >= -1e-10

    la_predictive_mean, la_predictive_covariance = la_gp.predict(
        predictive_points,
        la_mean,
        la_covariance,
        is_full_cov=True,
    )
    conditional_means, conditional_covariance = gibbs_gp.conditional_predict(
        predictive_points, is_full_cov=True
    )
    gibbs_predictive_mean, gibbs_predictive_covariance = _mixture_moments(
        conditional_means, conditional_covariance
    )
    assert conditional_means.shape == (2, args.gibbs_samples)
    assert conditional_covariance.shape == (2, 2)
    assert torch.isfinite(gibbs_predictive_covariance).all()
    torch.testing.assert_close(
        gibbs_predictive_covariance, gibbs_predictive_covariance.T
    )
    assert torch.linalg.eigvalsh(gibbs_predictive_covariance).min().item() >= -1e-10

    la_win_probability = _normal_win_probability(
        la_predictive_mean, la_predictive_covariance, args.noise_std
    )
    gibbs_win_probability = gibbs_gp.win_probability(
        predictive_points[0], predictive_points[1]
    )
    true_latent_values = _true_utility(oracle, la_gp.X_unique)
    true_predictive_values = _true_utility(oracle, predictive_points)
    assert true_latent_values.shape == la_mean.shape
    assert true_predictive_values.shape == la_predictive_mean.shape
    assert torch.isfinite(true_latent_values).all()
    assert torch.isfinite(true_predictive_values).all()
    assert 0.0 <= la_win_probability.item() <= 1.0
    assert 0.0 <= gibbs_win_probability.item() <= 1.0

    print("\nX_unique index mapping")
    mapping_count = min(n_unique, args.max_table_points)
    for index, point in enumerate(la_gp.X_unique[:mapping_count]):
        coordinates = ", ".join(f"{value.item():.5g}" for value in point)
        print(f"f_unique[{index}] = ({coordinates})")
    if mapping_count < n_unique:
        print(f"... {n_unique - mapping_count} additional points omitted")

    _print_moment_comparison(
        "Posterior moments at X_unique",
        [f"f_unique[{index}]" for index in range(n_unique)],
        true_latent_values,
        la_mean,
        la_covariance,
        gibbs_mean,
        gibbs_covariance,
        args.max_table_points,
    )
    _print_moment_comparison(
        "Posterior predictive moments at f_t",
        ["f_t[0]", "f_t[1]"],
        true_predictive_values,
        la_predictive_mean,
        la_predictive_covariance,
        gibbs_predictive_mean,
        gibbs_predictive_covariance,
        args.max_table_points,
    )
    print("\nPosterior win probability P(f_t[0] > f_t[1])")
    print(f"{'LA':<20} {la_win_probability.item():.6f}")
    print(f"{'Gibbs':<20} {gibbs_win_probability.item():.6f}")
    print(
        f"{'absolute difference':<20} "
        f"{abs(la_win_probability - gibbs_win_probability).item():.6f}"
    )
    true_margin = true_predictive_values[0] - true_predictive_values[1]
    if true_margin.item() > 0.0:
        true_winner = "f_t[0]"
    elif true_margin.item() < 0.0:
        true_winner = "f_t[1]"
    else:
        true_winner = "tie"
    print(f"{'true utility margin':<20} {true_margin.item():.6g}")
    print(f"{'true winner':<20} {true_winner}")
    print(
        f"\nutility={args.obj_func}, dimension={oracle.input_dim}, "
        f"duels={la_gp.num_duels}, unique points={n_unique}, "
        f"Gibbs samples={args.gibbs_samples}, Lambda shape={tuple(la_lambda.shape)}"
    )

    if args.plot or args.save_plot is not None:
        _visualize_comparison(
            true_latent_values,
            la_mean,
            la_covariance,
            gibbs_mean,
            gibbs_covariance,
            la_predictive_mean,
            la_predictive_covariance,
            gibbs_predictive_mean,
            gibbs_predictive_covariance,
            true_predictive_values,
            la_win_probability,
            gibbs_win_probability,
            args.save_plot,
            args.plot,
            args.max_table_points,
        )


if __name__ == "__main__":
    main()
