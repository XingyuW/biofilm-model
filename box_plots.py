"""
One-Dimensional Biofilm Model: thickness-keyed visualization suite.

Computation is performed in Rust (`biofilm_core`) and plotting/export are handled in Python.

Deliverables generated for L in {10, 100, 200, 300, 400, 500} µm:
1) Concentration trajectories (line + box)
2) Kinetic-rate trajectories (line + box)
3) Flux trajectories (line + box)
4) Derived biomass trajectories (line + box)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import biofilm_core

import src.fformat as fformat

fformat.ensure_initialized()  # Set up matplotlib parameters for consistent styling across figures.

FIGPATH = "./figures/monod_advanced/SI/"
THICKNESS_VALUES_UM = [10.0, 100.0, 200.0, 300.0, 400.0, 500.0]
N_TRAJECTORY_POINTS = 100
N_REPLICATES = 12
RANDOM_SEED = 20260222
PERTURBATION_STD = 0.03


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    y_label: str
    filename_stem: str


METRICS: list[MetricSpec] = [
    MetricSpec(
        key="substrate_concentration",
        title="Concentration Trajectories Across Biofilm Thickness",
        y_label="Mean Substrate Conc (µM)",
        filename_stem="concentration",
    ),
    MetricSpec(
        key="consumption_rate",
        title="Kinetic-Rate Trajectories Across Biofilm Thickness",
        # y_label="Depth-Averaged Consumption Rate (µM/s)",
        y_label="Consumption Rate (µM/s)",
        filename_stem="kinetics",
    ),
    MetricSpec(
        key="flux",
        title="Flux J Trajectories Across Biofilm Thickness",
        y_label="Diffusion Flux (µM·µm/s)",
        filename_stem="flux",
    ),
    MetricSpec(
        key="biomass",
        title="Derived Biomass X Trajectories Across Biofilm Thickness",
        y_label="Derived Biomass (relative·m)",
        filename_stem="biomass",
    ),
]


def save_fig(fig: Figure, base_name: str) -> None:
    os.makedirs(FIGPATH, exist_ok=True)
    fig.savefig(os.path.join(FIGPATH, f"{base_name}.png"), dpi=600, transparent=True)
    fig.savefig(os.path.join(FIGPATH, f"{base_name}.pdf"), dpi=600, transparent=True)
    plt.close(fig)


def build_thickness_color_map(thicknesses_um: list[float]) -> dict[float, Any]:
    ordered = sorted(thicknesses_um)
    cmap = plt.get_cmap("viridis")
    if len(ordered) == 1:
        return {ordered[0]: cmap(0.6)}
    return {thickness: cmap(i / (len(ordered) - 1)) for i, thickness in enumerate(ordered)}


def _dictify_nested_arrays(raw: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            converted[key] = _dictify_nested_arrays(value)
        else:
            converted[key] = value
    return converted


def _to_growth_trajectories(raw: Any) -> dict[float, dict[str, np.ndarray]]:
    trajectories: dict[float, dict[str, np.ndarray]] = {}
    for terminal, traj in dict(raw).items():
        trajectories[float(terminal)] = _dictify_nested_arrays(dict(traj))
    return trajectories


def _get_nominal_growth_trajectories(
    thicknesses_um: list[float],
    n_points: int,
) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, Any]]:
    run_model = cast(Callable[..., Any], getattr(biofilm_core, "run_model"))
    result = run_model(None, thicknesses_um, n_points)
    params = dict(result["params"])
    trajectories = _to_growth_trajectories(result["growth_trajectories"])
    return trajectories, params


def _build_replicate_params(
    base_params: dict[str, Any],
    rng: np.random.Generator,
    perturbation_std: float,
) -> dict[str, Any]:
    params = dict(base_params)

    # Minimal uncertainty mechanism: replicate-wise parameter perturbation (3% std)
    # around calibrated values to form endpoint distributions for box summaries.
    for key in ("D", "v_max", "K_s"):
        factor = float(rng.lognormal(mean=0.0, sigma=perturbation_std))
        params[key] = float(base_params[key]) * factor

    return params


def _compute_replicate_terminal_values(
    thicknesses_um: list[float],
    base_params: dict[str, Any],
    n_replicates: int,
    n_points: int,
    seed: int,
    perturbation_std: float,
) -> dict[str, dict[float, list[float]]]:
    rng = np.random.default_rng(seed)
    compute_all_growth_trajectories = cast(
        Callable[..., Any],
        getattr(biofilm_core, "compute_all_growth_trajectories"),
    )
    metric_samples: dict[str, dict[float, list[float]]] = {
        metric.key: {thickness: [] for thickness in thicknesses_um}
        for metric in METRICS
    }

    for _ in range(n_replicates):
        perturbed_params = _build_replicate_params(base_params, rng, perturbation_std)
        trajectories_raw = compute_all_growth_trajectories(
            thicknesses_um,
            perturbed_params,
            None,
            n_points,
        )
        trajectories = _to_growth_trajectories(trajectories_raw)

        for thickness in thicknesses_um:
            traj = trajectories[thickness]
            for metric in METRICS:
                terminal_value = float(traj[metric.key][-1])
                metric_samples[metric.key][thickness].append(terminal_value)

    return metric_samples


def _terminal_values_by_metric(
    trajectories: dict[float, dict[str, np.ndarray]],
) -> dict[str, list[float]]:
    ordered_thicknesses = sorted(trajectories.keys())
    values: dict[str, list[float]] = {
        metric.key: [] for metric in METRICS
    }
    for thickness in ordered_thicknesses:
        traj = trajectories[thickness]
        for metric in METRICS:
            values[metric.key].append(float(traj[metric.key][-1]))
    return values


def run_diagnostics(
    thicknesses_um: list[float],
    nominal_trajectories: dict[float, dict[str, np.ndarray]],
    colors: dict[float, Any],
) -> None:
    print("\n" + "-" * 64)
    print("DIAGNOSTICS: definitions, normalization, and trend checks")
    print("-" * 64)

    compute_outputs = cast(Callable[..., Any], getattr(biofilm_core, "compute_outputs"))
    terminal_values = _terminal_values_by_metric(nominal_trajectories)

    print("\nTerminal values by thickness (reported metric definitions):")
    for metric in METRICS:
        series = terminal_values[metric.key]
        print(f"  {metric.filename_stem:14s}: {series}")

    print("\nMetric definitions used for trend interpretation:")
    print("  concentration   = depth-averaged substrate concentration C̄ (µM)")
    print("  kinetics        = depth-averaged effective consumption r̄ = (1/L)∫r(x)dx (µM/s)")
    print("  flux            = wastewater-interface diffusion flux J(0) (µM·µm/s)")
    print("  biomass         = areal biomass X = ∫ρ(x)dx_m (relative·m)")

    expected_trends = {
        "consumption_rate": (
            "decreasing",
            "Kinetics should decrease with L (diffusion-limited depth-averaged rate)",
        ),
        "substrate_concentration": (
            "decreasing",
            "Mean substrate C̄ should decrease with L",
        ),
        "flux": (
            "decreasing",
            "Wastewater-interface flux should decrease with L",
        ),
        "biomass": (
            "increasing",
            "Areal biomass should increase with L",
        ),
    }

    print("\nMonotonicity checks (requested behavior):")
    for key, (direction, description) in expected_trends.items():
        series = np.array(terminal_values[key], dtype=float)
        diffs = np.diff(series)
        if direction == "decreasing":
            passed = bool(np.all(diffs <= 1e-10))
        else:
            passed = bool(np.all(diffs >= -1e-10))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {description}")
        if not passed:
            raise RuntimeError(
                f"Trend check failed for {key}. Values: {series.tolist()} Deltas: {diffs.tolist()}"
            )

    print("\nDefinition/normalization checks per thickness:")
    print(
        "  Columns: L, C_left, C_mid, C_right, C_mean, R_avg(report), "
        "R_int(raw), X_areal, X_areal/L, J(0), J(L), MB_residual"
    )

    mass_balance_abs_tolerance = 5e-3
    mass_balance_rel_tolerance = 2e-3
    for thickness in thicknesses_um:
        out = dict(compute_outputs(thickness, None, None))
        substrate_metrics = dict(out["substrate_metrics"])
        x_um = np.array(out["x_um"])
        c_profile = np.array(out["C"])
        r_local = np.array(out["R"])
        j_local = np.array(out["J"])

        r_avg_reported = float(out["consumption_rate"])
        x_areal = float(out["biomass"])
        l_safe = float(thickness)
        r_integrated = float(np.trapezoid(r_local, x_um))
        r_avg_from_integral = r_integrated / l_safe
        x_per_thickness = x_areal / l_safe

        if abs(r_avg_from_integral - r_avg_reported) > 1e-6:
            raise RuntimeError(
                "Consumption definition mismatch at "
                f"L={thickness}: expected r_avg={r_avg_from_integral}, reported={r_avg_reported}"
            )

        mass_balance_residual = abs((j_local[0] - j_local[-1]) - r_integrated)
        mass_balance_reference = max(abs(r_integrated), abs(j_local[0] - j_local[-1]), 1.0)
        mass_balance_tolerance = max(
            mass_balance_abs_tolerance,
            mass_balance_rel_tolerance * mass_balance_reference,
        )
        if mass_balance_residual > mass_balance_tolerance:
            raise RuntimeError(
                f"Mass balance failed at L={thickness}: residual={mass_balance_residual}, "
                f"J(0)-J(L)={j_local[0]-j_local[-1]}, integral={r_integrated}, "
                f"tolerance={mass_balance_tolerance}"
            )

        print(
            "  "
            f"L={int(thickness):3d} µm, "
            f"C_left={substrate_metrics['C_left']:.3f}, "
            f"C_mid={substrate_metrics['C_mid']:.3f}, "
            f"C_right={substrate_metrics['C_right']:.3f}, "
            f"C_mean={substrate_metrics['C_mean']:.3f}, "
            f"R_avg(report)={r_avg_reported:.6f}, "
            f"R_int(raw)={r_integrated:.6f}, "
            f"X_areal={x_areal:.9f}, "
            f"X_areal/L={x_per_thickness:.9f}, "
            f"J(0)={j_local[0]:.6f}, "
            f"J(L)={j_local[-1]:.6f}, "
            f"MB_residual={mass_balance_residual:.6e}"
        )

        if c_profile.ndim != 1 or r_local.ndim != 1:
            raise RuntimeError("Profile outputs are not one-dimensional arrays as expected.")

    plot_diagnostic_profiles(thicknesses_um, colors, compute_outputs)


def plot_diagnostic_profiles(
    thicknesses_um: list[float],
    colors: dict[float, Any],
    compute_outputs: Callable[..., Any],
) -> tuple[Figure, Figure]:
    fig_s, ax_s = plt.subplots(figsize=(10, 6))
    fig_r, ax_r = plt.subplots(figsize=(10, 6))

    for thickness in thicknesses_um:
        out = dict(compute_outputs(thickness, None, None))
        x_um = np.array(out["x_um"])
        c_profile = np.array(out["C"])
        r_profile = np.array(out["R"])

        label = f"L = {int(thickness)} µm"
        ax_s.plot(x_um, c_profile, color=colors[thickness], linewidth=2.2, label=label)
        ax_r.plot(x_um, r_profile, color=colors[thickness], linewidth=2.2, label=label)

    ax_s.set_xlabel("Depth Coordinate (µm)")
    ax_s.set_ylabel("Substrate concentration (µM)")
    ax_s.grid(False)
    ax_s.legend(loc="best", fontsize=18, ncol=2)

    ax_r.set_xlabel("Depth Coordinate (µm)")
    ax_r.set_ylabel("Local reaction rate (µM/s)")
    ax_r.grid(False)
    ax_r.legend(loc="best", fontsize=18, ncol=2)

    save_fig(fig_s, "SI_diagnostics_substrate_profiles_by_L")
    save_fig(fig_r, "SI_diagnostics_reaction_rate_profiles_by_L")
    return fig_s, fig_r


def plot_metric_line_overlay(
    metric: MetricSpec,
    trajectories: dict[float, dict[str, np.ndarray]],
    colors: dict[float, Any],
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    for thickness in sorted(trajectories.keys()):
        traj = trajectories[thickness]
        ax.plot(
            traj["growth_axis_um"],
            traj[metric.key],
            color=colors[thickness],
            label=f"L = {int(thickness)} µm",
        )

    ax.set_xlabel("Growth-state Thickness (µm)")
    ax.set_ylabel(metric.y_label)
    ax.legend(loc="best", fontsize=18, ncol=2)
    ax.grid(False)
    save_fig(fig, f"SI_thickness_{metric.filename_stem}_line")
    return fig


def plot_metric_box_summary(
    metric: MetricSpec,
    metric_samples: dict[float, list[float]],
    colors: dict[float, Any],
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    ordered_thicknesses = sorted(metric_samples.keys())
    data = [metric_samples[thickness] for thickness in ordered_thicknesses]

    box = ax.boxplot(
        data,
        tick_labels=[str(int(thickness)) for thickness in ordered_thicknesses],
        patch_artist=True,
        showfliers=True,
    )

    for patch, thickness in zip(box["boxes"], ordered_thicknesses):
        patch.set_facecolor(colors[thickness])
        patch.set_alpha(0.55)
        patch.set_edgecolor(colors[thickness])
        patch.set_linewidth(1.3)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel("Biofilm Thickness (µm)")
    ax.set_ylabel(metric.y_label)
    ax.grid(False)
    save_fig(fig, f"SI_thickness_{metric.filename_stem}_box")
    return fig


def main(diagnostics: bool = False) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, dict[float, list[float]]], dict[float, Any]]:
    print("=" * 64)
    print("BIOFILM MODEL: THICKNESS-KEYED VISUALIZATION SUITE")
    print("=" * 64)

    print("\nRunning nominal model for line overlays...")
    nominal_trajectories, params = _get_nominal_growth_trajectories(
        THICKNESS_VALUES_UM,
        N_TRAJECTORY_POINTS,
    )
    colors = build_thickness_color_map(THICKNESS_VALUES_UM)

    print("\nGenerating replicate ensemble for box summaries...")
    print(
        "Replicate mechanism: lognormal perturbation of D, v_max, and K_s "
        f"(σ={PERTURBATION_STD:.3f}) with seed {RANDOM_SEED}"
    )
    metric_samples = _compute_replicate_terminal_values(
        THICKNESS_VALUES_UM,
        params,
        N_REPLICATES,
        N_TRAJECTORY_POINTS,
        RANDOM_SEED,
        PERTURBATION_STD,
    )

    print("\nCreating publication-ready figures...")
    for metric in METRICS:
        plot_metric_line_overlay(metric, nominal_trajectories, colors)
        plot_metric_box_summary(metric, metric_samples[metric.key], colors)
        print(f"  ✓ {metric.filename_stem}: line + box")

    if diagnostics:
        run_diagnostics(THICKNESS_VALUES_UM, nominal_trajectories, colors)

    print("\nSaved figures:")
    print(f"  {FIGPATH}SI_thickness_<metric>_line.(png|pdf)")
    print(f"  {FIGPATH}SI_thickness_<metric>_box.(png|pdf)")
    print("where <metric> ∈ {concentration, kinetics, flux, biomass}")
    if diagnostics:
        print(f"  {FIGPATH}SI_diagnostics_substrate_profiles_by_L.(png|pdf)")
        print(f"  {FIGPATH}SI_diagnostics_reaction_rate_profiles_by_L.(png|pdf)")

    print("\n" + "=" * 64)
    print("Execution complete.")
    print("=" * 64)

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()

    return nominal_trajectories, metric_samples, colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run biofilm thickness-keyed visualization workflow.")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run mechanistic trend/unit/normalization diagnostics and assertions.",
    )
    args = parser.parse_args()
    main(diagnostics=args.diagnostics)
