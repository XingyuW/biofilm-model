"""
One-Dimensional Biofilm Model (Python plotting + Rust computation)

Computation and model evaluation are performed in Rust via PyO3 (`biofilm_core`).
This script keeps only visualization, layout, and file output in Python.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

import biofilm_core

figpath = "./figures/monod_advanced/SI/"
v_colors = ["#B949BB", "#239509", "#007A87"]


def save_fig(fig: Figure, base_name: str) -> None:
    os.makedirs(figpath, exist_ok=True)
    fig.savefig(os.path.join(figpath, f"{base_name}.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(figpath, f"{base_name}.pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_terminal_state_summary(terminal_summary: list[dict[str, float]]) -> None:
    os.makedirs(figpath, exist_ok=True)
    csv_path = os.path.join(figpath, "SI_growth_terminal_state_summary.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "terminal_thickness_um",
                "terminal_thickness_m",
                "flux_uM_um_per_s",
                "consumption_rate_uM_um_per_s",
                "biomass_relative_m",
                "substrate_Cbar_uM",
            ]
        )
        for row in terminal_summary:
            writer.writerow(
                [
                    row["terminal_thickness_um"],
                    row["terminal_thickness_m"],
                    row["flux_uM_um_per_s"],
                    row["consumption_rate_uM_um_per_s"],
                    row["biomass_relative_m"],
                    row["substrate_Cbar_uM"],
                ]
            )


def save_growth_trajectory_csvs(trajectories: dict[float, dict[str, np.ndarray]]) -> None:
    os.makedirs(figpath, exist_ok=True)
    for terminal_thickness_um in sorted(trajectories.keys()):
        trajectory = trajectories[terminal_thickness_um]
        csv_path = os.path.join(figpath, f"SI_growth_trajectory_L{int(terminal_thickness_um)}um.csv")
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "growth_axis_thickness_um",
                    "evaluated_thickness_um",
                    "evaluated_thickness_m",
                    "flux_uM_um_per_s",
                    "consumption_rate_uM_um_per_s",
                    "biomass_relative_m",
                    "substrate_Cbar_uM",
                ]
            )
            for i in range(len(trajectory["growth_axis_um"])):
                writer.writerow(
                    [
                        trajectory["growth_axis_um"][i],
                        trajectory["eval_thickness_um"][i],
                        trajectory["thickness_m"][i],
                        trajectory["flux"][i],
                        trajectory["consumption_rate"][i],
                        trajectory["biomass"][i],
                        trajectory["substrate_concentration"][i],
                    ]
                )


def plot_substrate_profiles(no_voltage: dict[str, np.ndarray], voltage_data: dict[float, dict[str, np.ndarray]], params: dict[str, Any]) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(no_voltage["x"], no_voltage["C"], "b-", label="No Voltage", zorder=4)

    linestyles = ["-", "--", ":"]
    for i, voltage in enumerate(sorted(voltage_data.keys())):
        profile = voltage_data[voltage]
        ax.plot(
            profile["x"],
            profile["C"],
            color=v_colors[i],
            linestyle=linestyles[i],
            label=f"{voltage} V Applied",
            zorder=3 - i,
        )

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=params["L"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Position x (µm)")
    ax.set_ylabel("Substrate Concentration (µM)")
    ax.legend(fontsize=20, loc="best")
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xlim(-5, 505)
    save_fig(fig, f"SI_biofilm_substrate_profiles_{int(params['L'])}um")
    return fig


def plot_consumption_profiles(no_voltage: dict[str, np.ndarray], voltage_data: dict[float, dict[str, np.ndarray]], params: dict[str, Any]) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(no_voltage["x"], no_voltage["R"], "b-", label="No Voltage", zorder=4)

    linestyles = ["-", "--", ":"]
    for i, voltage in enumerate(sorted(voltage_data.keys())):
        profile = voltage_data[voltage]
        ax.plot(
            profile["x"],
            profile["R"],
            color=v_colors[i],
            linestyle=linestyles[i],
            label=f"{voltage} V Applied",
            zorder=3 - i,
        )

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=params["L"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Biofilm Thickness x (µm)")
    ax.set_ylabel("Consumption Rate (µM/s)")
    ax.legend(fontsize=20, loc="best")
    ax.set_position((0.15, 0.15, 0.7, 0.8))
    save_fig(fig, f"SI_biofilm_consumption_profiles_{int(params['L'])}um")
    return fig


def plot_diffusion_flux(no_voltage: dict[str, np.ndarray], voltage_data: dict[float, dict[str, np.ndarray]], params: dict[str, Any]) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(no_voltage["x"], no_voltage["J"], "b-", label="No Voltage", zorder=4)

    linestyles = ["-", "--", ":"]
    for i, voltage in enumerate(sorted(voltage_data.keys())):
        profile = voltage_data[voltage]
        ax.plot(
            profile["x"],
            profile["J"],
            color=v_colors[i],
            linestyle=linestyles[i],
            label=f"{voltage} V Applied",
            zorder=3 - i,
        )

    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=params["L"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Biofilm Thickness x (µm)")
    ax.set_ylabel("Diffusion Flux (µM·µm/s)")
    ax.legend(fontsize=20, loc="best")
    ax.set_position((0.15, 0.15, 0.7, 0.8))
    save_fig(fig, f"SI_biofilm_diffusion_flux_profiles_{int(params['L'])}um")
    return fig


def plot_monod_correlation(
    no_voltage: dict[str, np.ndarray],
    voltage_data: dict[float, dict[str, np.ndarray]],
    monod_curve_c: np.ndarray,
    monod_curve_v: np.ndarray,
    params: dict[str, Any],
) -> tuple[Figure, Figure]:
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(monod_curve_c, monod_curve_v, "k-", label="Monod Curve", zorder=1)
    ax1.axvline(x=params["K_s"], color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(y=params["v_max"] / 2.0, color="gray", linestyle="--", alpha=0.5)

    n_samples = 15
    indices_no_v = np.linspace(0, len(no_voltage["C"]) - 1, n_samples).astype(int)
    colors_no_v = plt.get_cmap("Blues")(np.linspace(0.4, 0.9, n_samples))

    for i, idx in enumerate(indices_no_v):
        ax1.scatter(
            no_voltage["C"][idx],
            no_voltage["kinetic_rate"][idx],
            c=[colors_no_v[i]],
            s=50,
            edgecolors="blue",
            linewidth=0.5,
            zorder=2,
        )

    markers = ["s", "^", "D"]
    cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Oranges"), plt.get_cmap("RdPu")]

    for j, voltage in enumerate(sorted(voltage_data.keys())):
        profile = voltage_data[voltage]
        indices_v = np.linspace(0, len(profile["C"]) - 1, n_samples).astype(int)
        colors_v = cmaps[j](np.linspace(0.4, 0.9, n_samples))

        for i, idx in enumerate(indices_v):
            ax1.scatter(
                profile["C"][idx],
                profile["kinetic_rate"][idx],
                c=[colors_v[i]],
                s=50,
                marker=markers[j],
                edgecolors=v_colors[j],
                linewidth=0.5,
                zorder=2,
            )

    ax1.legend(fontsize=20, loc="best")
    ax1.set_xlabel("Substrate Concentration (µM)")
    ax1.set_ylabel("Kinetic Rate (µM/s)")
    ax1.set_position((0.15, 0.15, 0.7, 0.8))
    save_fig(fig1, f"SI_biofilm_monod_kinetics_correlation1_{int(params['L'])}um")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2_twin = ax2.twinx()

    lines = []
    lines.extend(ax2.plot(no_voltage["x"], no_voltage["C"], "b-", linewidth=2, label="Substrate (No V)"))
    lines.extend(
        ax2_twin.plot(
            no_voltage["x"],
            no_voltage["kinetic_rate"],
            "b--",
            alpha=0.7,
            label="Kinetic Rate (No V)",
        )
    )

    linestyles = ["-", "--", ":"]
    for i, voltage in enumerate(sorted(voltage_data.keys())):
        profile = voltage_data[voltage]
        lines.extend(
            ax2.plot(
                profile["x"],
                profile["C"],
                color=v_colors[i],
                linestyle=linestyles[i],
                label=f"Substrate ({voltage} V)",
            )
        )
        lines.extend(
            ax2_twin.plot(
                profile["x"],
                profile["kinetic_rate"],
                color=v_colors[i],
                linestyle=linestyles[i],
                alpha=0.7,
                label=f"Kinetic Rate ({voltage} V)",
            )
        )

    ax2.set_xlabel("Biofilm Thickness x (µm)")
    ax2.set_ylabel("Substrate Concentration (µM)")
    ax2_twin.set_ylabel("Monod Kinetic Rate (µM/s)")
    ax2.set_title("Spatial Correlation: Substrate ↔ Kinetics", fontsize=13, fontweight="bold")
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, fontsize=20, loc="upper right", ncol=2)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(x=params["L"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax2.set_position((0.15, 0.15, 0.7, 0.8))
    ax2_twin.set_position((0.15, 0.15, 0.7, 0.8))
    save_fig(fig2, f"SI_biofilm_monod_spatial_correlation2_{int(params['L'])}um")

    return fig1, fig2


def _plot_single_growth_metric(
    growth_axis_um: np.ndarray,
    metric_values: np.ndarray,
    terminal_thickness_um: float,
    y_label: str,
    title_metric: str,
    filename_metric_key: str,
    color: str,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(growth_axis_um, metric_values, color=color, linewidth=2.2)
    ax.set_xlim(0, terminal_thickness_um)
    ax.set_xlabel("Growth-state thickness coordinate (µm)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Growth trajectory up to L = {int(terminal_thickness_um)} µm ({title_metric})")
    save_fig(fig, f"SI_{filename_metric_key}_L{int(terminal_thickness_um)}um")
    return fig


def plot_growth_trajectory_figures(trajectories: dict[float, dict[str, np.ndarray]]) -> list[Figure]:
    all_figs: list[Figure] = []
    for terminal_thickness_um in sorted(trajectories.keys()):
        trajectory = trajectories[terminal_thickness_um]
        x = trajectory["growth_axis_um"]

        all_figs.append(
            _plot_single_growth_metric(
                x,
                trajectory["flux"],
                terminal_thickness_um,
                "Diffusion Flux J (µM·µm/s)",
                "J",
                "flux",
                "#1f77b4",
            )
        )
        all_figs.append(
            _plot_single_growth_metric(
                x,
                trajectory["consumption_rate"],
                terminal_thickness_um,
                "Consumption Rate R (µM·µm/s)",
                "R",
                "consumption_rate",
                "#d62728",
            )
        )
        all_figs.append(
            _plot_single_growth_metric(
                x,
                trajectory["biomass"],
                terminal_thickness_um,
                "Biomass X (relative·m)",
                "X",
                "biomass",
                "#2ca02c",
            )
        )
        all_figs.append(
            _plot_single_growth_metric(
                x,
                trajectory["substrate_concentration"],
                terminal_thickness_um,
                "Substrate Metric S = C̄ (µM)",
                "S",
                "substrate_concentration",
                "#9467bd",
            )
        )

    return all_figs


def _dictify_nested_arrays(raw: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            converted[key] = _dictify_nested_arrays(value)
        else:
            converted[key] = value
    return converted


def main() -> tuple[np.ndarray, np.ndarray, dict[float, dict[str, np.ndarray]], dict[str, Any], dict[float, dict[str, np.ndarray]], list[dict[str, float]]]:
    print("=" * 60)
    print("1D BIOFILM MODEL: CARBON DIFFUSION AND MONOD KINETICS")
    print("=" * 60)

    terminal_thickness_states_um = [10.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    run_model = cast(Callable[..., Any], getattr(biofilm_core, "run_model"))
    result = run_model(None, terminal_thickness_states_um, 100)

    params = dict(result["params"])
    no_voltage = _dictify_nested_arrays(dict(result["no_voltage"]))
    raw_voltage_data = dict(result["voltage_data"])
    voltage_data: dict[float, dict[str, np.ndarray]] = {
        float(voltage): _dictify_nested_arrays(dict(profile))
        for voltage, profile in raw_voltage_data.items()
    }
    raw_trajectories = dict(result["growth_trajectories"])
    growth_trajectories: dict[float, dict[str, np.ndarray]] = {
        float(terminal): _dictify_nested_arrays(dict(traj))
        for terminal, traj in raw_trajectories.items()
    }
    terminal_summary = [dict(row) for row in result["terminal_summary"]]

    print("\nModel Parameters:")
    print(f"  Biofilm thickness: {params['L']} µm")
    print(f"  Diffusion coefficient: {params['D']} µm²/s")
    print(f"  Monod v_max: {params['v_max']} µM/s")
    print(f"  Monod K_s: {params['K_s']} µM")
    print(f"  Wastewater carbon: {params['C_wastewater']} µM")
    print(f"  Electrode carbon (no V): {params['C_electrode_no_voltage']} µM")
    print(f"  Applied voltages: {params['voltages']} V")
    for voltage in params["voltages"]:
        print(f"    {voltage} V → {params['C_electrode_voltage'][voltage]} µM")

    print("\n" + "-" * 60)
    print("Rust compatibility check...")
    print("-" * 60)
    if result["compatibility_passed"]:
        print("✓ Backward-compatibility check passed at 500 µm (machine-precision tolerance)")
    else:
        raise RuntimeError("Backward compatibility check failed.")

    print("\n" + "-" * 60)
    print("Generating plots...")
    print("-" * 60)

    fig1 = plot_substrate_profiles(no_voltage, voltage_data, params)
    print("  ✓ Substrate concentration profiles")

    fig2 = plot_consumption_profiles(no_voltage, voltage_data, params)
    print("  ✓ Consumption rate profiles")

    fig3 = plot_diffusion_flux(no_voltage, voltage_data, params)
    print("  ✓ Diffusion flux profiles")

    fig4, fig5 = plot_monod_correlation(
        no_voltage,
        voltage_data,
        result["monod_curve_C"],
        result["monod_curve_v"],
        params,
    )
    print("  ✓ Monod kinetics correlation")

    save_terminal_state_summary(terminal_summary)
    save_growth_trajectory_csvs(growth_trajectories)
    trajectory_figs = plot_growth_trajectory_figures(growth_trajectories)
    print(f"  ✓ Growth-trajectory figures generated: {len(trajectory_figs)}")

    print("\n" + "-" * 60)
    print("Summary Statistics:")
    print("-" * 60)

    x_no_v = no_voltage["x"]
    c_no_v = no_voltage["C"]

    print("\nNo Voltage Applied:")
    print(f"  Substrate at x=0: {c_no_v[0]:.2f} µM")
    print(f"  Substrate at x=250: {c_no_v[len(c_no_v)//2]:.2f} µM")
    print(f"  Substrate at x=500: {c_no_v[-1]:.2f} µM")
    print(f"  Max consumption rate: {np.max(no_voltage['R']):.4f} µM/s")
    print(f"  Min consumption rate: {np.min(no_voltage['R']):.4f} µM/s")

    for voltage in sorted(voltage_data.keys()):
        profile = voltage_data[voltage]
        c_v = profile["C"]
        print(f"\n{voltage} V Applied:")
        print(f"  Substrate at x=0: {c_v[0]:.2f} µM")
        print(f"  Substrate at x=250: {c_v[len(c_v)//2]:.2f} µM")
        print(f"  Substrate at x=500: {c_v[-1]:.2f} µM")
        print(f"  Max consumption rate: {np.max(profile['R']):.4f} µM/s")
        print(f"  Min consumption rate: {np.min(profile['R']):.4f} µM/s")

    print("\n" + "=" * 60)
    print("Model execution complete!")
    print("=" * 60)

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()

    return x_no_v, c_no_v, voltage_data, params, growth_trajectories, terminal_summary


if __name__ == "__main__":
    main()
