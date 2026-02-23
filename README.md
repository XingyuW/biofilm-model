# biofilm-model

A one-dimensional model describing the carbon diffusion, consumption, and kinetics across biofilm, implemented in Rust with Python bindings for high-performance numerical simulations.

## Python API Reference

The biofilm model is exposed to Python through the `biofilm_core` module. All functions return numpy arrays and Python dictionaries for easy integration with scientific Python workflows.

### Parameter Structure

Most functions accept an optional `params` dictionary to customize the model. If not provided, default parameters are used.

#### Default Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | float | 500.0 | Biofilm thickness (µm) |
| `n_points` | int | 501 | Number of spatial grid points |
| `D` | float | 600.0 | Diffusion coefficient (µm²/s) |
| `v_max` | float | 0.5 | Maximum kinetic rate (µM/s) |
| `K_s` | float | 50.0 | Monod half-saturation constant (µM) |
| `C_wastewater` | float | 200.0 | Substrate concentration at wastewater boundary (µM) |
| `C_electrode_no_voltage` | float | 10.0 | Substrate concentration at electrode (no voltage) (µM) |
| `rho_wastewater` | float | 0.2 | Bacterial density at wastewater boundary (relative) |
| `rho_middle` | float | 0.3 | Baseline bacterial density in middle (relative) |
| `rho_electrode_no_voltage` | float | 1.0 | Minimum bacterial density (relative) |
| `voltages` | list[float] | [1.9, 2.1, 2.3] | Applied voltages (V) |
| `C_electrode_voltage` | dict[float, float] | {1.9: 60.0, 2.1: 80.0, 2.3: 100.0} | Substrate concentration at electrode per voltage (µM) |
| `rho_electrode_voltage` | dict[float, float] | {1.9: 0.5, 2.1: 0.6, 2.3: 0.7} | Bacterial density boost at electrode per voltage (relative) |

### Core Functions

#### 1. `get_default_params()`

**Purpose**: Retrieve the default model parameters.

**Signature**:
```python
def get_default_params() -> dict
```

**Parameters**: None

**Returns**:
- `dict`: Dictionary containing all default parameters with keys matching the parameter table above

**Example**:
```python
from biofilm_core import get_default_params

params = get_default_params()
print(f"Biofilm thickness: {params['L']} µm")
print(f"Diffusion coefficient: {params['D']} µm²/s")
```

---

#### 2. `solve_steady_state_diffusion(params=None, voltage=None)`

**Purpose**: Solve the steady-state diffusion-reaction equation for substrate concentration distribution across the biofilm.

**Signature**:
```python
def solve_steady_state_diffusion(
    params: dict | None = None,
    voltage: float | None = None
) -> tuple[np.ndarray, np.ndarray]
```

**Parameters**:
- `params` (dict, optional): Custom parameters dictionary. If None, uses defaults.
- `voltage` (float, optional): Applied voltage. Must match one of the voltages in `params['voltages']`. If None, assumes no voltage condition.

**Returns**:
- `tuple[np.ndarray, np.ndarray]`:
  - `x`: Spatial positions (µm), shape (n_points,)
  - `C`: Substrate concentration (µM), shape (n_points,)

**Example**:
```python
from biofilm_core import solve_steady_state_diffusion
import numpy as np

# Solve without voltage
x, C = solve_steady_state_diffusion()
print(f"Concentration at wastewater: {C[0]:.1f} µM")
print(f"Concentration at electrode: {C[-1]:.1f} µM")

# Solve with 2.1V applied
x_v, C_v = solve_steady_state_diffusion(voltage=2.1)
```

---

#### 3. `calculate_diffusion_flux(x, c, params=None)`

**Purpose**: Calculate the diffusion flux J = -D(dC/dx) at all spatial positions.

**Signature**:
```python
def calculate_diffusion_flux(
    x: np.ndarray,
    c: np.ndarray,
    params: dict | None = None
) -> np.ndarray
```

**Parameters**:
- `x` (np.ndarray): Spatial positions (µm), shape (n,)
- `c` (np.ndarray): Substrate concentration (µM), shape (n,)
- `params` (dict, optional): Custom parameters. If None, uses defaults (specifically uses `params['D']`).

**Returns**:
- `np.ndarray`: Diffusion flux (µM·µm/s), shape (n,)

**Notes**: 
- Input arrays must have the same length
- First and last gradient estimates use forward/backward differences
- Interior points use central differences

**Example**:
```python
from biofilm_core import solve_steady_state_diffusion, calculate_diffusion_flux

x, C = solve_steady_state_diffusion()
J = calculate_diffusion_flux(x, C)
print(f"Flux at electrode (J[-1]): {J[-1]:.3f} µM·µm/s")
```

---

#### 4. `consumption_rate(c, x, params=None, voltage=None)`

**Purpose**: Calculate the consumption rate R(x) = v_monod(C) × ρ(x) at each position.

**Signature**:
```python
def consumption_rate(
    c: np.ndarray,
    x: np.ndarray,
    params: dict | None = None,
    voltage: float | None = None
) -> np.ndarray
```

**Parameters**:
- `c` (np.ndarray): Substrate concentration (µM), shape (n,)
- `x` (np.ndarray): Spatial positions (µm), shape (n,)
- `params` (dict, optional): Custom parameters.
- `voltage` (float, optional): Applied voltage for bacterial density profile.

**Returns**:
- `np.ndarray`: Consumption rate (µM/s), shape (n,)

**Notes**:
- Combines Monod kinetics: v = v_max × C / (K_s + C)
- With bacterial density profile: ρ(x) (voltage-dependent)
- Consumption rate = v × ρ

**Example**:
```python
from biofilm_core import solve_steady_state_diffusion, consumption_rate

x, C = solve_steady_state_diffusion()
R = consumption_rate(C, x)
total_consumption = np.trapz(R, x)
print(f"Total consumption rate: {total_consumption:.2f} µM/s")
```

---

#### 5. `compute_outputs(thickness_um, params=None, voltage=None)`

**Purpose**: Compute all steady-state outputs (concentration, kinetics, fluxes, and derived metrics) for a given biofilm thickness.

**Signature**:
```python
def compute_outputs(
    thickness_um: float,
    params: dict | None = None,
    voltage: float | None = None
) -> dict
```

**Parameters**:
- `thickness_um` (float): Biofilm thickness (µm). Must be > 0.
- `params` (dict, optional): Custom parameters.
- `voltage` (float, optional): Applied voltage.

**Returns**:
- `dict`: Dictionary with keys:
  - `thickness_um` (float): Input thickness
  - `thickness_m` (float): Thickness in meters
  - `x_um` (np.ndarray): Spatial grid (µm)
  - `x_m` (np.ndarray): Spatial grid (m)
  - `C` (np.ndarray): Substrate concentration (µM)
  - `kinetic_rate` (np.ndarray): Monod kinetics v(x) (µM/s)
  - `R` (np.ndarray): Consumption rate (µM/s)
  - `J` (np.ndarray): Diffusion flux (µM·µm/s)
  - `flux` (float): Flux at electrode boundary (µM·µm/s)
  - `consumption_rate` (float): Integrated consumption rate (µM·µm/s)
  - `biomass` (float): Integrated bacterial biomass (relative)
  - `substrate_metrics` (dict): With keys `C_left`, `C_mid`, `C_right`, `C_mean`, `C_min`, `C_max`

**Example**:
```python
from biofilm_core import compute_outputs

output = compute_outputs(thickness_um=100.0)
print(f"Flux: {output['flux']:.3f} µM·µm/s")
print(f"Mean concentration: {output['substrate_metrics']['C_mean']:.1f} µM")
print(f"Biomass: {output['biomass']:.2f}")
```

---

#### 6. `compute_all_growth_trajectories(terminal_thicknesses_um, params=None, voltage=None, n_points=None)`

**Purpose**: Compute growth trajectories (evolution of metrics from 0 to terminal thickness) for multiple terminal thicknesses.

**Signature**:
```python
def compute_all_growth_trajectories(
    terminal_thicknesses_um: list[float],
    params: dict | None = None,
    voltage: float | None = None,
    n_points: int | None = None
) -> dict[float, dict]
```

**Parameters**:
- `terminal_thicknesses_um` (list[float]): List of target biofilm thicknesses (µm).
- `params` (dict, optional): Custom parameters.
- `voltage` (float, optional): Applied voltage (constant during growth).
- `n_points` (int, optional): Number of trajectory points. Defaults to 100 (minimum 50).

**Returns**:
- `dict[float, dict]`: Dictionary keyed by terminal thickness. Each value is a trajectory dict with:
  - `growth_axis_um` (np.ndarray): Thicknesses sampled during growth (µm)
  - `eval_thickness_um` (np.ndarray): Actual evaluated thicknesses (µm)
  - `thickness_m` (np.ndarray): Thickness in meters
  - `flux` (np.ndarray): Flux evolution (µM·µm/s)
  - `consumption_rate` (np.ndarray): Consumption rate evolution (µM·µm/s)
  - `biomass` (np.ndarray): Biomass evolution (relative)
  - `substrate_concentration` (np.ndarray): Mean substrate concentration evolution (µM)
  - `substrate_metric_name` (str): Descriptor of the metric

**Example**:
```python
from biofilm_core import compute_all_growth_trajectories

trajectories = compute_all_growth_trajectories(
    terminal_thicknesses_um=[100.0, 200.0, 300.0],
    n_points=50
)

for thickness, traj in trajectories.items():
    print(f"Terminal thickness {thickness} µm:")
    print(f"  Final flux: {traj['flux'][-1]:.3f} µM·µm/s")
    print(f"  Final biomass: {traj['biomass'][-1]:.2f}")
```

---

#### 7. `run_model(params=None, terminal_thicknesses_um=None, n_points=None)`

**Purpose**: Run a comprehensive model simulation including steady-state profiles for all voltages and growth trajectories.

**Signature**:
```python
def run_model(
    params: dict | None = None,
    terminal_thicknesses_um: list[float] | None = None,
    n_points: int | None = None
) -> dict
```

**Parameters**:
- `params` (dict, optional): Custom parameters.
- `terminal_thicknesses_um` (list[float], optional): Thicknesses for growth trajectories. Defaults to [10, 100, 200, 300, 400, 500] µm.
- `n_points` (int, optional): Growth trajectory points. Defaults to 100.

**Returns**:
- `dict`: Comprehensive results dictionary with keys:
  - `params` (dict): Parameters used
  - `no_voltage` (dict): Steady-state profiles without voltage:
    - `x` (np.ndarray): Spatial grid
    - `C`, `kinetic_rate`, `R`, `J` (np.ndarray): Profiles
  - `voltage_data` (dict[float, dict]): Profiles at each voltage in `params['voltages']`
    - Each voltage maps to dict with `x`, `C`, `kinetic_rate`, `R`, `J`
  - `growth_trajectories` (dict[float, dict]): Growth trajectories for each terminal thickness
  - `terminal_summary` (list[dict]): Summary statistics at terminal thicknesses with:
    - `terminal_thickness_um`, `terminal_thickness_m`
    - `flux_uM_um_per_s`, `consumption_rate_uM_um_per_s`
    - `biomass_relative_m`, `substrate_Cbar_uM`
  - `compatibility_passed` (bool): Internal validation flag
  - `monod_curve_C`, `monod_curve_v` (np.ndarray): Monod kinetics curve

**Example**:
```python
from biofilm_core import run_model

results = run_model(
    terminal_thicknesses_um=[50.0, 100.0, 200.0]
)

# Access various results
print("Model parameters:", results['params']['L'], "µm")
print("No-voltage flux profile shape:", results['no_voltage']['C'].shape)
print("Terminal summary:")
for summary in results['terminal_summary']:
    print(f"  {summary['terminal_thickness_um']} µm: "
          f"flux = {summary['flux_uM_um_per_s']:.3f}")
```

---

## Usage Examples

### Example 1: Simple Steady-State Solve

```python
import numpy as np
from biofilm_core import solve_steady_state_diffusion

# Solve for default parameters
x, C = solve_steady_state_diffusion()

# Plot results
import matplotlib.pyplot as plt
plt.plot(x, C)
plt.xlabel("Position (µm)")
plt.ylabel("Concentration (µM)")
plt.show()
```

### Example 2: Custom Parameters

```python
from biofilm_core import get_default_params, compute_outputs

# Modify default parameters
params = get_default_params()
params['L'] = 300.0  # Thinner biofilm
params['v_max'] = 1.0  # Faster kinetics

# Compute outputs
out = compute_outputs(thickness_um=300.0, params=params)
print(f"Consumption rate: {out['consumption_rate']:.2f} µM·µm/s")
```

### Example 3: Growth Trajectory Analysis

```python
from biofilm_core import compute_all_growth_trajectories
import numpy as np

trajectories = compute_all_growth_trajectories(
    terminal_thicknesses_um=[100.0, 200.0],
    n_points=50
)

# Analyze how flux changes with thickness
for thickness, traj in trajectories.items():
    growth = traj['growth_axis_um']
    flux = traj['flux']
    print(f"\nThickness: {thickness} µm")
    print(f"  Initial flux: {flux[0]:.3f} µM·µm/s")
    print(f"  Final flux: {flux[-1]:.3f} µM·µm/s")
```

### Example 4: Voltage Effect Analysis

```python
from biofilm_core import solve_steady_state_diffusion

# Compare no voltage vs 2.1V
x1, C1 = solve_steady_state_diffusion(voltage=None)
x2, C2 = solve_steady_state_diffusion(voltage=2.1)

diff = C2 - C1
print(f"Max concentration increase: {np.max(diff):.1f} µM")
```

---

## Building and Installation

The module is built using Rust with PyO3 bindings. See `rust/biofilm/` for the source and build configuration.
