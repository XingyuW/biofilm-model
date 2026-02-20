use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Clone, Debug)]
struct BiofilmParameters {
    l: f64,
    n_points: usize,
    d: f64,
    v_max: f64,
    k_s: f64,
    c_wastewater: f64,
    c_electrode_no_voltage: f64,
    voltages: Vec<f64>,
    c_electrode_voltage: Vec<(f64, f64)>,
    rho_wastewater: f64,
    rho_middle: f64,
    rho_electrode_no_voltage: f64,
    rho_electrode_voltage: Vec<(f64, f64)>,
}

impl Default for BiofilmParameters {
    fn default() -> Self {
        Self {
            l: 500.0,
            n_points: 501,
            d: 600.0,
            v_max: 0.5,
            k_s: 50.0,
            c_wastewater: 200.0,
            c_electrode_no_voltage: 10.0,
            voltages: vec![1.9, 2.1, 2.3],
            c_electrode_voltage: vec![(1.9, 60.0), (2.1, 80.0), (2.3, 100.0)],
            rho_wastewater: 0.2,
            rho_middle: 0.3,
            rho_electrode_no_voltage: 1.0,
            rho_electrode_voltage: vec![(1.9, 0.5), (2.1, 0.6), (2.3, 0.7)],
        }
    }
}

#[derive(Clone)]
struct OutputData {
    thickness_um: f64,
    thickness_m: f64,
    x_um: Vec<f64>,
    x_m: Vec<f64>,
    c: Vec<f64>,
    kinetic_rate: Vec<f64>,
    r: Vec<f64>,
    j: Vec<f64>,
    flux: f64,
    consumption_rate: f64,
    biomass: f64,
    c_left: f64,
    c_mid: f64,
    c_right: f64,
    c_mean: f64,
    c_min: f64,
    c_max: f64,
}

#[derive(Clone)]
struct TrajectoryData {
    growth_axis_um: Vec<f64>,
    eval_thickness_um: Vec<f64>,
    thickness_m: Vec<f64>,
    flux: Vec<f64>,
    consumption_rate: Vec<f64>,
    biomass: Vec<f64>,
    substrate_concentration: Vec<f64>,
}

fn thickness_um_to_m(thickness_um: f64) -> f64 {
    thickness_um * 1e-6
}

fn thickness_m_to_um(thickness_m: f64) -> f64 {
    thickness_m * 1e6
}

fn monod_scalar(c: f64, v_max: f64, k_s: f64) -> f64 {
    v_max * c / (k_s + c)
}

fn monod_vec(c: &[f64], v_max: f64, k_s: f64) -> Vec<f64> {
    c.iter().map(|&value| monod_scalar(value, v_max, k_s)).collect()
}

fn approx_voltage_lookup(entries: &[(f64, f64)], voltage: f64) -> Option<f64> {
    entries
        .iter()
        .find(|(v, _)| (*v - voltage).abs() < 1e-12)
        .map(|(_, value)| *value)
}

fn bacterial_density_profile(x: &[f64], params: &BiofilmParameters, voltage: Option<f64>) -> Vec<f64> {
    let mut rho = Vec::with_capacity(x.len());
    let rho_boost = voltage
        .and_then(|v| approx_voltage_lookup(&params.rho_electrode_voltage, v))
        .unwrap_or(0.0);

    for &x_i in x {
        let xi = x_i / params.l;
        let base_profile = params.rho_wastewater * (-3.0 * xi).exp()
            + params.rho_middle
            + 0.1 * (-10.0 * (xi - 1.0).powi(2)).exp();

        let mut value = base_profile;
        if voltage.is_some() {
            value += rho_boost * (-20.0 * (xi - 1.0).powi(2)).exp();
        }

        rho.push(value.max(params.rho_electrode_no_voltage));
    }

    rho
}

fn consumption_rate(c: &[f64], x: &[f64], params: &BiofilmParameters, voltage: Option<f64>) -> Vec<f64> {
    let kinetic_rate = monod_vec(c, params.v_max, params.k_s);
    let rho = bacterial_density_profile(x, params, voltage);
    kinetic_rate
        .iter()
        .zip(rho.iter())
        .map(|(kin, density)| kin * density)
        .collect()
}

fn calculate_diffusion_flux(x: &[f64], c: &[f64], params: &BiofilmParameters) -> PyResult<Vec<f64>> {
    if x.len() != c.len() {
        return Err(PyValueError::new_err("x and C must have the same length"));
    }
    if x.len() < 2 {
        return Err(PyValueError::new_err("x and C must contain at least 2 points"));
    }

    let n = x.len();
    let mut gradient = vec![0.0_f64; n];

    gradient[0] = (c[1] - c[0]) / (x[1] - x[0]);
    for i in 1..(n - 1) {
        gradient[i] = (c[i + 1] - c[i - 1]) / (x[i + 1] - x[i - 1]);
    }
    gradient[n - 1] = (c[n - 1] - c[n - 2]) / (x[n - 1] - x[n - 2]);

    Ok(gradient.into_iter().map(|dc_dx| -params.d * dc_dx).collect())
}

fn trapz(y: &[f64], x: &[f64]) -> PyResult<f64> {
    if y.len() != x.len() {
        return Err(PyValueError::new_err("x and y must have the same length for trapezoid integration"));
    }
    if y.len() < 2 {
        return Ok(0.0);
    }

    let mut sum = 0.0_f64;
    for i in 0..(y.len() - 1) {
        sum += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]);
    }
    Ok(sum)
}

fn linspace(start: f64, end: f64, n_points: usize) -> PyResult<Vec<f64>> {
    if n_points < 2 {
        return Err(PyValueError::new_err("n_points must be >= 2"));
    }
    let step = (end - start) / (n_points as f64 - 1.0);
    Ok((0..n_points).map(|i| start + i as f64 * step).collect())
}

fn solve_tridiagonal(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> PyResult<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if rhs.len() != n || lower.len() != n.saturating_sub(1) || upper.len() != n.saturating_sub(1) {
        return Err(PyValueError::new_err("Invalid tridiagonal system dimensions"));
    }

    let mut c_prime = vec![0.0_f64; n.saturating_sub(1)];
    let mut d_prime = vec![0.0_f64; n];

    if diag[0].abs() < 1e-18 {
        return Err(PyValueError::new_err("Singular tridiagonal system"));
    }

    if n > 1 {
        c_prime[0] = upper[0] / diag[0];
    }
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - lower[i - 1] * c_prime[i - 1];
        if denom.abs() < 1e-18 {
            return Err(PyValueError::new_err("Singular tridiagonal system during elimination"));
        }
        if i < n - 1 {
            c_prime[i] = upper[i] / denom;
        }
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut solution = vec![0.0_f64; n];
    solution[n - 1] = d_prime[n - 1];
    for i in (0..(n - 1)).rev() {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
    Ok(solution)
}

fn solve_steady_state_diffusion_internal(
    params: &BiofilmParameters,
    voltage: Option<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if params.n_points < 2 {
        return Err(PyValueError::new_err("params.n_points must be >= 2"));
    }
    if params.d <= 0.0 {
        return Err(PyValueError::new_err("params.D must be > 0"));
    }
    if params.k_s <= 0.0 {
        return Err(PyValueError::new_err("params.K_s must be > 0"));
    }

    let x = linspace(0.0, params.l, params.n_points)?;
    let dx = x[1] - x[0];
    let dx2 = dx * dx;
    let d_over_dx2 = params.d / dx2;

    let c_left = params.c_wastewater;
    let c_right = match voltage {
        Some(v) => approx_voltage_lookup(&params.c_electrode_voltage, v)
            .ok_or_else(|| PyValueError::new_err(format!("Voltage {v} not found in C_electrode_voltage")))?,
        None => params.c_electrode_no_voltage,
    };

    let mut c = linspace(c_left, c_right, params.n_points)?;
    if params.n_points == 2 {
        c[0] = c_left;
        c[1] = c_right;
        return Ok((x, c));
    }

    let rho_profile = bacterial_density_profile(&x, params, voltage);

    let tolerance = 1e-6_f64;
    let schedules: &[(f64, usize)] = &[(1.0, 5_000), (0.5, 10_000), (0.2, 20_000)];

    let n_int = params.n_points - 2;
    let lower = vec![-d_over_dx2; n_int.saturating_sub(1)];
    let diag = vec![2.0 * d_over_dx2; n_int];
    let upper = vec![-d_over_dx2; n_int.saturating_sub(1)];

    for (relaxation, max_iterations) in schedules {
        for _ in 0..*max_iterations {
            let c_old = c.clone();

            let mut rhs = vec![0.0_f64; n_int];
            for j in 0..n_int {
                let i = j + 1;
                let kinetic_rate = monod_scalar(c[i], params.v_max, params.k_s);
                rhs[j] = kinetic_rate * rho_profile[i];
            }

            rhs[0] += d_over_dx2 * c_left;
            rhs[n_int - 1] += d_over_dx2 * c_right;

            let c_interior = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;
            for j in 0..n_int {
                let i = j + 1;
                c[i] = *relaxation * c_interior[j] + (1.0 - *relaxation) * c[i];
            }

            c[0] = c_left;
            c[params.n_points - 1] = c_right;

            let mut max_error = 0.0_f64;
            for i in 0..params.n_points {
                let error = (c[i] - c_old[i]).abs();
                if error > max_error {
                    max_error = error;
                }
            }
            if max_error < tolerance {
                return Ok((x, c));
            }
        }
    }

    Err(PyValueError::new_err(
        "Steady-state solver did not converge within max_iterations",
    ))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn min(values: &[f64]) -> f64 {
    let mut current = f64::INFINITY;
    for &v in values {
        if v < current {
            current = v;
        }
    }
    current
}

fn max(values: &[f64]) -> f64 {
    let mut current = f64::NEG_INFINITY;
    for &v in values {
        if v > current {
            current = v;
        }
    }
    current
}

fn compute_outputs_internal(
    thickness_um: f64,
    params: &BiofilmParameters,
    voltage: Option<f64>,
) -> PyResult<OutputData> {
    if thickness_um <= 0.0 {
        return Err(PyValueError::new_err("thickness_um must be > 0"));
    }

    let thickness_m = thickness_um_to_m(thickness_um);
    let thickness_um_solver = thickness_m_to_um(thickness_m);

    let mut solver_params = params.clone();
    solver_params.l = thickness_um_solver;

    let (x_um, c) = solve_steady_state_diffusion_internal(&solver_params, voltage)?;
    let x_m: Vec<f64> = x_um.iter().map(|&x| thickness_um_to_m(x)).collect();
    let kinetic_rate = monod_vec(&c, solver_params.v_max, solver_params.k_s);
    let r = consumption_rate(&c, &x_um, &solver_params, voltage);
    let j = calculate_diffusion_flux(&x_um, &c, &solver_params)?;
    let rho = bacterial_density_profile(&x_um, &solver_params, voltage);

    let flux = j[j.len() - 1];
    let consumption_rate_scalar = trapz(&r, &x_um)?;
    let biomass = trapz(&rho, &x_m)?;

    let c_mid_index = c.len() / 2;
    Ok(OutputData {
        thickness_um,
        thickness_m,
        x_um,
        x_m,
        c_left: c[0],
        c_mid: c[c_mid_index],
        c_right: c[c.len() - 1],
        c_mean: mean(&c),
        c_min: min(&c),
        c_max: max(&c),
        c,
        kinetic_rate,
        r,
        j,
        flux,
        consumption_rate: consumption_rate_scalar,
        biomass,
    })
}

fn compute_growth_trajectory_internal(
    terminal_thickness_um: f64,
    params: &BiofilmParameters,
    voltage: Option<f64>,
    n_points: usize,
) -> PyResult<TrajectoryData> {
    if terminal_thickness_um <= 0.0 {
        return Err(PyValueError::new_err("terminal_thickness_um must be > 0"));
    }
    let n_points_safe = n_points.max(50);
    let growth_axis_um = linspace(0.0, terminal_thickness_um, n_points_safe)?;

    let min_eval_thickness_um = terminal_thickness_um.min(1.0);
    let mut eval_thickness_um = growth_axis_um.clone();
    eval_thickness_um[0] = min_eval_thickness_um;

    let mut thickness_m = Vec::with_capacity(n_points_safe);
    let mut flux = Vec::with_capacity(n_points_safe);
    let mut consumption_rate = Vec::with_capacity(n_points_safe);
    let mut biomass = Vec::with_capacity(n_points_safe);
    let mut substrate_concentration = Vec::with_capacity(n_points_safe);

    for &thickness in &eval_thickness_um {
        let out = compute_outputs_internal(thickness, params, voltage)?;
        thickness_m.push(out.thickness_m);
        flux.push(out.flux);
        consumption_rate.push(out.consumption_rate);
        biomass.push(out.biomass);
        substrate_concentration.push(out.c_mean);
    }

    Ok(TrajectoryData {
        growth_axis_um,
        eval_thickness_um,
        thickness_m,
        flux,
        consumption_rate,
        biomass,
        substrate_concentration,
    })
}

fn dict_get_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    match dict.get_item(key)? {
        Some(value) => Ok(Some(value.extract::<f64>()?)),
        None => Ok(None),
    }
}

fn dict_get_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    match dict.get_item(key)? {
        Some(value) => Ok(Some(value.extract::<usize>()?)),
        None => Ok(None),
    }
}

fn parse_params(params: Option<&Bound<'_, PyDict>>) -> PyResult<BiofilmParameters> {
    let mut parsed = BiofilmParameters::default();

    if let Some(config) = params {
        if let Some(value) = dict_get_f64(config, "L")? {
            parsed.l = value;
        }
        if let Some(value) = dict_get_usize(config, "n_points")? {
            parsed.n_points = value;
        }
        if let Some(value) = dict_get_f64(config, "D")? {
            parsed.d = value;
        }
        if let Some(value) = dict_get_f64(config, "v_max")? {
            parsed.v_max = value;
        }
        if let Some(value) = dict_get_f64(config, "K_s")? {
            parsed.k_s = value;
        }
        if let Some(value) = dict_get_f64(config, "C_wastewater")? {
            parsed.c_wastewater = value;
        }
        if let Some(value) = dict_get_f64(config, "C_electrode_no_voltage")? {
            parsed.c_electrode_no_voltage = value;
        }
        if let Some(value) = dict_get_f64(config, "rho_wastewater")? {
            parsed.rho_wastewater = value;
        }
        if let Some(value) = dict_get_f64(config, "rho_middle")? {
            parsed.rho_middle = value;
        }
        if let Some(value) = dict_get_f64(config, "rho_electrode_no_voltage")? {
            parsed.rho_electrode_no_voltage = value;
        }

        if let Some(value) = config.get_item("voltages")? {
            parsed.voltages = value.extract::<Vec<f64>>()?;
        }

        if let Some(value) = config.get_item("C_electrode_voltage")? {
            let voltage_map = value.downcast::<PyDict>()?;
            let mut items = Vec::with_capacity(voltage_map.len());
            for (k, v) in voltage_map.iter() {
                items.push((k.extract::<f64>()?, v.extract::<f64>()?));
            }
            parsed.c_electrode_voltage = items;
        }

        if let Some(value) = config.get_item("rho_electrode_voltage")? {
            let voltage_map = value.downcast::<PyDict>()?;
            let mut items = Vec::with_capacity(voltage_map.len());
            for (k, v) in voltage_map.iter() {
                items.push((k.extract::<f64>()?, v.extract::<f64>()?));
            }
            parsed.rho_electrode_voltage = items;
        }
    }

    Ok(parsed)
}

fn params_to_pydict(py: Python<'_>, params: &BiofilmParameters) -> PyResult<Py<PyDict>> {
    let result = PyDict::new_bound(py);
    result.set_item("L", params.l)?;
    result.set_item("n_points", params.n_points)?;
    result.set_item("D", params.d)?;
    result.set_item("v_max", params.v_max)?;
    result.set_item("K_s", params.k_s)?;
    result.set_item("C_wastewater", params.c_wastewater)?;
    result.set_item("C_electrode_no_voltage", params.c_electrode_no_voltage)?;
    result.set_item("voltages", params.voltages.clone())?;
    result.set_item("rho_wastewater", params.rho_wastewater)?;
    result.set_item("rho_middle", params.rho_middle)?;
    result.set_item("rho_electrode_no_voltage", params.rho_electrode_no_voltage)?;

    let c_voltage = PyDict::new_bound(py);
    for (k, v) in &params.c_electrode_voltage {
        c_voltage.set_item(*k, *v)?;
    }
    result.set_item("C_electrode_voltage", c_voltage)?;

    let rho_voltage = PyDict::new_bound(py);
    for (k, v) in &params.rho_electrode_voltage {
        rho_voltage.set_item(*k, *v)?;
    }
    result.set_item("rho_electrode_voltage", rho_voltage)?;

    Ok(result.unbind())
}

fn vec_to_pyarray(py: Python<'_>, values: Vec<f64>) -> Py<PyArray1<f64>> {
    PyArray1::from_vec_bound(py, values).unbind()
}

fn output_to_pydict(py: Python<'_>, out: &OutputData) -> PyResult<Py<PyDict>> {
    let result = PyDict::new_bound(py);
    result.set_item("thickness_um", out.thickness_um)?;
    result.set_item("thickness_m", out.thickness_m)?;
    result.set_item("x_um", vec_to_pyarray(py, out.x_um.clone()))?;
    result.set_item("x_m", vec_to_pyarray(py, out.x_m.clone()))?;
    result.set_item("C", vec_to_pyarray(py, out.c.clone()))?;
    result.set_item("kinetic_rate", vec_to_pyarray(py, out.kinetic_rate.clone()))?;
    result.set_item("R", vec_to_pyarray(py, out.r.clone()))?;
    result.set_item("J", vec_to_pyarray(py, out.j.clone()))?;
    result.set_item("flux", out.flux)?;
    result.set_item("consumption_rate", out.consumption_rate)?;
    result.set_item("biomass", out.biomass)?;
    result.set_item("substrate_metrics", {
        let metrics = PyDict::new_bound(py);
        metrics.set_item("C_left", out.c_left)?;
        metrics.set_item("C_mid", out.c_mid)?;
        metrics.set_item("C_right", out.c_right)?;
        metrics.set_item("C_mean", out.c_mean)?;
        metrics.set_item("C_min", out.c_min)?;
        metrics.set_item("C_max", out.c_max)?;
        metrics
    })?;
    Ok(result.unbind())
}

fn trajectory_to_pydict(py: Python<'_>, trajectory: &TrajectoryData) -> PyResult<Py<PyDict>> {
    let result = PyDict::new_bound(py);
    result.set_item("growth_axis_um", vec_to_pyarray(py, trajectory.growth_axis_um.clone()))?;
    result.set_item(
        "eval_thickness_um",
        vec_to_pyarray(py, trajectory.eval_thickness_um.clone()),
    )?;
    result.set_item("thickness_m", vec_to_pyarray(py, trajectory.thickness_m.clone()))?;
    result.set_item("flux", vec_to_pyarray(py, trajectory.flux.clone()))?;
    result.set_item(
        "consumption_rate",
        vec_to_pyarray(py, trajectory.consumption_rate.clone()),
    )?;
    result.set_item("biomass", vec_to_pyarray(py, trajectory.biomass.clone()))?;
    result.set_item(
        "substrate_concentration",
        vec_to_pyarray(py, trajectory.substrate_concentration.clone()),
    )?;
    result.set_item("substrate_metric_name", "thickness-averaged concentration (C_mean)")?;
    Ok(result.unbind())
}

#[pyfunction]
fn get_default_params(py: Python<'_>) -> PyResult<Py<PyDict>> {
    params_to_pydict(py, &BiofilmParameters::default())
}

#[pyfunction(signature = (params=None, voltage=None))]
fn solve_steady_state_diffusion(
    py: Python<'_>,
    params: Option<&Bound<'_, PyDict>>,
    voltage: Option<f64>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let parsed = parse_params(params)?;
    let (x, c) = solve_steady_state_diffusion_internal(&parsed, voltage)?;
    Ok((vec_to_pyarray(py, x), vec_to_pyarray(py, c)))
}

#[pyfunction(name = "consumption_rate", signature = (c, x, params=None, voltage=None))]
fn consumption_rate_py(
    py: Python<'_>,
    c: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray1<'_, f64>,
    params: Option<&Bound<'_, PyDict>>,
    voltage: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let parsed = parse_params(params)?;
    let c_vec = c.as_slice()?.to_vec();
    let x_vec = x.as_slice()?.to_vec();
    if c_vec.len() != x_vec.len() {
        return Err(PyValueError::new_err("c and x must have the same length"));
    }
    Ok(vec_to_pyarray(
        py,
        consumption_rate(&c_vec, &x_vec, &parsed, voltage),
    ))
}

#[pyfunction(name = "calculate_diffusion_flux", signature = (x, c, params=None))]
fn calculate_diffusion_flux_py(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    c: PyReadonlyArray1<'_, f64>,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let parsed = parse_params(params)?;
    let x_vec = x.as_slice()?.to_vec();
    let c_vec = c.as_slice()?.to_vec();
    let j = calculate_diffusion_flux(&x_vec, &c_vec, &parsed)?;
    Ok(vec_to_pyarray(py, j))
}

#[pyfunction(signature = (thickness_um, params=None, voltage=None))]
fn compute_outputs(
    py: Python<'_>,
    thickness_um: f64,
    params: Option<&Bound<'_, PyDict>>,
    voltage: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let parsed = parse_params(params)?;
    let out = compute_outputs_internal(thickness_um, &parsed, voltage)?;
    output_to_pydict(py, &out)
}

#[pyfunction(signature = (terminal_thicknesses_um, params=None, voltage=None, n_points=None))]
fn compute_all_growth_trajectories(
    py: Python<'_>,
    terminal_thicknesses_um: Vec<f64>,
    params: Option<&Bound<'_, PyDict>>,
    voltage: Option<f64>,
    n_points: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let parsed = parse_params(params)?;
    let points = n_points.unwrap_or(100);
    let trajectories = PyDict::new_bound(py);

    for &terminal in &terminal_thicknesses_um {
        let trajectory = compute_growth_trajectory_internal(terminal, &parsed, voltage, points)?;
        trajectories.set_item(terminal, trajectory_to_pydict(py, &trajectory)?)?;
    }

    Ok(trajectories.unbind())
}

#[pyfunction(signature = (params=None, terminal_thicknesses_um=None, n_points=None))]
fn run_model(
    py: Python<'_>,
    params: Option<&Bound<'_, PyDict>>,
    terminal_thicknesses_um: Option<Vec<f64>>,
    n_points: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let parsed = parse_params(params)?;
    let growth_states = terminal_thicknesses_um.unwrap_or_else(|| vec![10.0, 100.0, 200.0, 300.0, 400.0, 500.0]);
    let growth_points = n_points.unwrap_or(100);

    let result = PyDict::new_bound(py);
    result.set_item("params", params_to_pydict(py, &parsed)?)?;

    let (x_no_v, c_no_v) = solve_steady_state_diffusion_internal(&parsed, None)?;
    let no_v_kinetic = monod_vec(&c_no_v, parsed.v_max, parsed.k_s);
    let no_v_r = consumption_rate(&c_no_v, &x_no_v, &parsed, None);
    let no_v_j = calculate_diffusion_flux(&x_no_v, &c_no_v, &parsed)?;

    let no_voltage = PyDict::new_bound(py);
    no_voltage.set_item("x", vec_to_pyarray(py, x_no_v.clone()))?;
    no_voltage.set_item("C", vec_to_pyarray(py, c_no_v.clone()))?;
    no_voltage.set_item("kinetic_rate", vec_to_pyarray(py, no_v_kinetic.clone()))?;
    no_voltage.set_item("R", vec_to_pyarray(py, no_v_r.clone()))?;
    no_voltage.set_item("J", vec_to_pyarray(py, no_v_j.clone()))?;
    result.set_item("no_voltage", no_voltage)?;

    let voltage_data = PyDict::new_bound(py);
    for &voltage in &parsed.voltages {
        let (x_v, c_v) = solve_steady_state_diffusion_internal(&parsed, Some(voltage))?;
        let kinetic = monod_vec(&c_v, parsed.v_max, parsed.k_s);
        let r_v = consumption_rate(&c_v, &x_v, &parsed, Some(voltage));
        let j_v = calculate_diffusion_flux(&x_v, &c_v, &parsed)?;

        let profile = PyDict::new_bound(py);
        profile.set_item("x", vec_to_pyarray(py, x_v))?;
        profile.set_item("C", vec_to_pyarray(py, c_v))?;
        profile.set_item("kinetic_rate", vec_to_pyarray(py, kinetic))?;
        profile.set_item("R", vec_to_pyarray(py, r_v))?;
        profile.set_item("J", vec_to_pyarray(py, j_v))?;
        voltage_data.set_item(voltage, profile)?;
    }
    result.set_item("voltage_data", voltage_data)?;

    let trajectories = PyDict::new_bound(py);
    let terminal_rows = PyList::empty_bound(py);
    for &terminal in &growth_states {
        let trajectory = compute_growth_trajectory_internal(terminal, &parsed, None, growth_points)?;
        trajectories.set_item(terminal, trajectory_to_pydict(py, &trajectory)?)?;

        let terminal_idx = trajectory.growth_axis_um.len() - 1;
        let row = PyDict::new_bound(py);
        row.set_item("terminal_thickness_um", terminal)?;
        row.set_item("terminal_thickness_m", thickness_um_to_m(terminal))?;
        row.set_item("flux_uM_um_per_s", trajectory.flux[terminal_idx])?;
        row.set_item(
            "consumption_rate_uM_um_per_s",
            trajectory.consumption_rate[terminal_idx],
        )?;
        row.set_item("biomass_relative_m", trajectory.biomass[terminal_idx])?;
        row.set_item(
            "substrate_Cbar_uM",
            trajectory.substrate_concentration[terminal_idx],
        )?;
        terminal_rows.append(row)?;
    }
    result.set_item("growth_trajectories", trajectories)?;
    result.set_item("terminal_summary", terminal_rows)?;

    let legacy_output = compute_outputs_internal(500.0, &parsed, None)?;
    let mut compatibility_passed = true;
    if x_no_v.len() != legacy_output.x_um.len() || c_no_v.len() != legacy_output.c.len() {
        compatibility_passed = false;
    } else {
        for i in 0..x_no_v.len() {
            if (x_no_v[i] - legacy_output.x_um[i]).abs() > 1e-12
                || (c_no_v[i] - legacy_output.c[i]).abs() > 1e-12
                || (no_v_j[i] - legacy_output.j[i]).abs() > 1e-12
                || (no_v_r[i] - legacy_output.r[i]).abs() > 1e-12
            {
                compatibility_passed = false;
                break;
            }
        }
    }
    result.set_item("compatibility_passed", compatibility_passed)?;

    let c_range = linspace(0.0, parsed.c_wastewater * 1.2, 200)?;
    let v_monod = monod_vec(&c_range, parsed.v_max, parsed.k_s);
    result.set_item("monod_curve_C", vec_to_pyarray(py, c_range))?;
    result.set_item("monod_curve_v", vec_to_pyarray(py, v_monod))?;

    Ok(result.unbind())
}

#[pymodule]
fn biofilm_core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(get_default_params, module)?)?;
    module.add_function(wrap_pyfunction!(solve_steady_state_diffusion, module)?)?;
    module.add_function(wrap_pyfunction!(consumption_rate_py, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_diffusion_flux_py, module)?)?;
    module.add_function(wrap_pyfunction!(compute_outputs, module)?)?;
    module.add_function(wrap_pyfunction!(compute_all_growth_trajectories, module)?)?;
    module.add_function(wrap_pyfunction!(run_model, module)?)?;
    Ok(())
}