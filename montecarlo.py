"""
montecarlo.py
-------------
Monte Carlo RowHammer simulation front‑end for the DRAM circuit model provided by
`dram_array.Array`

⚠️  This module does not execute SPICE itself; it orchestrates all pre‑ and
post‑processing: waveform generation, netlist construction, batch execution via
GNU parallel + Xyce, and extraction of HCfirst metrics.

It can be used from the command line or imported as a library.  See the
`__main__` guard at the bottom for CLI usage.

Eda Deniz Demirel
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import csv
import datetime as _dt
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import warnings

# ---------------------------------------------------------------------------
# Attempt to import the user‑supplied Array class
# ---------------------------------------------------------------------------
try:
    from dram_array import Array  # preferred name given by the user
except ModuleNotFoundError:  # pragma: no cover – fallback for this template
    from array import Array  # type: ignore  # shadows stdlib 'array', OK here

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

# Global constants and configuration
import time
import random

# Generate a different seed each run based on current time
_RANDOM_SEED = int(time.time() * 1000) % 2**32
np.random.seed(_RANDOM_SEED)  # Seed NumPy's global RNG
random.seed(_RANDOM_SEED)     # Also seed Python's random module
_DEF_RNG = np.random.default_rng(_RANDOM_SEED)  # Create our default RNG

_PWL_EXT = ".pwl"                   # Extension for piecewise linear files
_PRN_EXT = ".cir.prn"               # Extension for Xyce output files


class MonteCarloError(RuntimeError):
    """Domain‑specific error for Monte Carlo workflow."""


# ------------------------ Data‑conversion utilities ------------------------

def convert_hex_to_bin(hex_string: str, width: int) -> str:
    """Convert hex strings (e.g. "0xF") to binary strings (e.g. "1111") of exact width.
    
    This is used to convert user-provided data patterns (in hex) to binary for cell initialization.
    The binary pattern is repeated across all cells in a row during simulation setup.
    
    Example:
        convert_hex_to_bin("0x3", 4) -> '0011'
    """
    if hex_string.lower().startswith("0x"):
        hex_string = hex_string[2:]
    value = int(hex_string, 16)
    if value >= 2 ** width:
        raise ValueError(
            f"Hex value 0x{value:X} exceeds width {width} bits (max {(2**width)-1})"
        )
    return format(value, f"0{width}b")


def calculate_pivot(distance_to_SA: int, rows: int, subarray_size: int) -> int:
    """Return the logical pivot row based on distance from sense amplifier.
    
    Calculates the pivot row according to hierarchical priority:
    1. Subarray size must be >= rows
    2. Distance to SA must be between 0 and subarray_size-2
    3. For distance_to_SA+2 > rows, we place pivot at rows//2
    
    The pivot row determines where aggressor/victim rows are positioned.
    """
    # Validate constraints
    if subarray_size < rows:
        raise MonteCarloError("Subarray size must be ≥ rows")
    if distance_to_SA < 0:
        raise MonteCarloError("distance_to_SA must be non-negative")
    if distance_to_SA >= subarray_size - 2:
        raise MonteCarloError(f"distance_to_SA must be < {subarray_size-2} (subarray_size-2)")
    
    # When distance_to_SA+2 > rows, use middle of array as pivot
    if distance_to_SA + 2 > rows:
        return rows // 2  # Use middle row as pivot when at the edge of array
    
    # Otherwise use distance directly (simple mapping with row 0 at sense-amp side)
    return distance_to_SA


# --------------------------- Sampling utilities ----------------------------

def _draw_value_normal(mean: float, var: float) -> float:
    """Draw a positive value from a normal distribution using rejection sampling.
    
    This function repeatedly samples from a normal distribution (with standard
    deviation sqrt(var)) until a value is obtained that is positive and not more than
    10σ away from the mean. If no valid sample is found after 100 tries, the mean is returned.
    """
    std = math.sqrt(var) if var >= 0 else abs(var)
    count = 0
    value = _DEF_RNG.normal(mean, std)
    while (value <= 0 or abs(value - mean) > 10 * std) and count < 100:
        value = _DEF_RNG.normal(mean, std)
        count += 1
    if value <= 0 or abs(value - mean) > 10 * std:
        value = mean
    return float(value)


def _draw_value_uniform(mean: float, var: float) -> float:
    """Draw a value from a uniform distribution centered at mean with half-width var.
    
    Used internally by draw_value() for 'uniform' distribution type.
    """
    half = abs(var)
    return float(_DEF_RNG.uniform(mean - half, mean + half))


def draw_value(mean: float, var: float, dist: str) -> float:
    """Draw a random value based on distribution type ('normal' or 'uniform').
    
    For normal distribution, var is treated as variance.
    For uniform distribution, var defines the half-range [mean-var, mean+var].
    
    This is used in parameter sampling to create Monte Carlo variations.
    """
    if dist == "normal":
        return _draw_value_normal(mean, var)
    if dist == "uniform":
        return _draw_value_uniform(mean, var)
    raise ValueError(f"Unsupported distribution '{dist}' (use 'normal'/'uniform')")

# ---------------------------------------------------------------------------
# Notebook-consistent sampling helpers (lognormal, normal_pos, uniform_pos)
# ---------------------------------------------------------------------------
EPS = 1e-30

def softplus(y: float) -> float:
    # stable softplus
    if y > 40.0:
        return y
    if y < -40.0:
        return math.exp(y)
    return math.log1p(math.exp(y))

def softplus_inv(x: float) -> float:
    # inverse of softplus; x>0 assumed
    if x > 40.0:
        return x
    return math.log(math.expm1(x))

def sigmoid(y: float) -> float:
    if y >= 0:
        t = math.exp(-y)
        return 1.0 / (1.0 + t)
    else:
        t = math.exp(y)
        return t / (1.0 + t)


def physical_to_latent(family: str, mean: float, std: float) -> tuple[float, float]:
    """Return (mu_lat, sigma_lat) for the given physical mean/std.

    All inputs are physical-space mean and std (std >= 0). Returns latent-space
    parameters used for sampling.
    """
    # Guard against tiny negative stds
    if std < 0:
        std = abs(std)

    # For numerical stability we keep a small sigma floor but we must still
    # compute the correct latent "mu" for the chosen family. Previously we
    # returned the physical mean as the latent mu when std < EPS which caused
    # values like softplus(≈0) -> ln(2) (≈0.693) to appear for tiny physical
    # means. Compute family-specific latent parameters here instead.
    if family == "lognormal":
        if mean <= 0:
            raise ValueError("lognormal family requires mean>0")
        v = std * std
        if std < EPS:
            # Zero (or negligible) spread: use a tiny sigma and set mu = log(mean)
            sigma = EPS
            mu = math.log(mean)
        else:
            sigma2 = math.log(1.0 + v / (mean * mean))
            sigma = math.sqrt(max(sigma2, EPS))
            mu = math.log(mean) - 0.5 * sigma2
        return mu, sigma

    elif family == "normal_pos":
        # Map physical mean -> latent mu via softplus inverse so that
        # softplus(mu_lat) ≈ mean when sigma_lat is small. Compute slope to map
        # physical std -> latent sigma correctly.
        mu_lat = softplus_inv(max(mean, EPS))
        slope = max(sigmoid(mu_lat), 1e-12)  # d softplus / d mu = sigmoid(mu)
        if std < EPS:
            sigma_lat = EPS
        else:
            sigma_lat = std / slope
        return mu_lat, max(sigma_lat, EPS)

    elif family == "uniform_pos":
        # For uniform_pos we keep latent center = physical mean and latent
        # std = physical std; the grid/sampling routines convert to half-range.
        if std < EPS:
            return mean, EPS
        return mean, std

    else:
        raise ValueError(f"Unknown family: {family}")


def sample_from_family(rng, family: str, mu_lat: float, sigma_lat: float) -> float:
    """Draw one sample X in PHYSICAL space from the specified family.

    rng may be the `random` module or a Random-like instance providing
    .normalvariate/gauss and .uniform.
    """
    if family == "lognormal":
        z = rng.gauss(0.0, 1.0)
        return math.exp(mu_lat + sigma_lat * z)
    elif family == "normal_pos":
        z = rng.gauss(0.0, 1.0)
        y = mu_lat + sigma_lat * z
        return max(EPS, softplus(y))
    elif family == "uniform_pos":
        half = math.sqrt(3.0) * sigma_lat
        y = rng.uniform(mu_lat - half, mu_lat + half)
        return max(EPS, y)
    else:
        raise ValueError(f"Unknown family: {family}")


def grid_points_for_family(family: str, mu_lat: float, sigma_lat: float) -> list[float]:
    """Return PHYSICAL-space deterministic grid points that mimic ±1σ variation.

    For lognormal and normal_pos we map latent ±1σ to physical via transform.
    For uniform_pos we use the latent endpoints (μ ± √3·σ) mapped directly.
    """
    if family in ("lognormal", "normal_pos"):
        pts_lat = [mu_lat - sigma_lat, mu_lat + sigma_lat]
        if family == "lognormal":
            return [math.exp(v) for v in pts_lat]
        else:  # normal_pos
            return [max(EPS, softplus(v)) for v in pts_lat]
    elif family == "uniform_pos":
        half = math.sqrt(3.0) * sigma_lat
        return [max(EPS, mu_lat - half), max(EPS, mu_lat + half)]
    else:
        raise ValueError(f"Unknown family: {family}")


_LEGACY_WARNED: set = set()

def _resolve_family_cfg(param_name: str, base_params: dict, variation_config: dict):
    """Resolve family, mean, std from variation_config with backward compat.

    Accepts new form {family, mean, std} or legacy {dist, mean, var} where
    'var' is treated as std for backward compatibility (with a one-time warning).
    """
    cfg = (variation_config or {}).get(param_name, {})
    family = cfg.get("family")
    if family is None:
        dist = cfg.get("dist", "normal")
        family = {"normal": "normal_pos", "uniform": "uniform_pos", "lognormal": "lognormal"}.get(dist, dist)

    mean = cfg.get("mean", base_params[param_name])
    if "std" in cfg:
        std = cfg["std"]
    elif "var" in cfg:
        # Legacy: treat 'var' as std (unify semantics). Warn once per param.
        if param_name not in _LEGACY_WARNED:
            warnings.warn(f"variation_config: 'var' for parameter {param_name} is deprecated; interpret as std")
            _LEGACY_WARNED.add(param_name)
        std = cfg["var"]
    else:
        std = 0.0
    return family, float(mean), float(std)


def generate_sampled_parameters(
    base_params: Dict[str, float],
    variation_config: Dict[str, Dict[str, Any]],
    N: int,
    sampling_type: str,
    *,
    seed: int | None = None,
) -> List[Dict[str, float]]:
    """Generate N sampled parameter dictionaries.

    variation_config entries are expected in physical space and may use the
    new form:

        {"family": "lognormal"|"normal_pos"|"uniform_pos", "mean": m, "std": s}

    Legacy keys ('dist', 'var') are accepted: 'var' is interpreted as std and a
    deprecation warning is emitted once per parameter.

    Sampling is performed in latent space (via physical_to_latent) and then
    mapped to physical space with the family-specific transform. No rejection
    sampling is used.
    """
    if sampling_type not in {"random", "grid"}:
        raise ValueError("sampling_type must be 'random' or 'grid'")

    param_names = list(base_params)

    # Precompute latent parameters for each parameter name
    latent = {}
    for p in param_names:
        family, mean, std = _resolve_family_cfg(p, base_params, variation_config)
        mu_lat, sigma_lat = physical_to_latent(family, mean, std)
        latent[p] = (family, mu_lat, sigma_lat)

    # Create deterministic RNG if seed provided, otherwise use global `random`
    rng = random.Random(seed) if seed is not None else random

    samples: List[Dict[str, float]] = []
    if sampling_type == "random":
        for _ in range(N):
            draw: Dict[str, float] = {}
            for p in param_names:
                fam, mu, sig = latent[p]
                draw[p] = sample_from_family(rng, fam, mu, sig)
            samples.append(draw)
        return samples

    # GRID sampling – Cartesian product of two-point axes per parameter
    axes: List[List[float]] = []
    for p in param_names:
        fam, mu, sig = latent[p]
        axes.append(grid_points_for_family(fam, mu, sig))

    grid = list(itertools.product(*axes))
    if not grid:
        grid = [tuple(base_params[p] for p in param_names)]

    for i in range(N):
        tup = grid[i % len(grid)]
        samples.append({p: tup[k] for k, p in enumerate(param_names)})
    return samples


# ------------------- Variation application & scaling ----------------------

def inject_variation(
    array: Array,
    variation_config: Dict[str, Dict[str, Any]],
    mode: str,
    victims: Sequence[int] | None = None,
    *,
    seed: int | None = None,
):
    """Apply Monte Carlo variations to victim cells with controlled parameter ranges.
    
    This function applies cell-level variations to the DRAM array parameters:
    
    - 'var' in JSON config is interpreted as standard deviation (σ)
    - For normal distribution, we sample until we get positive values within 10σ of mean
    - For uniform distribution, we use range [max(mean-√3·σ,0), mean+√3·σ]
    
    Parameters can be varied for all cells ('full' mode) or only victim cells ('victim_only'),
    which is more efficient for large arrays where we only care about victim behavior.
    
    The variation is applied directly to the array's parameter matrices, which are then
    used in netlist generation.
    """
    if mode not in {"full", "victim_only"}:
        raise ValueError("variation_mode must be 'full' or 'victim_only'")
    
    target_rows = victims if mode == "victim_only" else range(array.rows)
    if not target_rows:
        raise ValueError("victim rows must be provided and non-empty when mode='victim_only'")
    
    # For each parameter that has a variation entry, resolve family/mean/std and
    # draw per-cell samples using the same family samplers used by
    # generate_sampled_parameters(). This ensures identical semantics across
    # grid/random sampling and per-cell injection.
    # If a deterministic seed is provided, use generate_sampled_parameters to
    # obtain per-cell samples in the exact same draw order as the public
    # sampling function. This guarantees bit-for-bit reproducibility between
    # calls that use the same seed.
    # Build a list of (row, col) target cells in row-major order (or victims order).
    if mode == "victim_only":
        if victims is None:
            raise ValueError("victim rows must be provided when mode='victim_only'")
        target_cells = [(r, c) for r in victims for c in range(array.cols)]
    else:
        target_cells = [(r, c) for r in range(array.rows) for c in range(array.cols)]

    if seed is not None:
        # Generate N per-cell samples with the same API to ensure identical draws
        per_cell_samples = generate_sampled_parameters(
            array.dram_param_dict,
            variation_config,
            N=len(target_cells),
            sampling_type="random",
            seed=seed,
        )
        param_names = list(array.dram_param_dict)
        for idx, (r, c) in enumerate(target_cells):
            samp = per_cell_samples[idx]
            for p in param_names:
                array.dram_params[p][r][c] = samp[p]
        return

    # Otherwise use local RNG to draw parameter-by-parameter (non-deterministic order
    # matching previous behavior unless seed is given).
    rng = random

    for pname in array.dram_param_dict:
        # Determine family/mean/std (accepts legacy keys too)
        family, mean, std = _resolve_family_cfg(pname, array.dram_param_dict, variation_config)
        # Skip parameters with no variation
        if std <= 0:
            continue

        # Validate positive mean for physical parameters
        if mean is None or mean <= 0:
            raise MonteCarloError(f"Parameter {pname} must have a positive mean")

        mu_lat, sigma_lat = physical_to_latent(family, mean, std)

        for row in target_rows:
            for col in range(array.cols):
                value = sample_from_family(rng, family, mu_lat, sigma_lat)
                array.dram_params[pname][row][col] = value


def apply_distance_scaling(array: Array, M: float, mirror_M: float):
    """Scale parameters based on distance from sense amplifier.
    
    DRAM arrays have non-uniform characteristics depending on distance from sense amps.
    This function applies scaling factors M and mirror_M to the first and last rows
    to account for this distance effect.
    
    The scaling affects circuit parameters like resistances, capacitances, and currents,
    which impacts how easily cells are disturbed by hammering.
    """
    if abs(M - 1.0) < 1e-12 and abs(mirror_M - 1.0) < 1e-12:
        return  # Nothing to do if scaling factors are ~1.0
    
    # List of parameter types that should be scaled
    for pname in array.dram_params:
        if pname.startswith("C_") or pname.startswith("R_") or pname in {
            "I0",
            # new canonical names (10-variants)
            "I_double_10", "tau_double_10", "k_double_10",
            "I_single_10", "tau_single_10", "I_const_10",
            # 01-variants
            "I_double_01", "tau_double_01", "k_double_01",
            "I_single_01", "tau_single_01", "I_const_01",
            # other renamed/general keys
            "k_switch", "k_mult_10", "k_mult_01",
        }:
            # Scale first row (closest to sense amp)
            array.dram_params[pname][0, :] *= M
            # Scale last row (furthest from sense amp)
            array.dram_params[pname][-1, :] *= mirror_M            


# ------------------- Shared Array for common waveforms --------------------

def setup_shared_array(cfg: "Namespace", exp_dir: Path, pivot_row: int) -> Tuple[Array, float, float, List[int]]:
    """Initialize the DRAM array and generate common access patterns/waveforms.
    
    This creates the core Array object and configures:
    1. Precharge all cells
    2. Determine aggressor and victim rows based on attack type
    3. Initialize all rows with victim data
    4. Write aggressor data to aggressor rows
    5. Perform hammering (repeated access to aggressor rows)
    
    The shared_waveforms folder contains PWL files used by all netlists,
    which avoids duplicating waveforms for each Monte Carlo sample.
    
    Returns:
        - The configured Array object
        - Initial time (ns) when hammering begins
        - Hammer period (ns) for HCfirst calculation
        - List of victim row indices
    """
    if pivot_row <= 0 or pivot_row >= cfg.rows - 1:
        raise MonteCarloError(f"Pivot row {pivot_row} must be between 1 and {cfg.rows-2}")
    
    # Create Array object with proper DRAM library path
    shared = Array(
        base_folder=str(exp_dir),
        rows=cfg.rows,
        cols=cfg.cols,
        temperature=cfg.temperature,
        test="shared_waveforms",
        dram_lib_path="/workspace/dram.lib",
        dram_variation=None,  # Disable initial variation - we'll apply it ourselves
        step=50
    )

    # 1) Precharge all cells
    shared.precharge()

    # 2) Determine aggressor and victim rows based on attack type
    if cfg.attack_type == "single":
        # Single-sided: pivot is aggressor, victims are adjacent rows
        aggressor_rows = [pivot_row]
        victim_rows = [pivot_row - 1, pivot_row + 1]  # Two victims adjacent to aggressor
    elif cfg.attack_type == "double":
        # Double-sided: pivot is victim, aggressors sandwich it
        victim_rows = [pivot_row]  # Single victim in the middle
        aggressor_rows = [pivot_row - 1, pivot_row + 1]  # Two aggressors
    elif cfg.attack_type == "retention":
        # Retention test: do not hammer any rows; all rows are victim rows.
        aggressor_rows = []  # No aggressors
        victim_rows = list(range(cfg.rows))
    else:
        raise ValueError(f"Invalid attack type '{cfg.attack_type}' (use 'single', 'double', or 'retention')")

    # 3) Initialize all rows with victim data
    for r in range(cfg.rows):
        if r in aggressor_rows:
            shared.write(r, cfg.aggressor_data_bin)
        else:
            shared.write(r, cfg.victim_data_bin)
        shared.precharge()  # Precharge after each write
        shared.read(r)  # Read to stabilize the data
        shared.precharge()  # Precharge after read
    # Record initial time before hammering begins
    init_time_ns = shared.time_ns

    # Time for one complete hammer cycle (for HCfirst calculation)
    hammer_period_ns = cfg.tAggOn + cfg.tAggOff

    # 5) Hammer aggressor(s) or wait for retention period
    if cfg.attack_type == "retention":
        # For retention, do not hammer; wait for a specified duration.
        # You could use a configuration parameter, e.g., retention_duration_ns.
        shared.precharge(duration_ns=((cfg.tAggOn + cfg.tAggOff)* cfg.hammer_count))  # Precharge all rows
    else:
        # For hammer tests (single/double), proceed as before.
        for hc in range(cfg.hammer_count):
            for r in aggressor_rows:
                shared.read(r, duration_ns=cfg.tAggOn)  # Read aggressor
                shared.precharge(duration_ns=cfg.tAggOff)  # Precharge between accesses
        hammer_period_ns = cfg.tAggOn + cfg.tAggOff  # Time for one hammer cycl

    # Save all node voltage waveforms to files
    shared.save_nodes()
    shared.write_to_csv()
    
    # Return victim_rows as part of the tuple for later use
    return shared, init_time_ns, hammer_period_ns, victim_rows

# ------------------------ Monte Carlo analysis helpers -------------------

def analyze_monte_carlo_results(results_path: Path):
    # Load results
    df = pd.read_csv(results_path)
    
    # Replace infinity values with NaN for analysis
    df.replace([float('inf')], np.nan, inplace=True)
    
    print(f"Total cells: {df.shape[0]}")
    print(f"Cells with bit flips: {df['HCfirst'].notna().sum()}")
    print(f"Cells without bit flips: {df['HCfirst'].isna().sum()}")
    
    # Rest of your analysis code...


# ----------------------- Netlist helper per sample ------------------------

def construct_netlist_for_sample(
    cfg: "Namespace",
    exp_dir: Path,
    sample_idx: int,
    shared_arr: Array,  # This is our common Array object
    variation_rowset: Sequence[int],
    param_values: Dict[str, float],
    M: float,
    mirror_M: float,
    base_params: Dict[str, float],
    *,
    seed: int | None = None,
) -> Tuple[Path, Dict[str, float]]:
    """Build & save a netlist for one Monte Carlo sample with specific parameter values."""
    
    # IMPORTANT: Save the original parameter values to restore later
    original_params = {}
    for param in shared_arr.dram_param_dict:
        original_params[param] = shared_arr.dram_param_dict[param]
    
    # Save original parameter matrices (create deep copies)
    original_matrices = {}
    for param in shared_arr.dram_params:
        original_matrices[param] = shared_arr.dram_params[param].copy()
    
    # Start with base parameters
    for k, v in base_params.items():
        shared_arr.dram_param_dict[k] = v
        
    # Apply this sample's specific parameter values
    for k, v in param_values.items():
        shared_arr.dram_param_dict[k] = v
        
    # CRITICAL FIX: Update the parameter matrices with new dictionary values
    for k in shared_arr.dram_param_dict:
        if k in shared_arr.dram_params:
            # Update all cells in the matrix with the new value
            shared_arr.dram_params[k].fill(shared_arr.dram_param_dict[k])

    # Apply cell-level variations and scaling to the array
    inject_variation(shared_arr, cfg.variation_config, cfg.variation_mode, victims=variation_rowset, seed=seed)
    apply_distance_scaling(shared_arr, M, mirror_M)
    
    # Capture the final injected values from one representative victim cell
    mc_vars = [p for p, cfg_i in cfg.variation_config.items() if cfg_i.get("var", 0.0) > 0]
    # Do not fill the parameter matrices uniformly—keep the cell‐specific values set by inject_variation.
    # Instead, collect the unique per‑cell values into a nested dict.
    final_params = {p: {} for p in mc_vars}
    for r in variation_rowset:
        for c in range(cfg.cols):
            for p in mc_vars:
                final_params[p][(r, c)] = shared_arr.dram_params[p][r][c]
    # (Optionally, if the netlist generator uses dram_param_dict,
    # you might choose to update it for one representative cell or leave it unchanged.)

    # Create unique netlist path for this sample
    net_name = f"array_{cfg.rows}_{cfg.cols}_{sample_idx}.cir"
    net_path = exp_dir / net_name
    shared_arr.netlist_path = str(net_path)
    
    # Generate netlist with updated parameters
    shared_arr.netlist = []
    shared_arr.nodes_to_save = [f"SN_{r}_{c}" for r in variation_rowset for c in range(cfg.cols)]
    shared_arr.construct_netlist()
    
    # IMPORTANT: Restore original parameters to prevent accumulation
    for param, value in original_params.items():
        shared_arr.dram_param_dict[param] = value
    
    # Restore original parameter matrices
    for param, matrix in original_matrices.items():
        shared_arr.dram_params[param] = matrix.copy()
    
    if not net_path.exists():
        raise MonteCarloError(f"Failed to create netlist {net_path}")
    
    return net_path, final_params


# ----------------- Batch execution of SPICE via GNU parallel --------------

def run_simulations_in_batches(
    cir_paths: List[Path],
    batch_size: int,
    plugin: str | None = None,
):
    """Run SPICE simulations in parallel batches using GNU parallel + Xyce.
    
    To avoid overloading the system, simulations are executed in batches of
    'batch_size'. GNU parallel handles the distribution across available cores.
    
    This function spawns shell commands to run Xyce, monitors logs for errors,
    and continues even if some simulations fail.
    
    The plugin parameter specifies a Xyce plugin to load (e.g., dram_bundle.so).
    """
    if not cir_paths:
        return

    # Split into chunks of batch_size
    has_parallel = shutil.which("parallel") is not None
    xyce_bin = shutil.which("Xyce")
    for i in range(0, len(cir_paths), batch_size):
        chunk = cir_paths[i : i + batch_size]

        if has_parallel:
            # Build GNU parallel command
            shell_cmd = [
                "parallel",
                f"Xyce {{}} '>' {{}}.log",  # Quote the redirection operator
                "-plugin",
                plugin,
                ":::",
                *map(str, chunk)
            ]
            print("[MC] » Running:", " ".join(shell_cmd))
            try:
                # Execute the parallel simulation batch
                result = subprocess.run(" ".join(shell_cmd), shell=True, check=True)

                # Check log files for errors
                for cir in chunk:
                    log_file = Path(str(cir) + ".log")
                    if log_file.exists():
                        with open(log_file) as f:
                            log_content = f.read()
                            if "ERROR" in log_content or "FATAL" in log_content:
                                print(f"[ERROR] Found in {log_file}:")
                                print(log_content)

            except subprocess.CalledProcessError as exc:
                # Continue even if a batch fails
                print(f"[WARN] Batch starting at index {i} failed – continuing.")
                for cir in chunk:
                    log_file = Path(str(cir) + ".log")
                    if log_file.exists():
                        print(f"\nContents of {log_file}:")
                        with open(log_file) as f:
                            print(f.read())
        else:
            # Fallback: run Xyce sequentially for each cir if available
            if xyce_bin is None:
                print("[WARN] 'parallel' and 'Xyce' not found in PATH — skipping simulation runs.")
                continue

            for cir in chunk:
                cmd = [xyce_bin, str(cir)]
                if plugin:
                    cmd = [xyce_bin, "-plugin", plugin, str(cir)]
                print(f"[MC] » Running sequential: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError:
                    print(f"[WARN] Simulation failed for {cir} — continuing")
                # Check for log file
                log_file = Path(str(cir) + ".log")
                if log_file.exists():
                    with open(log_file) as f:
                        log_content = f.read()
                        if "ERROR" in log_content or "FATAL" in log_content:
                            print(f"[ERROR] Found in {log_file}:")
                            print(log_content)


# ------------------------------ Post‑process ------------------------------


def find_last_threshold_crossing(voltage_series: pd.Series, threshold: float, init_time_s: float, direction: str) -> float | None:
    """
    Find the last crossing time after init_time_s where the voltage series crosses the threshold.
    For a cell initialized to '1', look for a falling edge (voltage goes from ≥ threshold to below threshold).
    For a cell initialized to '0', look for a rising edge.
    Linear interpolation is used between adjacent points.
    
    Returns the crossing time in seconds or None if no crossing is found.
    """
    times = voltage_series.index.astype(float)
    # Consider only times after initialization
    mask = times >= init_time_s
    if not mask.any() or mask.sum() < 2:
        return None
    valid_times = times[mask]
    valid_values = voltage_series[mask].values

    crossing_times = []
    for i in range(len(valid_values) - 1):
        v1, v2 = valid_values[i], valid_values[i+1]
        t1, t2 = valid_times[i], valid_times[i+1]
        if direction == 'falling' and (v1 >= threshold and v2 < threshold):
            # Interpolate crossing time
            cross_t = t1 + (threshold - v1) * (t2 - t1) / (v2 - v1)
            crossing_times.append(cross_t)
        elif direction == 'rising' and (v1 < threshold and v2 >= threshold):
            cross_t = t1 + (threshold - v1) * (t2 - t1) / (v2 - v1)
            crossing_times.append(cross_t)
    if crossing_times:
        return crossing_times[-1]  # Return last crossing time
    else:
        return None

def process_victim_prn_file(prn_path: Path, init_time_ns: float, hammer_period_ns: float, victim_data: str, N: int) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(prn_path, delim_whitespace=True, comment="*", skipfooter=1, engine="python")
    except Exception as e:
        print(f"[ERROR] Cannot read {prn_path}: {e}")
        return []
    
    if "Index" in df.columns:
        df.drop(columns=["Index"], inplace=True)
    time_col = "TIME" if "TIME" in df.columns else df.columns[0]
    df = df.set_index(time_col)
    df.index = df.index.astype(float)
    
    init_time_s = init_time_ns * 1e-9
    hammer_period_s = hammer_period_ns * 1e-9

    results = []
    victim_columns = [col for col in df.columns if col.startswith("V(SN_")]
    # Sort victim_columns numerically by the column number
    victim_columns.sort(key=lambda col: int(col.split("_")[2].rstrip(")")))
    
    # Extract sample_id from the filename.
    try:
        name_str = prn_path.name  # e.g., "array_5_4_10.cir.prn" or "array_5_4_10.cir"
        if name_str.endswith(".cir.prn"):
            name_str = name_str[:-len(".cir.prn")]
        elif name_str.endswith(".cir"):
            name_str = name_str[:-len(".cir")]
        extracted_id = int(name_str.split("_")[-1])
        sample_id = extracted_id % N  # Wrap to expected range
    except Exception as e:
        print(f"[WARN] Could not parse sample_id from {prn_path.name}: {e}")
        sample_id = prn_path.name  # fallback as string

    for col in victim_columns:
        try:
            parts = col.strip("V()").split("_")
            victim_row = int(parts[1])
            victim_col = int(parts[2])
        except Exception:
            victim_row = np.nan
            victim_col = np.nan
        
        # Derive the expected state using the victim cell’s column number rather than loop index.
        expected_state = victim_data[victim_col % len(victim_data)]
        direction = 'falling' if expected_state == '1' else 'rising'
        
        voltage_series = df[col]
        cross_time = find_last_threshold_crossing(voltage_series, 0.615, init_time_s, direction)
        if cross_time is None:
            hcfirst = np.nan
        else:
            hc = (cross_time - init_time_s) / hammer_period_s
            hcfirst = np.nan if hc < 0 else hc
        
        results.append({
            "sample_id": sample_id,
            "victim_row": victim_row,
            "col": victim_col,
            "HCfirst": hcfirst
        })
    
    return results

def postprocess_all_results(exp_dir: Path, init_time_ns: float, hammer_period_ns: float, victim_data: str, N: int) -> pd.DataFrame:
    """
    Gather HCfirst results from all victim-only .cir.prn (or .cir) files in the experiment directory.
    
    Returns a DataFrame with one row per victim cell (with sample_id, victim_row, col, and HCfirst).
    """
    all_results = []
    prn_files = list(exp_dir.glob("array_*.cir.prn"))
    if not prn_files:
        # Fallback: try .cir files if .cir.prn files are not found
        prn_files = list(exp_dir.glob("array_*.cir"))
        if not prn_files:
            print("[WARN] No simulation output files found for postprocessing.")
            return pd.DataFrame()
    
    for prn_file in prn_files:
        results = process_victim_prn_file(prn_file, init_time_ns, hammer_period_ns, victim_data, N)
        all_results.extend(results)
    
    df_results = pd.DataFrame(all_results)
    return df_results


# -------------------------- CSV logging helpers ---------------------------

def save_variation_log(
    sample_id: int,
    victim_rows: Sequence[int],
    cols: int,
    cell_param_values: Dict[str, Dict[Tuple[int, int], float]],
    initial_data: str,
    variation_config: Dict[str, Dict[str, Any]],
    file_handle: csv.DictWriter
):
    """Save Monte Carlo variation data to CSV with per‑cell initial data.
    
    For each victim cell, records:
    - Sample ID, row, column
    - The cell’s initial data bit
    - Parameter values (unique per cell)
    - Placeholder for HCfirst (to be updated later)
    """
    # Get the names of parameters that are varied
    mc_vars = [param for param in variation_config.keys() 
               if variation_config[param].get("var", 0.0) > 0]
    
    for r in victim_rows:
        for c in range(cols):
            cell_data_bit = initial_data[c % len(initial_data)]
            row_entry = {
                "sample_id": sample_id,
                "row": r,
                "col": c,
                "initial_data": cell_data_bit,
                "HCfirst": np.nan
            }
            # For each varied parameter record its cell-specific value.
            for param in mc_vars:
                # Use the unique value for cell (r, c)
                row_entry[param] = cell_param_values[param].get((r, c))
            file_handle.writerow(row_entry)


def save_final_results_csv(df_params: pd.DataFrame, df_hc: pd.DataFrame, out_path: Path):
    """Merge parameter variations with HCfirst results and save to CSV.
    
    This function:
    1. Ensures HCfirst column exists
    2. Renames row→victim_row for consistent merging
    3. Merges the parameter table with HCfirst results
    4. Replaces negative HCfirst values with NaN (invalid values)
    5. Saves the merged table to CSV
    
    The final CSV contains all parameters and HCfirst values for each cell.
    """
    # Ensure 'HCfirst' column exists in params table
    if "HCfirst" not in df_params.columns:
        df_params["HCfirst"] = np.nan

    # Rename column for merging (row → victim_row)
    df_params = df_params.rename(columns={"row": "victim_row"})
    merge_cols = ["sample_id", "victim_row", "col"]
    
    # Verify required columns exist in both DataFrames
    for col in merge_cols:
        if col not in df_params.columns:
            raise KeyError(f"Column '{col}' missing from parameter variation table")
        if col not in df_hc.columns:
            raise KeyError(f"Column '{col}' missing from HCfirst results")
    
    # Merge on sample ID, victim row, and column
    merged = df_params.merge(df_hc, on=merge_cols, how="left")
    
    # Replace negative HCfirst values with NaN (invalid results)
    if "HCfirst" in merged.columns:
        merged["HCfirst"] = merged["HCfirst"].apply(lambda x: np.nan if (pd.notnull(x) and x < 0) else x)
    else:
        merged["HCfirst"] = np.nan

    # Save to CSV
    merged.to_csv(out_path, index=False)
    print(f"[MC] Final merged results → {out_path}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser with all simulation options."""
    p = argparse.ArgumentParser(description="Monte Carlo DRAM RowHammer simulator")
    p.add_argument("--N", type=int, required=True, help="Number of Monte Carlo samples")
    p.add_argument("--batch_size", type=int, default=4, help="GNU parallel batch size")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--subarray_size", type=int, default=512)
    p.add_argument("--distance_to_SA", type=int, default=0)
    p.add_argument("--sampling_type", choices=["random", "grid"], default="random")
    p.add_argument("--variation_mode", choices=["full", "victim_only"], default="full")
    p.add_argument("--variation_config", type=Path, required=True, help="JSON file")
    p.add_argument("--tAggOn", type=int, default=40)
    p.add_argument("--tAggOff", type=int, default=10)
    p.add_argument("--hammer_count", type=int, default=100000)
    p.add_argument("--attack_type", choices=["single", "double", "retention"], default="single")
    p.add_argument("--victim_data", default="0x0")
    p.add_argument("--aggressor_data", default="0xF")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--xyce_plugin", default="/workspace/dram_bundle.so")
    p.add_argument("--analyze", action="store_true", help="Analyze results after simulation")
    p.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    return p


def validate_variation_config(config: Dict[str, Dict[str, Any]], array: Array):
    """Validate the variation configuration against array defaults.
    
    Checks:
    1. All parameters exist in the Array
    2. All means are positive
    3. All variances are non-negative
    
    This catches invalid configuration before starting simulations.
    """
    for param, cfg in config.items():
        if param not in array.dram_param_dict:
            raise MonteCarloError(f"Unknown parameter {param} in variation config")
        if cfg.get("mean", 0) <= 0:
            raise MonteCarloError(f"Parameter {param} mean must be positive")
        if cfg.get("var", 0) < 0:
            raise MonteCarloError(f"Parameter {param} variance must be non-negative")


def main(cfg: argparse.Namespace | None = None):
    """Main execution workflow for Monte Carlo DRAM simulation.
    
    Overall process:
    1. Validate configuration
    2. Set up experiment directory
    3. Create shared array with common waveforms
    4. Generate parameter samples
    5. Create and save netlists for each sample
    6. Run simulations in batches
    7. Process results and calculate HCfirst values
    8. Save final CSV with parameters and results
    """
    if cfg is None:
        cfg = _build_arg_parser().parse_args()

    # ------------- Validation -------------
    if cfg.rows < 5:
        raise MonteCarloError("rows must be ≥ 5 by requirement")
    pivot = calculate_pivot(cfg.distance_to_SA, cfg.rows, cfg.subarray_size)

    # Convert hex patterns to binary strings of width = cols
    cfg.victim_data_bin = convert_hex_to_bin(cfg.victim_data, cfg.cols)
    cfg.aggressor_data_bin = convert_hex_to_bin(cfg.aggressor_data, cfg.cols)

    # Load variation config JSON
    with open(cfg.variation_config) as jf:
        cfg.variation_config = json.load(jf)

    # ------------- Experiment folder -------------
    # Create timestamped output directory
    now = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(
        f"MC_{cfg.attack_type}_{cfg.hammer_count}HC_{cfg.N}samples_{now}"
    ).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MC] Outputs in → {exp_dir}")
    print(f"[MC] Using random seed: {_RANDOM_SEED}")

    # Create shared array with common waveforms
    shared_arr, init_t, hammer_period, victim_rows = setup_shared_array(cfg, exp_dir, pivot)
    validate_variation_config(cfg.variation_config, shared_arr)

    # ------------- Parameter sampling -------------
    # Get base parameters from array and create N sampled parameter sets
    base_params = shared_arr.dram_param_dict.copy()
    sampled_param_sets = generate_sampled_parameters(
        base_params, cfg.variation_config, cfg.N, cfg.sampling_type
    )

    # Calculate distance-based scaling factors
    M = cfg.distance_to_SA - (cfg.rows - 1) + 1 if cfg.distance_to_SA > cfg.rows else 1
    mirror_M = cfg.subarray_size - M - 3
    if mirror_M <= 0:
        mirror_M = 1

    # ------------- CSV setup -------------
    # Create CSV file for parameter variations
    variation_tbl_path = exp_dir / "variation_table.csv" 
    var_file = variation_tbl_path.open("w", newline="")
    
    # Get parameters that have variation from config
    mc_vars = [param for param in cfg.variation_config.keys() 
               if cfg.variation_config[param].get("var", 0.0) > 0]
    
    # Set up CSV writer with all needed columns
    var_writer = csv.DictWriter(
        var_file,
        fieldnames=["sample_id", "row", "col", "initial_data", "HCfirst"] + mc_vars
    )
    var_writer.writeheader()

    # ------------- Netlist generation -------------
    # Create and save netlists for each sample
    cir_paths: List[Path] = []
    # If user provided a seed, use it to make parameter sampling deterministic
    sampled_param_sets = generate_sampled_parameters(
        base_params, cfg.variation_config, cfg.N, cfg.sampling_type, seed=cfg.seed
    )

    for idx, param_dict in enumerate(sampled_param_sets):
        # Generate netlist with specific parameter values
        per_sample_seed = None if cfg.seed is None else int(cfg.seed + idx)
        net_path, final_params = construct_netlist_for_sample(
            cfg,
            exp_dir,
            idx,
            shared_arr,
            victim_rows,
            param_dict,
            M,
            mirror_M,
            base_params,  # Pass base_params as an argument
            seed=per_sample_seed,
        )
        cir_paths.append(net_path)
        
        # Log parameter variations to CSV
        save_variation_log(idx, victim_rows, cfg.cols, final_params, 
                   cfg.victim_data_bin, cfg.variation_config, var_writer)
    
    var_file.close()

    # ------------- Run simulations -------------
    run_simulations_in_batches(cir_paths, cfg.batch_size, cfg.xyce_plugin)

    # ------------- Post-processing -------------
    print("[MC] Starting post-processing of simulation results...")
    df_hc = postprocess_all_results(exp_dir, init_t, hammer_period, cfg.victim_data_bin, cfg.N)

    # Load the previously saved variation table
    variation_tbl_path = exp_dir / "variation_table.csv"
    df_params = pd.read_csv(variation_tbl_path)

    # If no HC results were produced, skip merge and leave variation table as-is
    if df_hc.empty:
        print("[MC] No HCfirst results produced; skipping merge. Variation table left unchanged.")
        return

    # Rename columns to prepare for merging
    df_params = df_params.rename(columns={"row": "victim_row"})
    merge_cols = ["sample_id", "victim_row", "col"]

    for col in merge_cols:
        if col not in df_params.columns:
            raise KeyError(f"Column '{col}' missing from parameter variation table")
        if col not in df_hc.columns:
            raise KeyError(f"Column '{col}' missing from HCfirst results")

    # Merge the HCfirst results into the variation table (using a left-join so that every cell gets updated)
    merged = df_params.merge(df_hc, on=merge_cols, how="left", suffixes=("_old", ""))
    # If HCfirst from postprocessing is not null, update it; otherwise keep the original (NaN placeholder)
    merged["HCfirst"] = merged["HCfirst"].combine_first(merged["HCfirst_old"])
    merged.drop(columns=["HCfirst_old"], inplace=True)

    # Save the final merged table back to variation_table.csv
    merged.to_csv(variation_tbl_path, index=False)
    print(f"[MC] Final merged results saved to {variation_tbl_path}")

    # ------------- Final analysis -------------
  

# Entry point for command-line execution
if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except MonteCarloError as e:
        sys.exit(f"[MC‑ERROR] {e}")