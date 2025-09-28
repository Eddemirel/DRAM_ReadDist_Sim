#!/usr/bin/env python3
"""
config.py - Simple configuration interface for DRAM RowHammer Monte Carlo simulations

This script contains only the essential configuration parameters and a simple main 
function to launch the simulation. All helper functions have been moved to montecarlo.py.

Simply run: python config.py

Author: Eda Deniz Demirel
"""

import subprocess
import sys
import time

# -----------------------------------------------------------------------------
# Monte Carlo Configuration - Customize these parameters
# -----------------------------------------------------------------------------

# Simulation parameters
MONTE_CARLO_CONFIG = {
    "N":1,                    # Number of Monte Carlo samples
    "batch_size": 1,            # Parallel batch size
    "rows":5,                  # Number of DRAM rows
    "cols": 64,                 # Number of columns per row
    "subarray_size":32,         # Total subarray size
    "distance_to_SA": 15,        # Distance to sense amplifier
    "sampling_type": "random",  # 'random' or 'grid'
    "variation_mode": "victim_only", # 'full' or 'victim_only'
    
    # Hammer parameters
    #"tAggOn": [50, 200, 1000, 4000],  
    "tAggOn":36,             # Aggressor activation time (ns)
    #"tAggOff": [18*i for i in range(1, 6)],              # Precharge time (ns)
    "tAggOff": 18,              # Precharge time (ns)
    "hammer_count": 1000,        # Number of hammers
    "attack_type": "double",    # 'single' or 'double'
    
    # Data patterns
    "victim_data": "0xffffffffffffffff",       # Victim cell data pattern (hex)
    "aggressor_data": "0x00000000000000000000000000000",    # Aggressor cell data pattern (hex)
    
    # Environment
    "temperature": 300.0,       # Temperature in K
    "xyce_plugin": "/workspace/dram_bundle.so", # Path to Xyce plugin
    "seed": 52,               # Optional deterministic seed for reproducibility
}

# Parameter variation configuration
# Define which parameters vary and their statistical distributions
VARIATION_CONFIG = {
    # Storage elements (use multiplicative, positive lognormal variation)
    "Ron": {"family": "normal_pos", "mean": 1e4, "std": 1e3},
    "Roff": {"family": "normal_pos", "mean": 1e15, "std": 1e14},

    # Storage node parameters
    # Capacitances are small and positive; use lognormal with modest relative std
    "C_sn": {"family": "normal_pos", "mean": 8e-15, "std": 8e-16},
    "R_sn": {"family": "normal_pos", "mean": 1e1, "std": 1.0},

    # Coupling and parasitic capacitances
    "C_bl": {"family": "normal_pos", "mean": 0.3e-15, "std": 0.03e-15},
    "C_wl": {"family": "normal_pos", "mean": 0.3e-15, "std": 0.03e-15},
    "C_bl2bl": {"family": "normal_pos", "mean": 0.3e-15, "std": 0.03e-15},
    "C_wl2wl": {"family": "normal_pos", "mean": 0.3e-15, "std": 0.03e-15},
    # C_wl2bl was previously using a uniform with std equal to the mean (unstable).
    # Use a small relative std or uniform_pos if you want symmetric endpoints.
    "C_wl2bl": {"family": "normal_pos", "mean": 0.3e-15, "std": 0.03e-15},

    # Wire resistances
    "R_wl": {"family": "normal_pos", "mean": 100.0, "std": 10.0},
    "R_bl": {"family": "normal_pos", "mean": 100.0, "std": 10.0},

    # Current and voltage parameters
    "I0": {"family": "normal_pos", "mean": 1e-12, "std": 1e-16},
    # Threshold voltages are better modeled additively with positivity enforced
    "VTH0": {"family": "normal_pos", "mean": 0.6, "std": 0},

 
  "I_double_10":   { "family": "lognormal",   "mean": 1.0e-11,         "std": 0.0 },
  "tau_double_10": { "family": "lognormal",   "mean": 1.446976355e-9, "std": 6.254077851e-10 },
  "tau_single_10": { "family": "lognormal",   "mean": 1.505310870e-8, "std": 4.011507515e-11 },
  "k_double_10":   { "family": "normal_pos",  "mean": 6.583433619,    "std": 6.136673773e-7 },
  "k_mult_10":     { "family": "normal_pos",  "mean": 2.624716948,    "std": 3.772427659e-2 },
  "I_single_10":   { "family": "lognormal",   "mean": 8.633903362e-10, "std": 1.240884801e-11 },
  "I_const_10":    { "family": "lognormal",   "mean": 8.017451958e-11, "std": 9.525191575e-12 },

  "I_double_01":   { "family": "lognormal",   "mean": 1.0e-11,         "std": 0.0 },
  "tau_double_01": { "family": "lognormal",   "mean": 1.446976355e-9, "std": 6.254077851e-48 },
  "k_double_01":   { "family": "normal_pos",  "mean": 6.583433619,    "std": 6.136673773e-7 },
  "I_single_01":   { "family": "lognormal",   "mean": 8.633903362e-10, "std": 1.240884801e-11 },
  "I_const_01":    { "family": "lognormal",   "mean": 8.017451958e-11, "std": 9.525191575e-12 },
  "tau_single_01": { "family": "lognormal",   "mean": 1.505310870e-8, "std": 4.011507515e-11 },
  "k_mult_01":     { "family": "normal_pos",  "mean": 2.624716948,    "std": 3.772427659e-2 },

  "k_switch":      { "family": "normal_pos",  "mean": 2.333436172e1,  "std": 5.138572236e-1 }

}


def main():
    """Main function to run the Monte Carlo simulation with configured parameters."""
    print("======================================================")
    print("DRAM RowHammer Monte Carlo Simulation")
    print("======================================================")
    print(f"Starting simulation with {MONTE_CARLO_CONFIG['N']} samples...")
    print(f"Variation parameters: {list(VARIATION_CONFIG.keys())}")
    
    # Set start time
    start_time = time.time()
    
    # Save variation config to a file and pass it
    import json
    with open("variation_config.json", "w") as f:
        json.dump(VARIATION_CONFIG, f, indent=2)
    print(f"[CONFIG] Wrote variation_config.json with {len(VARIATION_CONFIG)} parameters")
    
    # Check if tAggOn and/or tAggOff are lists. If not, wrap them into lists.
    tAggOn_vals = MONTE_CARLO_CONFIG["tAggOn"] if isinstance(MONTE_CARLO_CONFIG["tAggOn"], list) else [MONTE_CARLO_CONFIG["tAggOn"]]
    tAggOff_vals = MONTE_CARLO_CONFIG["tAggOff"] if isinstance(MONTE_CARLO_CONFIG["tAggOff"], list) else [MONTE_CARLO_CONFIG["tAggOff"]]
    
    # Iterate over all combinations of tAggOn and tAggOff
    for tAggOn in tAggOn_vals:
        for tAggOff in tAggOff_vals:
            # Create a local config copy and update tAggOn and tAggOff
            local_config = dict(MONTE_CARLO_CONFIG)
            local_config["tAggOn"] = tAggOn
            local_config["tAggOff"] = tAggOff
            
            # Build command to run Monte Carlo script using the active Python interpreter
            cmd = [sys.executable, "montecarlo.py", "--analyze"]
            for key, value in local_config.items():
                # Skip None values to avoid passing empty args
                if value is None:
                    continue
                cmd.append(f"--{key}")
                cmd.append(str(value))
            cmd.extend(["--variation_config", "variation_config.json"])
            
            seed_val = local_config.get("seed")
            if seed_val is None:
                print(f"\nRunning with tAggOn={tAggOn} and tAggOff={tAggOff}: {' '.join(cmd)}")
                print("[CONFIG] seed not set — runs will not be deterministic by default")
            else:
                print(f"\nRunning with tAggOn={tAggOn} and tAggOff={tAggOff}: {' '.join(cmd)}")
                print(f"[CONFIG] seed={seed_val} — passing deterministic seed to montecarlo")
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f"\nSimulation failed with exit code {result.returncode}")
                sys.exit(1)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print("======================================================")

if __name__ == "__main__":
    main()