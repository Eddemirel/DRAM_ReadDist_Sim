"""
array.py
---------
An open-source Python module containing the `Array` class, which manages
DRAM circuit netlist generation and associated signal waveforms for Xyce
simulations.

Copyright (c) 2023 YOUR_NAME_OR_ORG
SPDX-License-Identifier: MIT  # or choose your license
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

class Array:
    """
    The Array class models a DRAM-like circuit structure and associated
    signals, generating netlists and PWL waveform files for Xyce-based
    simulations.

    Features:
    - Parameter dictionaries for DRAM & sense amplifiers
    - Automatic netlist construction & floating-node shunts
    - Time-based signal transitions for read/write/precharge
    - Saving waveforms as .pwl files
    """

    # ------------------- Class Constants and Defaults -----------------------
    # These "V_xxx" values define the default voltages used for signals
    V_HIGH_GENERIC = 1.2   # Voltage high for generic signals (e.g. SAE, WE, etc.)
    V_LOW_GENERIC  = 0.0   # Voltage low for generic signals
    V_HIGH_W       = 1.2   # Voltage high for W signals (write path)
    V_LOW_W        = 0  # Voltage low for W signals
    V_HIGH_WL      = 2.5
    V_LOW_WL       = 0.0
    V_REF          = 1.25
    V_PRECHARGE    = 0.6
    V_CELL         = 0.6

    # Simulation time defaults (in nanoseconds)        # Not widely used, but can define an internal step
    PRECHARGE_DURATION  = 18
    READ_WRITE_DURATION = 30
    TRANSITION_DT       = 1e-9      # 0.5 ns => slope time for signal transitions
    SIM_STEP            = 0          # For .TRAN 10n?

    # Xyce netlist options
    C_SHUNT = 1e-16   # Shunt capacitance for floating nodes
    R_SHUNT = 1e16
    RELTOL  = 1e-4
    ABSTOL  = 1e-15
    GMIN = 1e-4
    initial_cond = True
    parallel_processing = False
    np = 1            # Number of parallel domains if using .OPTIONS PARALLEL

    # Form a string with the .OPTIONS line
    OPT_COMMAND = f".OPTIONS TIMEINT ABSTOL={ABSTOL} RELTOL={RELTOL} \n.OPTIONS DEVICE GMIN={GMIN} \n.OPTIONS LINSOL TYPE=KLU \n"

    def __init__(self, base_folder, rows=4, cols=4, temperature=300, test="default",
                 dram_param_dict=None, dram_variation=None,
                 senseamp_param_dict=None, senseamp_variation=None,
                 dram_lib_path="", step=50, commands=""):
        """
        Initialize an Array object.

        Parameters
        ----------
        base_folder : str
            Directory where netlist and waveform files are created.
        rows : int
            Number of DRAM rows.
        cols : int
            Number of DRAM columns.
        temperature : float
            Operating temperature in Kelvin.
        test : str
            Name of the test scenario (used as sub-folder).
        dram_param_dict : dict or None
            Base DRAM parameters. If None, defaults are used.
        dram_variation : dict or None
            Variation (in percent) for DRAM parameters. If None, defaults are zero.
        senseamp_param_dict : dict or None
            Base sense amplifier parameters. If None, defaults are used.
        senseamp_variation : dict or None
            Variation (in percent) for sense amp parameters. If None, defaults are zero.
        dram_lib_path : str
            Path to the DRAM library file (e.g., 'dram.lib').
        step : int
            Simulation step in ns (for controlling times).
        commands : str
            Additional netlist commands appended at the end of the netlist.

        Notes
        -----
        - Creates subdirectories as needed.
        - Manages signal transitions in self.signal_transitions,
          storing times and voltages for each net-driven signal.
        """
        self.base_folder = base_folder
        self.test = test
        self.rows = rows
        self.cols = cols
        self.node_count = 0
        self.netlist = []

        # We'll store transitions in a dict of lists, and track current values
        self.signal_transitions = {}  # {signal_name: [(time_s, voltage), ...]}
        self.current_values = {}      # {signal_name: last_voltage_set}

        # We'll track "time" in ns. Convert to s when writing transitions
        self.time_ns = 0
        self.sim_time = 0
        self.step = step
        self.nodes_to_save = []

        # Global scenario variables
        self.temperature = temperature
        self.vdd = self.V_HIGH_GENERIC
        self.vpre = self.V_PRECHARGE
        self.vss = self.V_LOW_GENERIC

        # Parameter dictionaries
        self.dram_param_dict = dram_param_dict if dram_param_dict is not None else self._default_dram_params()
        self.dram_variation  = dram_variation  if dram_variation  is not None else self._default_variation()
        self.senseamp_param_dict = senseamp_param_dict if senseamp_param_dict is not None else self._default_senseamp_params()
        self.senseamp_variation  = senseamp_variation  if senseamp_variation  is not None else self._default_variation()

        # Expand row×col param arrays for DRAM and senseamp
        self.dram_params     = self._generate_dram_params()
        self.senseamp_params = self._generate_senseamp_params()

        # DRAM library path, if referencing .lib
        self.dram_lib_path = dram_lib_path

        # For netlist final lines
        self.simulation_commands = commands
        self.netlist_path = ""

        # Ensure base folder exists
        os.makedirs(self.base_folder, exist_ok=True)

        # Prepare signals
        self.initialize_signals()

    # -------------------------------------------------------------------------
    #  Default parameter sets and variations
    # -------------------------------------------------------------------------
    def _default_dram_params(self):
        """Return a dictionary of default DRAM parameters."""
        return {
    "Ron":     1e4,    "Roff":    1e14,
    "C_sn":    8e-15,    "R_sn":    10,
    "C_bl":    0.3e-15,    "C_wl":    0.3e-15,    "C_bl2bl": 0.3e-15,    "C_wl2wl": 0.3e-15,    "C_wl2bl": 0.3e-15,
    "R_wl":    100,    "R_bl":    100,    
    "I0":      1e-15,    "VTH0":    0.7,
    "I_double_10":   20e-9,     "tau_double_10":     20e-9,    "k_double_10":100,
    "I_single_10":  1e-15,    "tau_single_10":1,
    "I_const_10":  2e-9,    "k_mult_10":   2,
    "k_switch":     20, 
    "I_single_01":  10e-12,    "tau_single_01":   100e-9,
    "I_const_01":   2e-12,    "k_mult_01":    2.0,
    # Newly added parameters (defaulted to same magnitude as 10-variants;
    # please adjust if you want different defaults)
    "I_double_01": 20e-9, "tau_double_01": 20e-9, "k_double_01": 100,
}

    def _default_senseamp_params(self):
        """Return a dictionary of default sense amplifier parameters."""
        return {
            "Ron_SA": 1e4, "Roff_SA": 1e15, "Vt_SA": 0.1, "C_SA": 5e-15, "slope":40
        }

    def _default_variation(self):
        """
        Return a dictionary of zero variations (no randomization).
        Variation is in percent, e.g. 5 => ±5% range.
        """
        return {key: 0.0 for key in self._default_dram_params()}

    # -------------------------------------------------------------------------
    #  Random parameter generation
    # -------------------------------------------------------------------------
    def _generate_param_value(self, base_value, variation_percent):
        """
        Generates one random parameter value given:
          - base_value: float
          - variation_percent: e.g. 5 => ±5%
        """
        variation_abs = base_value * (variation_percent / 100.0)
        low  = base_value - variation_abs
        high = base_value + variation_abs
        return np.random.uniform(low, high)

    def _generate_dram_params(self):
        """
        Create a rows×cols array of DRAM parameters, each with random variation.
        Returns a dict: {param_name: ndarray(rows, cols)}
        """
        params = {}
        for key, base_value in self.dram_param_dict.items():
            variation = self.dram_variation.get(key, 0.0)
            params[key] = np.array([
                [self._generate_param_value(base_value, variation) for _ in range(self.cols)]
                for _ in range(self.rows)
            ])
        return params

    def _generate_senseamp_params(self):
        """
        Create a rows×cols array of senseamp parameters, each with random variation.
        Some senseamps might be 1-per-column (or 1 total), but we generalize to rows×cols.
        """
        params = {}
        for key, base_value in self.senseamp_param_dict.items():
            variation = self.senseamp_variation.get(key, 0.0)
            params[key] = np.array([
                [self._generate_param_value(base_value, variation) for _ in range(self.cols)]
                for _ in range(self.rows)
            ])
        return params

    # -------------------------------------------------------------------------
    #  Netlist Construction
    # -------------------------------------------------------------------------
    def construct_dram_netlist(self):
        """
        Construct netlist lines for the DRAM cells. Each cell is an instance of
        'dram_cell' subcircuit with row×col parameter variations.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                cell_name = f"XC{i}_{j}"

                # Potential custom logic for node naming
                x = (i + 1) if (i + j) % 2 == 0 else (i - 1)
                y = (i + 1) if (i + j) % 2 == 1 else (i - 1)

                if x < 0 or x >= self.rows:
                    node_x = "0"
                else:
                    node_x = f"WL_in{j}_{x}"

                if y < 0 or y >= self.rows:
                    node_y = "0"
                else:
                    node_y = f"WL_in{j}_{y}"

                # The node list for this cell
                nodes = (
                    f"BL_in{i}_{j}",  # 0  (bitline row i, col j)
                    "Vcell",         # 1
                    f"WL_in{j}_{i}", # 2  (wordline col j, row i)
                    f"BB_{i}_{j+1}", # 3
                    f"BB_{i}_{j}",   # 4
                    f"BL_in{i+1}_{j}", # 5
                    f"SN_{i}_{j}",   # 6
                    f"WL_in{j+1}_{i}",  # 7
                    f"WW_{j}_{i}",      # 8
                    f"WW_{j}_{i+1}",    # 9
                    node_x,            # 10
                    node_y,            # 11
                )

                # Build parameter string from self.dram_params
                params = {
                    "Ron":   self.dram_params["Ron"][i][j],
                    "Roff":  self.dram_params["Roff"][i][j],
                    "C_sn":  self.dram_params["C_sn"][i][j],
                    "R_sn":  self.dram_params["R_sn"][i][j],
                    "C_bl":  self.dram_params["C_bl"][i][j],
                    "C_wl":  self.dram_params["C_wl"][i][j],
                    "C_bl2bl": self.dram_params["C_bl2bl"][i][j],
                    "C_wl2wl": self.dram_params["C_wl2wl"][i][j],
                    "C_wl2bl": self.dram_params["C_wl2bl"][i][j],
                    "R_wl":  self.dram_params["R_wl"][i][j],
                    "R_bl":  self.dram_params["R_bl"][i][j],
                    # renamed/converted parameters (canonical names only)
                    "tau_double_10":   self.dram_params["tau_double_10"][i][j],
                    "k_mult_10": self.dram_params["k_mult_10"][i][j],
                    "I0":    self.dram_params["I0"][i][j],
                    "VTH0":  self.dram_params["VTH0"][i][j],
                    "I_double_10": self.dram_params["I_double_10"][i][j],
                    "I_single_10": self.dram_params["I_single_10"][i][j],
                    "I_const_10": self.dram_params["I_const_10"][i][j],
                    "tau_single_10": self.dram_params["tau_single_10"][i][j],
                    "k_double_10": self.dram_params["k_double_10"][i][j],
                    "k_switch":   self.dram_params["k_switch"][i][j],
                    "I_single_01": self.dram_params["I_single_01"][i][j],
                    "I_const_01": self.dram_params["I_const_01"][i][j],
                    "tau_single_01": self.dram_params["tau_single_01"][i][j],
                    "k_mult_01": self.dram_params["k_mult_01"][i][j],
                    # newly added mirrored-01 variants
                    "I_double_01": self.dram_params["I_double_01"][i][j],
                    "tau_double_01": self.dram_params["tau_double_01"][i][j],
                    "k_double_01": self.dram_params["k_double_01"][i][j],
                }
                param_str = " ".join(f"{k}={v}" for k,v in params.items())

                # One netlist line per cell
                self.netlist.append(
                    f"{cell_name} {' '.join(nodes)} dram_cell params: {param_str}"
                )

    def construct_senseamp_netlist(self):
        """
        Construct netlist lines for sense amplifiers. 
        In this example, we create one sense amp per column, referencing 'senseamp'.
        """
        for j in range(self.cols):
            sa_name = f"YSENSEAMP SA_{j}"
            nodes = (
                "Eq", "Vss", "SAE", "SAEd", "Vdd", "Vpre",
                f"W_{j}", "WE", f"WI_{j}", f"BL_in0_{j}", f"BL_inL_{j}"
            )
            # Just picking row=0 for param lookup:
            params = {
                "Ron":  self.senseamp_params["Ron_SA"][0][j],
                "Roff": self.senseamp_params["Roff_SA"][0][j],
                "Vt_SA":   self.senseamp_params["Vt_SA"][0][j],
                "C_SA": self.senseamp_params["C_SA"][0][j],
                "slope": self.senseamp_params["slope"][0][j],
            }
            param_str = " ".join(f"{k}={v}" for k,v in params.items())
            self.netlist.append(
                f"{sa_name} {' '.join(nodes)} {param_str}"
            )

    def construct_voltage_sources(self):
        """
        Construct DC voltage sources and PWL sources for row/column signals.

        Example for DC sources:
          Vdd  Vdd   0   DC 1.2
          VREF VREF  0   DC 1.25
          ...
        """
        # DC sources
        self.netlist.append(f"Vdd Vdd 0 DC {self.vdd}")
        self.netlist.append(f"Vss Vss 0 DC {self.vss}")
        self.netlist.append(f"Vpre Vpre 0 DC {self.vpre}")
        self.netlist.append(f"Vcell Vcell 0 DC {self.V_CELL}")

        # PWL sources for signals
        # We'll produce them from self.signal_transitions, so we just name them here
        pwl_sources = (
            [f"WL_in0_{i}" for i in range(self.rows)] +
            [f"W_{j}" for j in range(self.cols)] +
            [f"WI_{j}" for j in range(self.cols)] +
            ["Eq", "SAE", "SAEd", "WE"]
        )

        # For each PWL source, reference a .pwl file in base_folder/test
        for source_name in pwl_sources:
            csv_path = os.path.join(self.base_folder, self.test, f"{source_name}.pwl")
            # Must use PWL FILE= syntax for Xyce:
            self.netlist.append(
                f"V{source_name} {source_name} 0 PWL FILE \"{csv_path}\""
            )

    def analyze_and_shunt_floating_nodes(self, exclude_nodes=None, shunt_value="{R_shunt}"):
        """
        Scan netlist for floating nodes (referenced once or not at all),
        then append a capacitor to ground for each 'floating' node to ensure
        stable DC operating points.

        Parameters
        ----------
        exclude_nodes : set or None
            Node names to exclude from being shunted (e.g., '0', 'Vdd', etc.)
        shunt_value : str
            The value for the shunt capacitor. Could be '{C_shunt}' or '1e-17'
        """
        if exclude_nodes is None:
            exclude_nodes = {"0", "Vss", "Vdd", "Vcell", "Vpre", "VREF"}

        node_usage = defaultdict(int)

        # Parse netlist lines for node references
        for line in self.netlist:
            # skip lines that are comments or directives
            if line.startswith("*") or line.startswith(".") or line.startswith("+"):
                continue
            tokens = line.split()

            # skip device name
            if tokens[0][0] == "Y":
                tokens = tokens[2:]
            else:
                tokens = tokens[1:]
            for token in tokens:
                if "=" in token:
                    # skip parameter assignments
                    break
                elif token in ("dram_cell", "senseamp", "params:", "PWL", "FILE=", "1.2", "0", "0.0", "2.5", "1.5", "-0.3", "DC"):
                    # end node listing
                    break
                clean_token = token.strip(",()")
                if clean_token not in exclude_nodes:
                    node_usage[clean_token] += 1

        # Identify floating or single-connection nodes
        floating_nodes = [n for n, count in node_usage.items() if count <= 1]

        # Add a capacitor to ground for each floating node
        for node in floating_nodes:
            self.netlist.append(f"Rf_shunt_{node} {node} 0 {shunt_value}")

    def construct_netlist(self):
        """
        Full netlist construction routine:
          1. Construct DRAM cell subcircuits
          2. Construct senseamps
          3. Construct DC + PWL voltage sources
          4. Shunt floating nodes
          5. Save netlist to disk
        """
        self.construct_dram_netlist()
        self.construct_senseamp_netlist()
        self.construct_voltage_sources()
        self.analyze_and_shunt_floating_nodes(
            exclude_nodes={"0", "Vdd", "Vss", "Vcell", "Vpre"}
        )
        # Finally, write the netlist
        self.save_netlist()

    def save_netlist(self):
        """
        Write out the netlist to array_{rows}_{cols}.cir in the base_folder.

        Format:
          *circuit name
          .OPTIONS ...
          .TRAN ...
          .INCLUDE ...
          .SAVE ...
          .PARAM ...
          ...
          .end
        """
        if self.netlist_path:
            print(f"Netlist path already assigned as '{self.netlist_path}'.")
            pass
        else:
            netlist_path = os.path.join(self.base_folder, f"array_{self.rows}_{self.cols}.cir")
            self.netlist_path = netlist_path
        with open(self.netlist_path, "w") as file:
            file.write(f"*circuit array_{self.rows}_{self.cols}\n")
            file.write(f"*created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            file.write(self.OPT_COMMAND + "\n")
            file.write(f"\n.OPTIONS DEVICE TEMP={self.temperature}\n")

            # e.g. .TRAN 10n 200n UIC
            file.write(f".TRAN {self.SIM_STEP}n {self.time_ns}n")
            if self.initial_cond:
                file.write(" UIC")
            file.write("\n")

            if self.parallel_processing:
                file.write(f".OPTIONS PARALLEL DOMAINS={self.np}\n")

            file.write(f".INCLUDE {self.dram_lib_path}\n")
            file.write(".PRINT TRAN " + " ".join([f"V({n.strip()})" for n in self.nodes_to_save]) + "\n")
            file.write(f".PARAM C_shunt={self.C_SHUNT}\n")
            file.write(f".PARAM R_shunt={self.R_SHUNT}\n")

            # Add all netlist lines from self.netlist
            for line in self.netlist:
                file.write(line + "\n")

            # Possibly user-provided extra commands
            file.write(self.simulation_commands)

            file.write("\n.end\n")
        print(f"Netlist saved to '{self.netlist_path}'.")

    def save_nodes(self, node_list=None):
        """
        List which nodes (or signals) you want to save in .SAVE lines for Xyce.

        node_list : list
            Additional node names to track.
        """
        if node_list is None:
            node_list = []
        self.nodes_to_save = [f"SN_{i}_{j}" for i in range(self.rows) for j in range(self.cols)]
        if len(node_list) != 0:
            self.nodes_to_save += node_list

    # -------------------------------------------------------------------------
    #  Signal Initialization & Setting
    # -------------------------------------------------------------------------
    def initialize_signals(self):
        """
        Prepopulate self.signal_transitions and self.current_values
        with keys for all signals used in read/write/precharge.

        This helps keep code consistent and avoid missing signals.
        """
        all_signals = []

        # Row-based signals
        for i in range(self.rows):
            all_signals.append(f"WL_in0_{i}")

        # Col-based signals
        for j in range(self.cols):
            all_signals.append(f"W_{j}")
            all_signals.append(f"WI_{j}")

        # Extra signals
        extra_signals = ["WE", "SAE", "SAEd", "Eq"]
        all_signals.extend(extra_signals)

        for sig in all_signals:
            self.signal_transitions[sig] = []
            self.current_values[sig] = None

    def set_signal(self, signal_name, time_ns, new_value):
        """
        Sets 'signal_name' to 'new_value' at 'time_ns' (int, ns). If the old value
        differs, we add a short slope from (time_ns - TRANSITION_DT, old_value)
        to (time_ns, new_value).

        This prevents infinite dV/dt in the PWL file.
        """
        old_value = self.current_values[signal_name]
        if old_value is None:
            # First time => just append the first point
            self.signal_transitions[signal_name].append(
                (time_ns * 1e-9, new_value)
            )
            self.current_values[signal_name] = new_value
            return

        # If no meaningful change, do nothing
        if abs(new_value - old_value) < 1e-15:
            return

        # Create two points: (time_ns - dt, old_value) and (time_ns, new_value)
        time_s = time_ns * 1e-9
        dt_s = self.TRANSITION_DT
        if time_s >= dt_s:
            self.signal_transitions[signal_name].append((time_s - dt_s, old_value))
        self.signal_transitions[signal_name].append((time_s, new_value))

        self.current_values[signal_name] = new_value

    # -------------------------------------------------------------------------
    #  Routines to simulate memory operations
    # -------------------------------------------------------------------------
    def precharge(self, duration_ns=None):
        """
        Simulate a precharge operation for 'duration_ns'.

        Steps:
         1) Turn 'Eq' high at the start to tie bitlines, then low near the end.
         2) Keep everything else default low or safe states.
        """
        if duration_ns is None:
            duration_ns = self.PRECHARGE_DURATION

        t_start   = self.time_ns
        t_eq_off  = t_start + duration_ns - 2
        t_end     = t_start + duration_ns

        # Setup row/col signals
        for i in range(self.rows):
            self.set_signal(f"WL_in0_{i}", t_start, self.V_LOW_WL)
        for j in range(self.cols):
            self.set_signal(f"W_{j}",   t_start, self.V_LOW_W)
            self.set_signal(f"WI_{j}",  t_start, self.V_HIGH_W)

        # SAE, SAEd, WE, Eq
        self.set_signal("SAE", t_start, self.V_LOW_GENERIC)
        self.set_signal("SAEd",t_start, self.V_HIGH_GENERIC)
        self.set_signal("WE",  t_start, self.V_LOW_GENERIC)
        self.set_signal("Eq",  t_start, self.V_HIGH_GENERIC)

        # Turn off Eq near the end
        self.set_signal("Eq", t_eq_off, self.V_LOW_GENERIC)

        # Advance time
        self.time_ns = t_end

    def read(self, row, duration_ns=None):
        """
        Simulate a read operation on 'row' for duration_ns.
        1) WL_in0[row] goes HIGH after 2 ns, then LOW 4 ns before end
        2) SAE toggles high/low mid-cycle
        3) Everything else is set to safe states
        """
        if duration_ns is None:
            duration_ns = self.READ_WRITE_DURATION

        t_start        = self.time_ns
        t_activate_sa  = t_start + 2
        t_deactivate_wl= t_start + duration_ns - 4
        t_deactivate_sa= t_start + duration_ns - 2
        t_end          = t_start + duration_ns

        # At start, set everything to known state
        for i in range(self.rows):
            if i == row:
                self.set_signal(f"WL_in0_{i}", t_start, self.V_HIGH_WL)
            else:
                self.set_signal(f"WL_in0_{i}", t_start, self.V_LOW_WL)

        for j in range(self.cols):
            self.set_signal(f"W_{j}",  t_start, self.V_LOW_W)
            self.set_signal(f"WI_{j}", t_start, self.V_HIGH_W)

        self.set_signal("WE",  t_start, self.V_LOW_GENERIC)
        self.set_signal("SAE", t_start, self.V_LOW_GENERIC)
        self.set_signal("SAEd",t_start, self.V_HIGH_GENERIC)
        self.set_signal("Eq",  t_start, self.V_LOW_GENERIC)

        # Activate sense amps after 2 ns
        self.set_signal("SAE", t_activate_sa,  self.V_HIGH_GENERIC)
        self.set_signal("SAEd",t_activate_sa,  self.V_LOW_GENERIC)

        # Deactivate row & SA near end
        self.set_signal(f"WL_in0_{row}", t_deactivate_wl, self.V_LOW_WL)
        self.set_signal("SAE",           t_deactivate_sa, self.V_LOW_GENERIC)
        self.set_signal("SAEd",          t_deactivate_sa, self.V_HIGH_GENERIC)

        self.time_ns = t_end

    def write(self, row, data_pattern, duration_ns=None):
        """
        Simulate a write operation:
         - WL_in0[row] goes HIGH at t_start+2, LOW near the end
         - WE goes HIGH at the same time, then LOW near end
         - W/ WI signals are set according to data_pattern (e.g., "1010")

        data_pattern : list/str
            e.g. ["1","0","1","1"] or "1011" for col bits
        """
        if duration_ns is None:
            duration_ns = self.READ_WRITE_DURATION

        t_start     = self.time_ns
        t_activate  = t_start + 2
        t_deactivate= t_start + duration_ns - 2
        t_end       = t_start + duration_ns

        # At start, everything low except W/ WI set per data bit
        for i in range(self.rows):
            self.set_signal(f"WL_in0_{i}", t_start, self.V_LOW_WL)

        # pattern indexing
        for j in range(self.cols):
            bit = data_pattern[j % len(data_pattern)]
            w_val  = self.V_HIGH_W if bit == "1" else self.V_LOW_W
            wi_val = self.V_LOW_W  if bit == "1" else self.V_HIGH_W
            self.set_signal(f"W_{j}",  t_start, w_val)
            self.set_signal(f"WI_{j}", t_start, wi_val)

        self.set_signal("WE",  t_start, self.V_LOW_GENERIC)
        self.set_signal("SAE", t_start, self.V_LOW_GENERIC)
        self.set_signal("SAEd",t_start, self.V_HIGH_GENERIC)
        self.set_signal("Eq",  t_start, self.V_LOW_GENERIC)

        # Activate row for writing
        self.set_signal(f"WL_in0_{row}", t_activate, self.V_HIGH_WL)
        self.set_signal("WE",           t_activate, self.V_HIGH_GENERIC)

        # Deactivate near the end
        self.set_signal(f"WL_in0_{row}", t_deactivate, self.V_LOW_WL)
        self.set_signal("WE",           t_deactivate, self.V_LOW_GENERIC)

        self.time_ns = t_end

    def wait(self, duration_ns):
        """
        Just advance the time by 'duration_ns' without changing any signals.
        """
        self.time_ns += duration_ns

    # -------------------------------------------------------------------------
    #  Final: Dump PWL Files
    # -------------------------------------------------------------------------
    def write_to_csv(self):
        """
        Write each signal's transitions to a .pwl file under base_folder/test.
        Each .pwl file has lines:  time_s   voltage
        """
        test_dir = os.path.join(self.base_folder, self.test)
        os.makedirs(test_dir, exist_ok=True)

        for signal_name, transitions in self.signal_transitions.items():
            transitions.sort(key=lambda x: x[0])  # sort by time
            df = pd.DataFrame(transitions, columns=["time_s", "voltage"])
            csv_path = os.path.join(test_dir, f"{signal_name}.pwl")
            df.to_csv(csv_path, index=False, header=False, sep="\t")
        print(f"All .pwl files written to '{test_dir}'.")

    # -------------------------------------------------------------------------
    #  Debug / Print Utilities
    # -------------------------------------------------------------------------
    def print_params(self):
        """Print DRAM and senseamp param matrices to console (for debugging)."""
        print("DRAM Parameters:")
        for key, value in self.dram_params.items():
            print(f"{key}:")
            print(value)
            print()
        print("Sense Amplifier Parameters:")
        for key, value in self.senseamp_params.items():
            print(f"{key}:")
            print(value)

    def print_variation(self):
        """Print the variation dictionary for debugging."""
        print("DRAM Variation:")
        for key, value in self.dram_variation.items():
            print(f"{key}: {value}")
        print("Sense Amplifier Variation:")
        for key, value in self.senseamp_variation.items():
            print(f"{key}: {value}")

    def run_simulation(self):
        """
        Stub method. In a future version, you might call Xyce here,
        for instance:
          mpirun -np <N> xyce <netlist>
        """
        print("Simulation method not implemented. Use external scripts to run Xyce.")
