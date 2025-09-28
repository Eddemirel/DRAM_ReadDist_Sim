# DRAM_ReadDist_Sim
Fast, measurement-calibrated circuit-level simulator for DRAM read-disturbance (RowHammer/RowPress). SPICE/Verilog-A models, DRAMBender-based fitting, Monte Carlo analysis, and pattern search. Runs on Xyce for reproducible attack/defense studies.

## General Workflow

### 0) Define your workspace (host absolute path)
export WORKSPACE=/abs/path/to/DRAM_ReadDist_Sim

### 1) Place library files in $WORKSPACE e.g., $WORKSPACE/va_modules/*.va, $WORKSPACE/*.cir, $WORKSPACE/*.py

### 2) Pull the public Docker image
docker pull docker.io/eddemirel13/xyce:amd64-v1

### 3) Run the container (link to your workspace)
docker run -it --rm \
  --platform linux/amd64 \
  --cpus=12 \
  -v "$WORKSPACE:/workspace" \
  docker.io/eddemirel13/xyce:amd64-v1 bash

### (Optional) Verify Xyce and buildxyceplugin exist
Xyce -v
buildxyceplugin -v

### 4) Inside the container: compile Verilog-A to a plugin
buildxyceplugin -o /workspace/dram_bundle \
  /workspace/va_modules/Isub.va \
  /workspace/va_modules/edge_pulse_rev.va \
  /workspace/va_modules/senseamp.va \
  /workspace

### (Optional) Verify artifact and expose it to Xyce
ls -lh /workspace/dram_bundle.so
export XYCE_PLUGIN_PATH=/workspace
echo 'export XYCE_PLUGIN_PATH=/workspace' >> ~/.bashrc

### 5) Run your Monte Carlo configuration
cd /workspace
python3 config.py

### Outputs:
MC_<attack_type>_<hammer_count>HC_<N>samples_<timestamp>/

   ├─ waveforms/
   
   └─ varation_table.csv (or similar)

<h2 id="library-description">Library Description</h2>

<h3>config.py</h3>
<ul>
  <li><strong>Inputs:</strong> User-editable <code>MONTE_CARLO_CONFIG</code> and <code>VARIATION_CONFIG</code>; optional calibrated params.</li>
  <li><strong>Outputs:</strong> <code>variation_config.json</code>, <code>variation_table.csv</code>, run folder names, and a launched Monte Carlo run.</li>
  <li><strong>Connections:</strong> Builds CLI args and seeds, then invokes <code>montecarlo.py</code>. Central place to reproduce experiments from the Dockerized <code>/workspace</code>.</li>
</ul>

<h3>main.py</h3>
<ul>
  <li><strong>Inputs:</strong> CLI overrides to paramater dictionary (e.g., <code>--N</code>, <code>--rows</code>, <code>--attack</code>); optional hardcoded param dicts for quick tests.</li>
  <li><strong>Outputs:</strong> Generated netlist(s) and PWL files, plus status logs.</li>
  <li><strong>Connections:</strong> Instantiates <code>dram_array.Array</code>, runs preset read/write/precharge sequences, and writes local artifacts for rapid checks.</li>
</ul>

<h3>dram_array.py</h3>
<ul>
  <li><strong>Inputs:</strong> Base parameters, per-cell variation matrices, and control sequences.</li>
  <li><strong>Outputs:</strong> Synthesized <code>.cir</code> netlists, <code>.pwl</code> waveforms, and node CSVs.</li>
  <li><strong>Connections:</strong> High-level API used by <code>montecarlo.py</code>/<code>main.py</code> to wire wordlines/bitlines and couple aggressor–victim cells for Xyce.</li>
</ul>

<h3>montecarlo.py</h3>
<ul>
  <li><strong>Inputs:</strong> <code>variation_config.json</code>, CLI args (N, rows, cols, seed, timing), and environment (<code>XYCE_PLUGIN_PATH</code>).</li>
  <li><strong>Outputs:</strong> Per-sample <code>.cir</code>, <code>variation_table.csv</code>, waveforms, and <em>HCfirst</em> summaries in <code>MC_*</code> folders.</li>
  <li><strong>Connections:</strong> Orchestrates sampling and shared waveforms, calls <code>dram_array</code>, runs Xyce, and post-processes outputs.</li>
</ul>

<h3>dram.lib</h3>
<ul>
  <li><strong>Inputs:</strong> <code>.PARAM</code> values passed from each netlist instance.</li>
  <li><strong>Outputs:</strong> Reusable subcircuits (cell, interconnect, wrappers) for simulations.</li>
  <li><strong>Connections:</strong> Included via <code>.include /workspace/dram.lib</code> by generated netlists; complements behavioral <code>.va</code> models.</li>
</ul>

<h3>edge_pulse_rev.va</h3>
<ul>
  <li><strong>Inputs:</strong> Wordline/enable signals and calibrated pulse parameters.</li>
  <li><strong>Outputs:</strong> Leakage current blackbox as analog current source.</li>
  <li><strong>Connections:</strong> Compiled into <code>dram_bundle.so</code>; referenced by netlists to model aggressor excitation dynamics.</li>
</ul>

<h3>Isub.va</h3>
<ul>
  <li><strong>Inputs:</strong> Node voltages, leakage/temperature coefficients.</li>
  <li><strong>Outputs:</strong> Subthreshold/leakage currents driving charge decay.</li>
  <li><strong>Connections:</strong> Part of <code>dram_bundle.so</code>; used wherever baseline retention and read-disturb leakage matter.</li>
</ul>

<h3>senseamp.va</h3>
<ul>
  <li><strong>Inputs:</strong> Bitline/bitline-bar nodes, enable/precharge conditions, initial imbalance.</li>
  <li><strong>Outputs:</strong> Regeneration currents and resolved logic state.</li>
  <li><strong>Connections:</strong> Part of <code>dram_bundle.so</code>; engaged during readout to determine flip behavior with pulses/leakage.</li>
</ul>

<h3>plot.ipynb</h3>
<ul>
  <li><strong>Inputs:</strong> CSV/Xyce <code>.prn</code> exports and experiment CSVs (e.g., <code>variation_table.csv</code>).</li>
  <li><strong>Outputs:</strong> Inline figures and metrics (KDE/Bhattacharyya).</li>
  <li><strong>Connections:</strong> Compares experimental vs simulated Hammer-Count distributions; used interactively for fitting and QC.</li>
</ul>

<h3>DRAM_read_disturbance_paramselect.ipynb</h3>
<ul>
  <li><strong>Inputs:</strong> Experimental Hammer-Count CSVs; optional native/edge-pulse emulators.</li>
  <li><strong>Outputs:</strong> Fitted parameter summaries and diagnostic plots.</li>
  <li><strong>Connections:</strong> Calibration notebook; produces params consumed by <code>config.py</code>/<code>montecarlo.py</code> for measurement-aligned runs.</li>
</ul>


