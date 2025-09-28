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



