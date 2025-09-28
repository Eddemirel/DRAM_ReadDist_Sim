# DRAM_ReadDist_Sim
Fast, measurement-calibrated circuit-level simulator for DRAM read-disturbance (RowHammer/RowPress). SPICE/Verilog-A models, DRAMBender-based fitting, Monte Carlo analysis, and pattern search. Runs on Xyce for reproducible attack/defense studies.

## General Workflow

1) Download library files on your $WORKSPACE

2) pull docker image

docker pull docker.io/eddemirel13/xyce:amd64-v1

3) run docker container linking it to your $WORKSPACE

docker run -it --rm \
  --platform linux/amd64 \
  --cpus=12 \
  -v "$WORKSPACE:/workspace" \
  xyce:amd64-v1 bash


4) compile Verilog-A packages
buildxyceplugin -o dram_bundle \
    /workspace/vA_modules/Isub.va \
    /workspace/vA_modules/edge_pulse_rev.va \
    /workspace/vA_modules/senseamp.va \
    /workspace

dram_bundle.so must appear under /workspace

5) run your MC configuration

python config.run

resulting waveforms and HCfirst values are under the folder MC_<attack_type>_<hammer_count>HC_<N>samples_<timestamp>


## About Library 


