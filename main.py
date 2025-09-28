# main.py
from dram_array import Array

base_path ="/workspace"
row =5
col =4
temperature=300
test="defaultC"
dram_param_dict = dram_param_dict = {
    "Ron": 1e4, "Roff": 1e14,
    "C_sn": 8e-15, "R_sn": 10,
    "C_bl": 0.3e-15, "C_wl": 0.3e-15, "C_bl2bl": 0.3e-15, "C_wl2wl": 0.3e-15, "C_wl2bl": 0.3e-15,
    "R_wl": 100, "R_bl": 100,
    "I0": 1e-12, "VTH0": 0.6,

    "I_double_10": 1.0e-11,
    "tau_double_10": 1.446976355e-9,
    "k_double_10": 6.583433619,
    "I_single_10": 8.633903362e-10,
    "tau_single_10": 1.505310870e-8,
    "I_const_10": 8.017451958e-11,
    "k_mult_10": 2.624716948,

    "k_switch": 2.333436172e1,

    "I_single_01": 8.633903362e-10,
    "tau_single_01": 1.505310870e-8,
    "I_const_01": 8.017451958e-11,
    "k_mult_01": 2.624716948,
    "I_double_01": 1.0e-11,
    "tau_double_01": 1.446976355e-9,
    "k_double_01": 6.583433619,
}

sense_amp_param_dict= {
            "Ron_SA": 1e4, "Roff_SA": 1e15, "Vt_SA": 0.1, "C_SA": 5e-15, "slope":40
        }

dram_lib_path= "/workspace/dram.lib"
step=10 # in ns

def main():
    array = Array(base_folder=base_path, rows=row, cols=col, temperature=temperature, test=test,
              dram_param_dict=dram_param_dict, senseamp_param_dict=sense_amp_param_dict,
              dram_lib_path=dram_lib_path, step=step)

    # signal definitions
    array.time_ns = 0

    # -------------- DRAM command queue ----------------
    
    # Example command sequence that initializes the array and
    # implements a RowHammer attack with Hammer Count = hc and aggressors agg
    hc = 2
    agg = [0]
    aggressor_dp = "1111"
    victim_dp = "1111"

    array.precharge()

    for r in range(array.rows):
        if r in agg:
            array.write(r, aggressor_dp)
        else:
            array.write(r, victim_dp)
        array.precharge()
        array.read(r)
        array.precharge()

    for h in range(hc):
        for r in agg:
            array.read(r)
            array.precharge()

    array.write_to_csv()
    array.save_nodes([])

    array.construct_netlist()
    print(array.time_ns )

if __name__ == "__main__":
    main()
