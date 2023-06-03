import pathlib
import time
from anisoap.utils import SimpleTimer

start_time = time.perf_counter()
from ase.io import read
from skmatter.preprocessing import StandardFlexibleScaler
from anisoap.representations import EllipsoidalDensityProjection
from anisoap.utils import standardize_keys, ClebschGordanReal, cg_combine
import numpy as np
import equistore
import_duration = time.perf_counter() - start_time
del start_time

def single_pass(file_path: str, *, timer: SimpleTimer = None):
    if timer is not None:
        timer.mark_start()

    a1 = 3.0
    a3 = 0.8
    l_max = 10  # ??
    if timer is not None:
        timer.mark("1. variable init")

    frames = read(file_path, ':')
    if timer is not None:
        timer.mark("2. file reading")

    representation = EllipsoidalDensityProjection(max_angular=l_max, 
                                                  radial_basis_name='gto', 
                                                  rotation_type='quaternion',
                                                  rotation_key='c_q',
                                                  cutoff_radius=6.0,
                                                  radial_gaussian_width=6.0)
    if timer is not None:
        timer.mark("3. constructing EDP")

    for frame in frames:
        frame.arrays["c_diameter[1]"] = a1 * np.ones(len(frame))
        frame.arrays["c_diameter[2]"] = a1 * np.ones(len(frame))
        frame.arrays["c_diameter[3]"] = a3 * np.ones(len(frame))
        frame.arrays["quaternions"] = frame.arrays['c_q']
    if timer is not None:
        timer.mark("4. constructing frames")
    
    # timer works internally inside "transform"
    if timer is not None:
        internal_timer = SimpleTimer()
        rep_raw = representation.transform(frames, show_progress=False, timer=internal_timer)
        timer.mark("5. repr transform")
        timer.collect_and_append(internal_timer)
        timer.mark_start()

    else:
        rep_raw = representation.transform(frames, show_progress=False)

    rep = equistore.operations.mean_over_samples(rep_raw, sample_names="center")
    if timer is not None:
        timer.mark("6. mean over samples")
    
    anisoap_nu1 = standardize_keys(rep)
    if timer is not None:
        timer.mark("7. standardize keys")
        internal_timer.clear_time()

    my_cg = ClebschGordanReal(l_max, timer=internal_timer)
    if timer is not None:
        timer.mark("8. constructing CGR")
        timer.collect_and_append(internal_timer)
        timer.mark_start()

    anisoap_nu2 = cg_combine(
        anisoap_nu1,
        anisoap_nu1,
        clebsch_gordan=my_cg,
        l_cut=0,
        other_keys_match=["species_center"],
    )
    if timer is not None:
        timer.mark("9. cg_combine")

    x_raw = anisoap_nu2.block().values.squeeze()
    x_raw = x_raw[:, x_raw.var(axis=0)>1E-12]
    if timer is not None:
        timer.mark("10. squeeze and filter")

    x_scaler = StandardFlexibleScaler(column_wise=True).fit(
        x_raw
    )
    if timer is not None:
        timer.mark("11. SFS fit")

    x = x_scaler.transform(x_raw)
    if timer is not None:
        timer.mark("12. scaler transform")

    return x

if __name__ == "__main__":
    write_name = str(pathlib.Path(__file__).parent.absolute()) + "/time_results/" + "ignore_this" + ".csv"
    out_file = open(write_name, "w")
    out_file.write("initial import\n")
    out_file.write(str(import_duration) + "\n")

    timing_criteria = []
    file_names = {  # file name: repeat number
        "ellipsoid_frames": 16,
        # "both_rotating_in_z": 1, # Results in key error in frames.arrays['c_q']
        # "face_to_face": 1,
        # "random_rotations": 1,
        # "side_to_face": 1,
        # "side_to_side": 1,
        # "single_rotating_in_y": 1,
        # "single_rotating_in_z": 1
    }
    
    flatten_name = []
    for name, iter_val in file_names.items():
        for iter_num in range(iter_val):
            flatten_name.append(name + ": iter" + str(iter_num + 1))
    out_file.write("stage," + ",".join(flatten_name) + "\n")
    
    code_timer = SimpleTimer()
    for test_file, rep_num in file_names.items():
        file_path = str(pathlib.Path(__file__).parent.parent.absolute()) + "/benchmarks/two_particle_gb/" + test_file + ".xyz"
        for rep_index in range(rep_num):
            print(f"Computation for {test_file}, iteration {rep_index + 1} has started")
            single_pass(file_path, timer=code_timer)
            print(f"Computation for {test_file}, iteration {rep_index + 1} has successfully finished")
    
    for key, val in code_timer.sorted_dict().items():
        out_file.write(key + "," + ",".join([str(v) for v in val]) + "\n")
    out_file.close()
