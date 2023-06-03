import pathlib
import time
from anisoap.utils import SimpleTimer
from anisoap.utils.cyclic_list import CGRCacheList

# ------------------------ Configurations Explanation ------------------------ # 
"""
_file_name: str
    _file_name is used to name the actual file name with
    
    if _write_mode == "time" 
        actual_file_name = _file_name + "_" + SimpleTimer.default_coll_mode
    elif _write_mode == "result"
        actual_file_name = _file_name + "_" + _moment_fn_lang

    The result will be saved to {proj_root}/tests/time_results/<actual_file_name>.csv

_write_mode: str ("time", "result")
    _write_mode = "time" records the code execution time from SimpleTimer.
    _write_mode = "result" records the resulting 2D matrix with each number
                  represented as signed, 10 decimal digit floats

SimpleTimer.default_coll_mode: str ("avg", "sum", "min", "max) 
                            or lambda x: ( ... ) with x as a list of floats
    When certain bits of code is executed multiple times (due to loops), the
    timer records time for each iterations and collects the times into a single
    float. Available modes: "avg", "sum", "min", "max", or any lambda functions
    of form lambda x: ( ... ) in which x is a list of floats.
    This is a static variable inside the SimpleTimer class, hence this notation.

_coll_mode_name: str
    This is used to generate actual save file name when SimpleTimer.default_coll_mode
    is a lambda function. Please use a short but descriptive name for your lambda
    function. It can be None if SimpleTimer.default_coll_mode was one of the strings.

_moment_fn_lang: str ("rust" or "python")
    Selects a language for computing moments.
    "rust" calls compute_moments defined in Rust FFI (see inside anisoap_rust_lib)
    "python" calls compute_moments_inefficient_implementation in moments_generator.py

_test_files: dict[str, int]
    A dictionary that determines how many iterations each test files will run.
    The files must be stored at {proj_root}/benchmarks/two_particle_gb/ directory
    with .xyz extensions.
    
    However, when _write_mode == "time", note that the maximum number of single
    pass supported is 16. (This is due to limitation of my Excel file used to
    analyze the results). Therefore, sum of all integers (next to file names)
    must be less than or equal to 16 if _write_mode == "time"; otherwise, it will
    not work.

_cache_list: CGRCacheList(cache_size: int)
    A cyclic list used to cache the pre-compted CG matrices for given l_max values.
    You can set it to None to disable caching.
"""
# ------------------------------ Configurations ------------------------------ # 
# See above for explanations for each variables.
_file_name = "res_cmp" 
_write_mode = "result"
SimpleTimer.default_coll_mode = "avg"
_coll_mode_name = None
_moment_fn_lang = "rust"
_test_files = {  # file name: repeat number
        "ellipsoid_frames": 4,
        # "both_rotating_in_z": 1, # Results in key error in frames.arrays['c_q']
        # "face_to_face": 1,
        # "random_rotations": 1,
        # "side_to_face": 1,
        # "side_to_side": 1,
        # "single_rotating_in_y": 1,
        # "single_rotating_in_z": 1
    }
_cache_list = CGRCacheList(5)

# --------------------------- Configuration Checks --------------------------- #
if type(_file_name) != str:
    raise TypeError("Type of _file_name must be a string")

if _write_mode not in ["time", "result"]:
    raise ValueError("_write_mode currently supports only 'time' or 'result' modes")

if SimpleTimer.default_coll_mode not in ["avg", "sum", "min", "max"]:
    if not callable(SimpleTimer.default_coll_mode):
        raise ValueError("SimpleTimer.default_coll_mode has to be of specific types. Please read the hints above")

if _moment_fn_lang not in ["rust", "python"]:
    raise ValueError("_moment_fn_lang currently supports only rust or python implementations")

if type(_cache_list) != CGRCacheList and _cache_list is not None:
    raise TypeError("_cache_list must be CGRCacheList (to enable caching) or None (to disable caching)")
 
# -------------------------------- Main Codes -------------------------------- # 
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
        rep_raw = representation.transform(frames, show_progress=False, timer=internal_timer, moment_fn_lang=_moment_fn_lang)
        timer.mark("5. repr transform")
        timer.collect_and_append(internal_timer)
        timer.mark_start()
    else:
        rep_raw = representation.transform(frames, show_progress=False, moment_fn_lang=_moment_fn_lang)

    rep = equistore.operations.mean_over_samples(rep_raw, sample_names="center")
    if timer is not None:
        timer.mark("6. mean over samples")
    
    anisoap_nu1 = standardize_keys(rep)
    if timer is not None:
        timer.mark("7. standardize keys")
        internal_timer.clear_time()

    my_cg = ClebschGordanReal(l_max, timer=internal_timer, cache_list=_cache_list)
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
    if _write_mode == "result":
        actual_file_name = _file_name + "_" + _moment_fn_lang
    elif _write_mode == "time":
        if type(SimpleTimer.default_coll_mode) == str:
            actual_file_name = _file_name + "_" + SimpleTimer.default_coll_mode
        else:
            actual_file_name = _file_name + "_" + _coll_mode_name
    else:
        actual_file_name = "unknown_mode"

    write_name = str(pathlib.Path(__file__).parent.absolute()) + "/time_results/" + actual_file_name + ".csv"
    out_file = open(write_name, "w")
    if _write_mode == "result":
        out_file.write("Final results\n")
    else:
        out_file.write("initial import\n")
        out_file.write(str(import_duration) + "\n")

    if _write_mode == "time":
        flatten_name = []
        for name, iter_val in _test_files.items():
            for iter_num in range(iter_val):
                flatten_name.append(name + ": iter" + str(iter_num + 1))
        out_file.write("stage," + ",".join(flatten_name) + "\n")
    
    code_timer = SimpleTimer()
    for test_file, rep_num in _test_files.items():
        file_path = str(pathlib.Path(__file__).parent.parent.absolute()) + "/benchmarks/two_particle_gb/" + test_file + ".xyz"
        for rep_index in range(rep_num):
            print(f"Computation for {test_file}, iteration {rep_index + 1} has started")
            
            if _write_mode == "result":
                out_file.write(f"{test_file}, iter {rep_index + 1}\n")
            
            comp_result = single_pass(file_path, timer=code_timer)

            if _write_mode == "result":
                for row in comp_result:
                    out_file.write(",".join([f"{val:0.10f}" for val in row])) 
                    out_file.write("\n")
                out_file.write("\n")

            print(f"Computation for {test_file}, iteration {rep_index + 1} has successfully finished")
    
    if _write_mode == "time":
        for key, val in code_timer.sorted_dict().items():
            out_file.write(key + "," + ",".join([str(v) for v in val]) + "\n")
    out_file.close()
