import pathlib
import time
from anisoap.utils.code_timer import SimpleTimer
from anisoap.utils.cyclic_list import CGRCacheList
from io import TextIOWrapper
from typing import Any
import gc   # for manual garbage collection

_MOST_RECENT_VER = 1

# ------------------------ Configurations Explanation ------------------------ # 
"""
_comp_version: list[int]
    Compares the runtime / SSE for all versions specified. Version 0 (Python
    implementation without caching) will always be added as the baseline.

    Simplified changelogs:
    Version 0: Initial implementation. Guaranteed to be correct, but it is slow.
    Version 1: Ported computation of moments to Rust + CGR matrix is cached for
               each l_max.

_cache_size: int
    Determines the cache size for CGRCacheList. It will only be used for version >= 1.
    The cache will be reset and recomputed when the version changes.

_test_files: list[str]
    A list that determines which .xyz files to test for.
    The files must be stored at {proj_root}/benchmarks/two_particle_gb/ directory
    with .xyz extensions.
"""
# ------------------------------ Configurations ------------------------------ # 
# See above for explanations for each variables.
_comp_version = [_MOST_RECENT_VER]
_cache_size = 5
_test_files = [  # file name: repeat number
        "ellipsoid_frames"
        # "both_rotating_in_z", # Results in key error in frames.arrays['c_q']
        # "face_to_face",
        # "random_rotations",
        # "side_to_face",
        # "side_to_side",
        # "single_rotating_in_y",
        # "single_rotating_in_z"
    ]

# ------------------------------ Hyperparameter ------------------------------ #
# Performs one iteration for each parameter sets per file.
#          lmax     σ       r_cut,  σ1,     σ2,     σ3
_params = [[10,     2.0,    5.0,    3.0,    2.0,    1.0],
           [10,     3.0,    4.0,    3.0,    2.0,    1.0],
           [ 7,     4.0,    6.0,    3.0,    2.0,    1.0],
           [ 5,     3.0,    5.0,    5.0,    3.0,    1.0],
           [10,     2.0,    5.0,    3.0,    3.0,    0.8]]

# --------------------------- Configuration Checks --------------------------- #
# None
if 0 not in _comp_version:
    _comp_version.insert(0, 0)

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

def single_pass(
        file_path: str,
        params: list[float], *,
        version: int = _MOST_RECENT_VER,
        cache_list: CGRCacheList = None,
        timer: SimpleTimer = None
    ) -> (list[list[float]], list[Any]):  # returns (result, extra_info)
    if timer is not None:
        timer.mark_start()

    # parameter decomposition
    l_max = params[0]
    sigma = params[1]
    r_cut = params[2]
    a1, a2, a3 = params[3:]

    if timer is not None:
        timer.mark("1. variable init")

    frames = read(file_path, ':')
    if timer is not None:
        timer.mark("2. file reading")

    representation = EllipsoidalDensityProjection(max_angular=l_max, 
                                                  radial_basis_name='gto', 
                                                  rotation_type='quaternion',
                                                  rotation_key='c_q',
                                                  cutoff_radius=r_cut,
                                                  radial_gaussian_width=sigma)
    if timer is not None:
        timer.mark("3. constructing EDP")

    for frame in frames:
        frame.arrays["c_diameter[1]"] = a1 * np.ones(len(frame))
        frame.arrays["c_diameter[2]"] = a2 * np.ones(len(frame))
        frame.arrays["c_diameter[3]"] = a3 * np.ones(len(frame))
        frame.arrays["quaternions"] = frame.arrays['c_q']

    if timer is not None:
        timer.mark("4. constructing frames")
    
    # timer works internally inside "transform"
    if timer is not None:
        internal_timer = SimpleTimer()
        rep_raw = representation.transform(frames, show_progress=False, timer=internal_timer, version=version)
        timer.mark("5. repr transform")
        timer.collect_and_append(internal_timer)
        timer.mark_start()
    else:
        rep_raw = representation.transform(frames, show_progress=False, version=version)

    rep = equistore.operations.mean_over_samples(rep_raw, sample_names="center")
    if timer is not None:
        timer.mark("6. mean over samples")
    
    anisoap_nu1 = standardize_keys(rep)
    if timer is not None:
        timer.mark("7. standardize keys")
        internal_timer.clear_time()

    if timer is not None:
        my_cg = ClebschGordanReal(l_max, version=version, cache_list=cache_list, timer=internal_timer)
        timer.mark("8. constructing CGR")
        timer.collect_and_append(internal_timer)
        timer.mark_start()
    else:
        my_cg = ClebschGordanReal(l_max, version=version, cache_list=cache_list)

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

    # NOTE: frame.arrays["quaternions"] seems to output a matrix with two quaternions
    #       each one represented as a row of the matrix. It also seems that two quaternions
    #       are identical, so it is set to return only the first row for now.
    #!      Change the return statement if that changes
    return x, [frame.arrays["quaternions"][0]]

def total_error(result1: list[list[float]], result2: list[list[float]]) -> float:
    if len(result1) != len(result2):
        raise ValueError("Two matrices must have same number of rows.")
    
    sse = 0.0
    for i in range(len(result1)):
        if len(result1[i]) != len(result2[i]):
            raise ValueError(f"Two matrices have different number of columns at row {i}")
        
        for j in range(len(result1[i])):
            sse = (result1[i][j] - result2[i][j]) ** 2
    return sse

def get_key(ver: int, param_index: int, file: str) -> str:
    """
    Get the key of the dictionary, given version, parameter index, and the file name.
    Format is: v{ver}_p{param_index}_{file}
    """
    return f"v{ver}_p{param_index}_{file}"

def get_comp_key(key: str) -> str:
    """
    Get the key of the equivalent parameter set and file of the original implementation (v0),
    given the current key.
    """
    return "v0_" + "_".join(key.split("_")[1:]) # replace v"n" with v0 for any number n.

def keys_in_order() -> list[str]:
    """
    Returns a list of all tested combinations, in order.
    """
    all_tests_list = []
    for ver in _comp_version:
        for file_name in _test_files:
            for param_index in range(len(_params)):
                all_tests_list.append(get_key(ver, param_index + 1, file_name))
    return all_tests_list

def write_param_summary(file: TextIOWrapper, extra_info: list[Any]):
    file.write("----------------- Parameter Summary -----------------\n")
    file.write("Parameter Set,l_max,sigma,r_cut,sigma_1,sigma_2,sigma_3,Rotation Quaternion\n")
    quat_vec = ["i", "j", "k"]

    for param_index, param in enumerate(_params):
        # l_max (param[0]) is an integer, so it will be treated differently from the rest.
        other_params = ",".join([f"{val:.01f}" for val in param[1:]])
        file.write(f"p{param_index + 1},{param[0]},{other_params},")

        # NOTE: Since the rotation quaternion does not depend on the file or version
        #       (considering the constructor of EDP, which does not depend on either),
        #       we can use v0 and any test_file's (in this case, index 0) rotation quaternion. 
        quat_key = get_key(0, param_index + 1, _test_files[0])
        quat = extra_info.get(quat_key)[0]
        file.write(f"{quat[0]:.4f}")
        for i, suffix in enumerate(quat_vec):
            if quat[i] < 0:
                file.write(" - ")
            else:
                file.write(" + ")

            file.write(f"{abs(quat[i + 1]):.4f}{suffix}")
        file.write("\n")

def write_result_summary(file: TextIOWrapper, timer: SimpleTimer, err_dict: dict[str, float]):
    file.write("------------------ Overall Summary ------------------\n")
    all_tests_list = keys_in_order()
    file.write("Name,Runtime (sec),Runtime Change (from v0),SSE (from v0)\n")
    timer_dict = timer.sorted_dict()

    for key in all_tests_list:
        curr_runtime = timer_dict.get(key)[0]
        original_runtime = timer_dict.get(get_comp_key(key))[0]
        percent_diff = (curr_runtime - original_runtime) / original_runtime * 100
        file.write(f"{key},{curr_runtime:.4f},{percent_diff:.2f}%,{err_dict.get(key):.4e}\n")

    file.write("Note: Negative runtime change means computation was faster compared to the original implementation.\n")

def write_raw_data(file: TextIOWrapper, raw_data: dict[str, list[list[float]]]):
    file.write("----------------- Raw Data (Result) -----------------\n")
    all_tests_list = keys_in_order()
    for key in all_tests_list:
        file.write(f"Output of {key}\n")
        result_mat = raw_data.get(key)
        for row in result_mat:
            file.write(",".join({f"{val:.10f}" for val in row}) + "\n")
        file.write("\n")


if __name__ == "__main__":
    actual_file_name = "comp_result_v" + ",".join([str(ver) for ver in _comp_version])
    write_name = str(pathlib.Path(__file__).parent.absolute()) + "/time_results/" + actual_file_name + ".csv"
    raw_results = dict()
    extra_infos = dict()
    errors = dict()
    
    with open(write_name, "w") as out_file:
        out_file.write(f"Comparison of versions [{' '.join([str(ver) for ver in _comp_version])}]\n\n")
        
        out_file.write("---------------- Initialization Info ----------------\n")
        out_file.write(f"initial_import time (sec), {import_duration: .04f}\n\n")

        single_pass_timer = SimpleTimer()
        internal_timer = None   # disable internal timer

        for ver in _comp_version:
            matrix_cache = CGRCacheList(_cache_size)
            for test_file in _test_files:
                file_path = str(pathlib.Path(__file__).parent.parent.absolute()) + "/benchmarks/two_particle_gb/" + test_file + ".xyz"

                for param_index, param in enumerate(_params):
                    iter_str = get_key(ver, param_index + 1, test_file)
                    print(f"Computation for {iter_str} has started")

                    single_pass_timer.mark_start()
                    comp_result, ex_info = single_pass(file_path, param, version=ver, cache_list=matrix_cache,timer=internal_timer)
                    single_pass_timer.mark(iter_str)    

                    raw_results.update({iter_str: comp_result})
                    extra_infos.update({iter_str: ex_info})

                    # Get SSE based on the original implementation (v0) of equivalent parameter set and the test file
                    errors.update({iter_str: total_error(raw_results.get(get_comp_key(iter_str)), comp_result)})

                    print(f"Computation for {iter_str} has successfully finished")
            
            # Make sure garbage collection does not interfere with the next iteration (version change).
            matrix_cache.clear_cache()
            del matrix_cache
            gc.collect()

        write_param_summary(out_file, extra_infos)
        out_file.write("\n")
        write_result_summary(out_file, single_pass_timer, errors)
        out_file.write("\n")
        write_raw_data(out_file, raw_results)
