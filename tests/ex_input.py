import equistore
import numpy as np
from anisoap.utils import standardize_keys, ClebschGordanReal, cg_combine
from anisoap.representations import EllipsoidalDensityProjection
from skmatter.preprocessing import StandardFlexibleScaler
from ase.io import read
import pathlib
import time
from anisoap.utils.code_timer import SimpleTimer, SimpleTimerCollectMode
from anisoap.utils.cyclic_list import CGRCacheList

# Imported for type annotation purposes
from io import TextIOWrapper
from typing import Any

import gc           # for manual garbage collection
import warnings     # to disable warning messages

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# ------------------------ Configurations Explanation ------------------------ #
"""
_comp_version: list[int]
    Compares the runtime / SSE for all versions specified. Version 0 (Python
    implementation without caching) will always be added as the baseline.

    Simplified changelogs:
    Version 0: Initial implementation. Guaranteed to be correct, but it is slow.
    Version 1: Ported computation of moments to Rust + CGR matrix is cached for
               each l_max.
    Version 2: Version 1 change is maintained. Changed the loop structure of
               pairwise_ellip_expansion such that it does not loop through every
               possible combinations of species pair (only loops through pairs in
               neighbor_list)

_test_files: list[str]
    A list that determines which .xyz files to test for.
    The files must be stored at {proj_root}/benchmarks/two_particle_gb/ directory
    with .xyz extensions.

_timer_collect_mode: SimpleTimerCollectMode
    If there are multiple passes performed for same set of parameters, this mode
    determines how to process time for final output file. Default mode is MAX, in
    which the timer records the worst time across all iterations.
    Enter all modes to report as a list.
    
_cache_size: int
    Determines the cache size for CGRCacheList. It will only be used for version >= 1.
    The cache will be reset and recomputed when the version changes.

_skip_raw_data: bool
    Determines whether or not to write raw output (matrix) to the output file.
    Default is false. If set to "True", the csv file may become quite large!
"""
# ------------------------------ Configurations ------------------------------ #
_MOST_RECENT_VER = 2

# See above for explanations for each variables.
_comp_version = [1, _MOST_RECENT_VER]
_test_files = [
    "ellipsoid_frames"
    # "both_rotating_in_z", # Results in key error in frames.arrays['c_q']
    # "face_to_face",
    # "random_rotations",
    # "side_to_face",
    # "side_to_side",
    # "single_rotating_in_y",
    # "single_rotating_in_z"
]
_timer_collect_mode = [SimpleTimerCollectMode.MAX,
                       SimpleTimerCollectMode.AVG, SimpleTimerCollectMode.MED]
_cache_size = 5
_skip_raw_data = True

# set to ignore all waring messages.
# As of version 1, there are two warnings for each iteration:
#   1. UserWarning: In quaternion mode, quaternions are assumed to be in (w,x,y,z) format.
#          * From ellipsoidal_density_projection.py
#   2. UserWarning: periodic boundary conditions are disabled, but the cell matrix is not zero, we will set the cell to zero.
#          * From ase.py
warnings.filterwarnings("ignore")

# ------------------------------ Hyperparameter ------------------------------ #
# Performs one iteration for each parameter sets per file.
# NOTE: If there are identical set of parameters, the repeat number will be merged
#       For example, below example should result in 8 iterations each for p1 ~ p6.
#       p7 ~ p11 should be deleted.

#            lmax    σ       r_cut   σ1      σ2      σ3     repeat
_params = [[[10,     2.0,    5.0,    3.0,    2.0,    1.0],  6],  # p1
           [[10,     3.0,    4.0,    3.0,    2.0,    1.0],  4],  # p2
           [[7,     4.0,    6.0,    3.0,    2.0,    1.0],  3],  # p3
           [[5,     3.0,    5.0,    5.0,    3.0,    1.0],  6],  # p4
           [[10,     2.0,    5.0,    3.0,    3.0,    0.8],  8],  # p5
           #    [[ 8,     0.5,    1.0,    1.0,    0.8,    0.5],  8], # p? << Results in bug from Rascaline?
           [[8,    10.0,   10.0,   10.0,    7.0,    5.0],  8],  # p6

           [[10,     2.0,    5.0,    3.0,    2.0,    1.0],  1],  # identical to p1
           [[5,     3.0,    5.0,    5.0,    3.0,    1.0],  2],  # identical to p4
           [[10,     2.0,    5.0,    3.0,    2.0,    1.0],  1],  # identical to p1
           [[7,     4.0,    6.0,    3.0,    2.0,    1.0],  5],  # identical to p3
           [[10,     3.0,    4.0,    3.0,    2.0,    1.0],  4]]  # identical to p2

# --------------------------- Configuration Checks --------------------------- #


def _identical_param(param1_index: int, param2_index: int) -> bool:
    # Turn on strict, as all parameters should have same length.
    # Index 0 contains all parameters as a list.
    for p1, p2 in zip(_params[param1_index][0], _params[param2_index][0], strict=True):
        #! This loop assumes all parameters are given as numbers. Change the condition
        #! if that changes in the future.
        if abs(p1 - p2) >= 1e-6:
            # return false if any of the parameters differ significantly.
            return False
    return True


def _remove_and_merge_duplicate_param() -> None:
    to_merge: dict[int, list[int]] = dict()  # keeps track of indices to merge.
    to_delete = []
    # Loop to check for all repeating parameters
    for (param_index, _) in enumerate(_params):
        # With loop starting from 0, this code should find the first index at which
        # the parameter repetition occurs.
        for prev_param_index in range(param_index):
            if _identical_param(param_index, prev_param_index) and param_index not in to_delete:
                if param_index not in to_delete:
                    to_delete.append(param_index)

                if prev_param_index not in to_merge:
                    to_merge.update({prev_param_index: [param_index]})
                else:
                    to_merge[prev_param_index].append(param_index)

    # Loops to update _params
    for prev_index, index_list in to_merge.items():
        for index in index_list:
            # Index "1" stores the repetition number. Merge number to the first instance of
            # the identical parameter set.
            _params[prev_index][1] += _params[index][1]

    # Sort the indices to delete in descending order, as deleting element at larger
    # index does not disturb the lower index, while the converse does not hold.
    to_delete = sorted(to_delete, reverse=True)
    for del_index in to_delete:
        _params.pop(del_index)


_remove_and_merge_duplicate_param()

# If any of the repeat number is not valid (0 or less), artificially set to 1.
for param in _params:
    if param[1] < 1:
        param[1] = 1

if _timer_collect_mode == []:
    _timer_collect_mode = [SimpleTimerCollectMode.MAX]

# Always have version 0 to compare against.
if 0 not in _comp_version:
    _comp_version.insert(0, 0)

# Remove duplicate with set. Convert to list then sort the versions in ascending order.
_comp_version = sorted(list(set(_comp_version)))

ClebschGordanReal.cache_list = CGRCacheList(_cache_size)

# -------------------------------- Main Codes -------------------------------- #
start_time = time.perf_counter()
import_duration = time.perf_counter() - start_time
del start_time


def single_pass(
    file_path: str,
    params: list[float], *,
    version: int = _MOST_RECENT_VER
) -> (list[list[float]], list[Any]):  # returns (result, extra_info)
    # parameter decomposition
    l_max = params[0]
    sigma = params[1]
    r_cut = params[2]
    a1, a2, a3 = params[3:]

    frames = read(file_path, ':')
    representation = EllipsoidalDensityProjection(max_angular=l_max,
                                                  radial_basis_name='gto',
                                                  rotation_type='quaternion',
                                                  rotation_key='c_q',
                                                  cutoff_radius=r_cut,
                                                  radial_gaussian_width=sigma)
    for frame in frames:
        frame.arrays["c_diameter[1]"] = a1 * np.ones(len(frame))
        frame.arrays["c_diameter[2]"] = a2 * np.ones(len(frame))
        frame.arrays["c_diameter[3]"] = a3 * np.ones(len(frame))
        frame.arrays["quaternions"] = frame.arrays['c_q']

    # timer works internally inside "transform"
    rep_raw = representation.transform(
        frames, show_progress=False, version=version)
    rep = equistore.operations.mean_over_samples(
        rep_raw, sample_names="center")

    anisoap_nu1 = standardize_keys(rep)
    my_cg = ClebschGordanReal(l_max, version=version)

    anisoap_nu2 = cg_combine(
        anisoap_nu1,
        anisoap_nu1,
        clebsch_gordan=my_cg,
        l_cut=0,
        other_keys_match=["species_center"],
    )

    x_raw = anisoap_nu2.block().values.squeeze()
    x_raw = x_raw[:, x_raw.var(axis=0) > 1E-12]
    x_scaler = StandardFlexibleScaler(column_wise=True).fit(
        x_raw
    )

    x = x_scaler.transform(x_raw)

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
            raise ValueError(
                f"Two matrices have different number of columns at row {i}")

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
    return "v0_" + "_".join(key.split("_")[1:])  # replace v"n" with v0 for any number n.


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


def get_version(key_str: str) -> str:
    return key_str.split("_")[0]


def write_param_summary(file: TextIOWrapper, extra_info: list[Any]):
    file.write("----------------- Parameter Summary -----------------\n")
    file.write(
        "Parameter Set,l_max,sigma,r_cut,sigma_1,sigma_2,sigma_3,Rotation Quaternion\n")
    quat_vec = ["i", "j", "k"]

    for (param_index, (param, _)) in enumerate(_params):
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

    file.write("Name,")
    for mode in _timer_collect_mode:
        if mode == SimpleTimerCollectMode.AVG:
            mode_str = "Average"
        elif mode == SimpleTimerCollectMode.SUM:
            mode_str = "Total"
        elif mode == SimpleTimerCollectMode.MAX:
            mode_str = "Maximum"
        elif mode == SimpleTimerCollectMode.MIN:
            mode_str = "Minimum"
        elif mode == SimpleTimerCollectMode.MED:
            mode_str = "Median"

        file.write(f"{mode_str} runtime (sec),")
    file.write("SSE (from v0)\n")

    prev_ver = "v0"  # version 0 is always the first to be written.
    for key in all_tests_list:
        curr_ver = get_version(key)
        if prev_ver != curr_ver:
            prev_ver = curr_ver
            file.write("\n")

        file.write(f"{key},")
        for mode in _timer_collect_mode:
            collected_timer = SimpleTimer()
            collected_timer.collect_and_append(timer, mode)
            timer_dict = collected_timer.sorted_dict()

            curr_runtime = timer_dict.get(key)[0]
            original_runtime = timer_dict.get(get_comp_key(key))[0]
            percent_diff = (curr_runtime - original_runtime) / \
                original_runtime * 100
            file.write(f"{curr_runtime:.4f} ({percent_diff:.2f}%),")

        file.write(f"{err_dict.get(key):.4e}\n")

    file.write("\nNote: Number in parenthesis after runtime refers to corresponding runtime change compared to original implementation (version 0).\n")
    file.write(
        "      Negative runtime change means computation was faster compared to the original implementation.\n")


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
    actual_file_name = "comp_result_v" + \
        ",".join([str(ver) for ver in _comp_version])
    write_name = str(pathlib.Path(__file__).parent.absolute()) + \
        "/time_results/" + actual_file_name + ".csv"
    raw_results = dict()
    extra_infos = dict()
    errors = dict()

    with open(write_name, "w") as out_file:
        out_file.write(
            f"Comparison of versions [{' '.join([str(ver) for ver in _comp_version])}]\n\n")

        out_file.write(
            "---------------- Initialization Info ----------------\n")
        out_file.write(
            f"initial_import time (sec), {import_duration: .04f}\n\n")

        single_pass_timer = SimpleTimer()

        for ver in _comp_version:
            for test_file in _test_files:
                file_path = str(pathlib.Path(__file__).parent.parent.absolute(
                )) + "/benchmarks/two_particle_gb/" + test_file + ".xyz"

                for (param_index, (param, repeat_no)) in enumerate(_params):
                    iter_str = get_key(ver, param_index + 1, test_file)

                    for rep_index in tqdm(range(repeat_no), desc=f"{iter_str}"):
                        single_pass_timer.mark_start()
                        comp_result, ex_info = single_pass(
                            file_path, param, version=ver)
                        single_pass_timer.mark(iter_str)

                    # Only stores the result and extra info from the last iteration, as all iterations
                    # should lead to identical results
                    raw_results.update({iter_str: comp_result})
                    extra_infos.update({iter_str: ex_info})

                    # Get SSE based on the original implementation (v0) of equivalent parameter set and the test file
                    errors.update({iter_str: total_error(
                        raw_results.get(get_comp_key(iter_str)), comp_result)})

            # Make sure garbage collection does not interfere with the next iteration (version change).
            ClebschGordanReal.cache_list.clear_cache()
            gc.collect()

        write_param_summary(out_file, extra_infos)
        out_file.write("\n")

        write_result_summary(out_file, single_pass_timer, errors)
        out_file.write("\n")

        if not _skip_raw_data:
            write_raw_data(out_file, raw_results)
