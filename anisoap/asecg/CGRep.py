#%%
import numpy as np
from ase import Atom, Atoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from featomic import SoapPowerSpectrum
import metatensor 
import copy
from skmatter.preprocessing import StandardFlexibleScaler
import matplotlib.pyplot as plt

#%%
# Create Dataclass for CG'ed atom
from dataclasses import dataclass

@dataclass
class CGInfo():
    """
    Stores information about the indices to convert to CG Beads.
    """
    cg_indices: list[int] | list[bool]
    name: str
    symbol: str

@dataclass
class CGFrame():
    frame: Atoms 
    cg_indices: list[list]

@dataclass
class CGSystem():
    cg_beads: list[CGFrame]
    rep: metatensor.TensorMap
    gfre: float

# Helper Functions
def get_frames(cg_beads:list[CGFrame]) -> list[Atoms]:
    return [cg_bead.frame for cg_bead in cg_beads]


# To Coarsen:
# def coarsen(AA frame, indices to CG):
    # Return CG'ed bead.

# def find_rigid(CG frame):
    # Returns bead indices within CG frames to CG

# def get_AA_indices(bead_indices):
    # Loop through each bead_index
    # Find the coresponding AA indices. 
    # 

import numpy as np
import ase 
from scipy.spatial.transform import Rotation as R 
from skmatter.metrics import global_reconstruction_error as GRE
from itertools import combinations_with_replacement

def get_center_of_mass(frame: Atoms):
    # For unwrapped systems, set mic=False.
    # For wrapped systems, USUALLY, mic=True, but not always.
    displacements = frame.get_all_distances(mic=False, vector=True)  # vectors is a nxnx3 array, it's the displacement from each point to all other points
    positions = frame[0].position + displacements[0]
    masses = frame.get_masses()
    com = masses @ positions / masses.sum()
    return com 

def get_moments_of_inertia(frame: Atoms, vectors=False):
    """
    COPIED FROM ASE Source: https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_moments_of_inertia
    MODIFIED TO Include MINIMUM IMAGE CONVENTION. 
    Get the moments of inertia along the principal axes.

    The three principal moments of inertia are computed from the
    eigenvalues of the symmetric inertial tensor. Periodic boundary
    conditions are ignored. Units of the moments of inertia are
    amu*angstrom**2.
    """
    # For unwrapped systems, set mic=False.
    # For wrapped systems, USUALLY, mic=True, but not always.
    displacements = frame.get_all_distances(mic=False, vector=True)  # vectors is a nxnx3 array, it's the displacement from each point to all other points
    # com = frame.get_center_of_mass()   # ASE's incorrect center of mass without mic. 
    com = get_center_of_mass(frame)
    positions = frame[0].position + displacements[0] 
    positions -= com  # translate center of mass to origin
    masses = frame.get_masses()

    # Initialize elements of the inertial tensor
    I11 = I22 = I33 = I12 = I13 = I23 = 0.0
    for i in range(len(frame)):
        x, y, z = positions[i]
        m = masses[i]

        I11 += m * (y ** 2 + z ** 2)
        I22 += m * (x ** 2 + z ** 2)
        I33 += m * (x ** 2 + y ** 2)
        I12 += -m * x * y
        I13 += -m * x * z
        I23 += -m * y * z

    Itensor = np.array([[I11, I12, I13],
                        [I12, I22, I23],
                        [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(Itensor)
    if vectors:
        return evals, evecs.transpose()
    else:
        return evals

def get_quat_and_semiaxes(frame: Atoms):
    """
    Returns the correct ellipsoidal semiaxes and orientation.
    """
    mom, evecs = get_moments_of_inertia(frame, vectors=True)
    if np.allclose(np.linalg.det(evecs), -1):
        evecs *= -1
    assert(np.allclose(np.linalg.det(evecs), 1))
    assert(np.allclose(evecs @ evecs.T, np.eye(3)))
    quat = R.from_matrix(evecs.T).as_quat()     # IT's the ROWS! Not the columns, that are the vectors that we care about.
    quat = np.roll(quat, 1)
    squared_semiradii = np.linalg.solve(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), mom) * 5 / frame.get_masses().sum()
    for i, _ in enumerate(squared_semiradii):
        if np.abs(squared_semiradii[i]) < 1e-8:
            squared_semiradii[i] = 0.1
    
    semiradii = np.sqrt(squared_semiradii)
    semiaxes = semiradii * 2 
    return semiaxes, quat

def wrap_AAframe(AAframe: Atoms):
    """
    Returns a new CGFrame object that properly includes AA indices.
    """
    cg_indices = [[i] for i, _ in enumerate(AAframe)]
    return CGFrame(AAframe, cg_indices)

def coarsen(AAframe: Atoms, bead: CGFrame, idxs: np.ndarray, symbol:str="X", delete_indices=False, calc_geometry=False):
    """
    Given a frame and a list of indices to coarse-grain away,
    Calculate the size and orientation of the Minimum Bounding Ellipsoid
    Mutably changes the frame, so a deep copy must be made to retain the original frame.
    """
    atom_indices = []
    for i in idxs:
        atom_indices += bead.cg_indices[i]

    com = get_center_of_mass(AAframe[atom_indices])
    # mom, evecs = get_moments_of_inertia(frame[idxs], vectors=True)
    # if np.allclose(np.linalg.det(evecs), -1):
    #     evecs *= -1
    # assert(np.allclose(np.linalg.det(evecs), 1))
    # assert(np.allclose(evecs @ evecs.T, np.eye(3)))
    # quat = R.from_matrix(evecs.T).as_quat()     # IT's the ROWS! Not the columns, that are the vectors that we care about.
    # quat = np.roll(quat, 1)
    semiaxes, quat = get_quat_and_semiaxes(AAframe[atom_indices])
    if not calc_geometry:
        a = np.max(semiaxes)
        semiaxes = np.array([a, a, a])
        quat = np.array([1, 0, 0, 0])
    ell = Atom(symbol, position=com)
    # curr_cg_indices = []
    # for i in idxs:
    #     curr_cg_indices += frame[i].arrays['cg_indices']
    bead.frame.append(ell)
    bead.frame.arrays["c_diameter[1]"][-1] = semiaxes[0]
    bead.frame.arrays["c_diameter[2]"][-1] = semiaxes[1]
    bead.frame.arrays["c_diameter[3]"][-1] = semiaxes[2]
    bead.frame.arrays["c_q"][-1] = quat
    # frame.arrays["cg_indices"] = curr_cg_indices

    # Update the bead's indices
    bead.cg_indices.append(atom_indices)    
    if delete_indices:
        del bead.frame[idxs]
        bead.cg_indices = [atom_index for cg_index, atom_index in enumerate(bead.cg_indices) if cg_index not in idxs]
    
    return bead

def coarsen_frame(frame: Atoms, cg_info: CGInfo, calc_geometry=False) -> Atoms:
    """
    Given a frame, calculate the CG Bead, and return it
    """
    
    # Todo: validate to ensure that indices are valid.
    # Identify com and quat 
    com = get_center_of_mass(frame[cg_info.cg_indices])
    semiaxes, quat = get_quat_and_semiaxes(frame[cg_info.cg_indices])
    # If calc_geometry is false, just create a spherical bead large enough to
    # encompass all atoms.
    if not calc_geometry:
        a = np.max(semiaxes)
        semiaxes = np.array([a, a, a])
        quat = np.array([1, 0, 0, 0])
    
    cg_frame = Atoms(symbols=cg_info.symbol, positions=com.reshape(1, -1), cell=frame.cell, pbc=frame.pbc)
    cg_frame.arrays["c_diameter[1]"] = np.array([semiaxes[0]])
    cg_frame.arrays["c_diameter[2]"] = np.array([semiaxes[1]])
    cg_frame.arrays["c_diameter[3]"] = np.array([semiaxes[2]])
    cg_frame.arrays["c_q"] = quat.reshape(1, -1)
    cg_frame.arrays["atom_indices"] = np.asarray(cg_info.cg_indices).reshape(1, -1)
    return cg_frame

def coarsen_frame_manybeads(frame: Atoms, cg_infos: list[CGInfo], calc_geometries: list[bool])->Atoms:
    cg_frames = []
    assert len(cg_infos) == len(calc_geometries)
    for cg_info, calc_geometry in zip(cg_infos, calc_geometries):
        cg_frames.append(coarsen_frame(frame, cg_info, calc_geometry))

    # Now, turn list into a single Atoms object:
    cg_ = cg_frames[0]
    for i in range(1, len(cg_frames)):
        cg_.extend(cg_frames[i])
    return cg_

# def coarsen_manyframes_manybeads(frames: list[Atoms], cg_infos: list[CGInfo], calc_geometries: list[bool])->
def find_gre(rep1: metatensor.TensorMap, rep2: metatensor.TensorMap)->float:
    rep1_mean = metatensor.mean_over_samples(rep1, sample_names="center")
    rep2_mean = metatensor.mean_over_samples(rep2, sample_names="center")
    rep1_vec = np.hstack([block.values.squeeze() for block in rep1_mean.blocks()])
    rep2_vec = np.hstack([block.values.squeeze() for block in rep2_mean.blocks()])
    return GRE(rep1_vec, rep2_vec)

def find_rigid_indices(rep: metatensor.TensorMap) -> dict:
    center_atom = 6
    block_id = int(np.where(rep.keys['types_center'] == center_atom)[0][0])
    centers_no_hydrogen = [i for i in rep.keys["types_center"] if i != 1]
    values = dict()
    for two_neighbors in combinations_with_replacement(centers_no_hydrogen, 2):
        # two_neighbors = np.array([1, 6])
        rep.block(block_id).properties.values 
        idxs = (rep.block(block_id).properties.view(["neighbor_types_1_a", "neighbor_types_1_b"]).values == two_neighbors).all(axis=1).nonzero()[0]
        subset_with_this_triplet = rep.block(block_id).values[:, :, idxs]
        subset_tensorblock = metatensor.TensorBlock(values=subset_with_this_triplet, samples=rep.block(block_id).samples, components=rep.block(block_id).components, properties=metatensor.Labels(names=rep.block(block_id).properties.names, values=rep.block(block_id).properties.values[idxs]))
        mean_subset_tensorblock = metatensor.mean_over_samples_block(subset_tensorblock, sample_names="type")
        mean_subset_with_this_triplet = mean_subset_tensorblock.values.squeeze()
        mag_mean_subset_with_this_triplet = np.linalg.norm(mean_subset_with_this_triplet, axis=-1)
        var_mean_subset_with_this_triplet = mean_subset_with_this_triplet.var(axis=-1)

        print("Center", "Mean SOAPVEC", "Var SOAPVEC", two_neighbors)
        sorted_by_varsoap = list(sorted(zip(mean_subset_tensorblock.samples, mag_mean_subset_with_this_triplet, var_mean_subset_with_this_triplet), key=lambda x: x[2]))
        for i in sorted_by_varsoap:
            print(f"{i[0]}, {i[1]:.3e}. {i[2]:.3e}")

        values[two_neighbors] = sorted_by_varsoap[0]
    return values 

def find_rigid_indices_gen(rep: metatensor.TensorMap):
    indicies = [91, 87, 101, 100]
    for i in indicies:
        yield i

def extract_frames_from_cg_systems(cg_system_list: list[CGSystem], idx: int=0):
    frames = []
    for cg_system in cg_system_list:
        frames.append(cg_system.cg_beads[idx].frame)
    return frames

def equal_point_clouds(frame1: Atoms, frame2: Atoms, atol=8):
    """
    Two point clouds are equal if 
    1. They have the same number of atoms
    2. They have the same number of each atom type.
    3. They have the same moment of inertia, up to a tolerance.
    """
    if (len(frame1) != len(frame2)):
        return False 
    
    unique, counts = np.unique(frame1.get_chemical_symbols(), return_counts=True)
    count_dict1 = {k:v for k, v in zip(unique, counts)}
    unique, counts = np.unique(frame2.get_chemical_symbols(), return_counts=True)
    count_dict2 = {k:v for k, v in zip(unique, counts)}

    if (count_dict1 != count_dict2):
        return False 

    return np.allclose(get_moments_of_inertia(frame1), get_moments_of_inertia(frame2), atol=atol)

def frame_without_cged_atoms(cg_frame: CGFrame) -> CGFrame:
    atom_indices_to_cg = []
    assert len(cg_frame.frame) == len(cg_frame.cg_indices)
    for i, atom in enumerate(cg_frame.frame):
        if atom.symbol == 'X':
            atom_indices_to_cg += cg_frame.cg_indices[i]

    # Now, filter out these atom_indices within cg_frame.frame and cg_frame.cg_indices
    atom_indices_to_retain = [i for i in range(len(cg_frame.frame)) if i not in atom_indices_to_cg]
    _frame = copy.deepcopy(cg_frame.frame[atom_indices_to_retain])
    _cg_indices = copy.deepcopy([cg_frame.cg_indices[i] for i in atom_indices_to_retain])
    return CGFrame(_frame, _cg_indices)
    
def equal_point_clouds_dep(frame1: Atoms, frame2: Atoms, rtol=1e-8) -> bool:
    """
    THIS FUNCTION IS OVERLY SENSITIVE, 
    it's impossible to find a 3x3 matrix that transforms one set of displacements
    to another with minimal residual
    Returns true if two frames (i.e point clouds) are equal.
    Two point clouds are equal if 
    1. They have the same number of atoms
    2. They have the same number of each atom type.
    3. Ranking the atoms in terms of distance from center of mass results in same ordering of atom types.
    4. The residual between the Point Cloud 1 and Point Cloud 2 after rotation is below rtol
    """
    if (len(frame1) != len(frame2)):
        return False 
    
    unique, counts = np.unique(frame1.get_chemical_symbols(), return_counts=True)
    count_dict1 = {k:v for k, v in zip(unique, counts)}
    unique, counts = np.unique(frame2.get_chemical_symbols(), return_counts=True)
    count_dict2 = {k:v for k, v in zip(unique, counts)}

    if (count_dict1 != count_dict2):
        return False 
    
    com1 = get_center_of_mass(frame1)
    com2 = get_center_of_mass(frame2)

    displacements_from_com1 = frame1.positions - com1 
    displacements_from_com2 = frame2.positions - com2 

    distances_from_com1 = np.linalg.norm(displacements_from_com1, axis=1)
    distances_from_com2 = np.linalg.norm(displacements_from_com2, axis=1)

    idx1 = np.argsort(distances_from_com1)
    idx2 = np.argsort(distances_from_com2)

    if not (frame1.symbols[idx1] == frame2.symbols[idx2]).all():
        return False 

    rot_mat, res, _, _ = np.linalg.lstsq(displacements_from_com1[idx1], displacements_from_com2[idx2], rcond=None)
    print(f"Residuals:{res}")
    print(f"Rotation Mat:{rot_mat}")
    print(f"Frame1 displacements:{displacements_from_com1[idx1]}")
    print(f"Frame2 displacements:{displacements_from_com2[idx2]}")
    print(f"{displacements_from_com1[idx1]@rot_mat=}")
    return (res < rtol).all()
# %%
