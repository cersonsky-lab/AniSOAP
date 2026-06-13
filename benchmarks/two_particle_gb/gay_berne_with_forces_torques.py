import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from ase.io import write
from ase import Atoms
import json
import chemiscope 

# # Default Parameters and Functions

a0 = c0 = 1.0
b0 = 2.0
S0 = np.diagflat([a0, b0, c0])

sigma0 = min(a0, b0, c0)

e_a0 = sigma0 * (a0 / (b0 * c0))
e_b0 = sigma0 * (b0 / (a0 * c0))
e_c0 = sigma0 * (c0 / (b0 * a0))
e0 = np.array([e_a0, e_b0, e_c0])

A0 = np.eye(3)

L = 20
ry = 0
rz = 0
ENERGY_THRESHOLD = 0.01
FORCE_THRESHOLD = 8

# Small functions for rotating matrices
def rot_y(A, angle):
    return A @ np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


def rot_z(A, angle):
    return A @ np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# Gay-Berne Potential, as defined by Everaers and Ejtehadi
def gay_berne(
    A1, A2, S1, S2, r12, e, sigma=None, nu=1.0, mu=1.0, gamma=1.0, eps_GB=1.0
):
    if sigma is None:
        sigma = np.min([np.diag(S1).min(), np.diag(S2).min()])

    def G1(A, S):
        return A.T @ S**2.0 @ A

    def B(A, E):
        return A.T @ E @ A

    def s(S):
        a, b, c = np.diag(S)
        return (a * b + c * c) * (a * b) ** (0.5)

    G12 = G1(A1, S1) + G1(A2, S2)
    sigma_12 = (0.5 * r12.T @ np.linalg.pinv(G12) @ r12) ** (-0.5)
    h12 = np.linalg.norm(r12) - sigma_12

    rho = sigma / (h12 + gamma * sigma)
    Ur = 4 * eps_GB * ((rho) ** 12.0 - rho**6.0)

    E = np.diagflat(e ** (-1 / mu))
    B12 = B(A1, E) + B(A2, E)
    chi_12 = (2 * r12.T @ np.linalg.pinv(B12) @ r12) ** mu

    s1 = s(S1)
    s2 = s(S2)
    eta_12 = ((2 * s1 * s2) / np.linalg.det(G12)) ** (nu / 2.0)
    return Ur * eta_12 * chi_12


# Numerical forces and torques from the existing Gay-Berne potential.
# This intentionally leaves gay_berne(), the quaternion conventions, and frame construction unchanged.
def gay_berne_force_torque(
    A1,
    A2,
    S1,
    S2,
    r12,
    e,
    sigma=None,
    nu=1.0,
    mu=1.0,
    gamma=1.0,
    eps_GB=1.0,
    dr=1.0e-5,
    dtheta=1.0e-1,
):
    """Return energy, forces, and torques for a two-particle Gay-Berne configuration.

    Forces are computed as F = -grad_x U using central finite differences in r12.
    Since r12 = r2 - r1, the returned force on particle 2 is -dU/dr12 and the
    force on particle 1 is equal and opposite.

    Torques are computed as tau = -dU/dtheta for small body rotations about the
    lab-frame x/y/z axes.  The torque sign convention matches the force convention:
    positive torque lowers energy for a positive infinitesimal rotation.
    """

    r12 = np.asarray(r12, dtype=float)

    def energy(A1_eval, A2_eval, r_eval):
        return gay_berne(
            A1_eval,
            A2_eval,
            S1,
            S2,
            r_eval,
            e,
            sigma=sigma,
            nu=nu,
            mu=mu,
            gamma=gamma,
            eps_GB=eps_GB,
        )

    U = energy(A1, A2, r12)

    dU_dr = np.zeros(3)
    for i in range(3):
        delta = np.zeros(3)
        delta[i] = dr
        dU_dr[i] = (energy(A1, A2, r12 + delta) - energy(A1, A2, r12 - delta)) / (2 * dr)

    forces = np.zeros((2, 3))
    forces[1] = -dU_dr
    forces[0] = -forces[1]

    torques = np.zeros((2, 3))
    axes = np.eye(3)
    for i, axis in enumerate(axes):
        dR_plus = R.from_rotvec(dtheta * axis).as_matrix()
        dR_minus = R.from_rotvec(-dtheta * axis).as_matrix()

        dU_dtheta_1 = (energy(dR_plus @ A1, A2, r12) - energy(dR_minus @ A1, A2, r12)) / (2 * dtheta)
        dU_dtheta_2 = (energy(A1, dR_plus @ A2, r12) - energy(A1, dR_minus @ A2, r12)) / (2 * dtheta)

        torques[0, i] = -dU_dtheta_1
        torques[1, i] = -dU_dtheta_2

    return U, forces, torques


def add_energy_forces_torques(frame, A1, A2, S1, S2, r12, e, sigma=None, **kwargs):
    """Attach energy, forces, and torques to an ASE frame."""

    energy, forces, torques = gay_berne_force_torque(
        A1, A2, S1, S2, r12, e, sigma=sigma, **kwargs
    )
    frame.info["energy"] = energy
    frame.arrays["forces"] = forces
    frame.arrays["torques"] = torques
    return frame


# Silly function to make a print-out when writing
def verbose_write(filename, frames):
    print("Writing {} frames to {}.".format(len(frames), filename))
    write(filename, frames)
    accumulate_frames(frames)


def accumulate_frames(frames):
    """Add these frames to the combined Chemiscope trajectory."""

    all_frames.extend(frames)

# Classic Gay-Berne Plots to show the Class Side-to-Side, Face-to-Face,
# and Side-to-Face

rs = np.linspace(1.25 * sigma0, 3 * sigma0, 100)
side_side = np.array([gay_berne(A0, A0, S0, S0, np.array([r, 0, 0]), e0) for r in rs])
face_face = np.array(
    [
        gay_berne(
            rot_z(A0, np.pi / 2), rot_z(A0, np.pi / 2), S0, S0, np.array([r, 0, 0]), e0
        )
        for r in rs
    ]
)
side_face = np.array(
    [gay_berne(A0, rot_z(A0, np.pi / 2), S0, S0, np.array([r, 0, 0]), e0) for r in rs]
)

plt.plot(
    rs,
    side_side,
    label="Side-to-Side",
)
plt.plot(
    rs,
    face_face,
    label="Face-to-Face",
)
plt.plot(
    rs,
    side_face,
    label="Side-to-Face",
)
plt.legend()

plt.gca().set_ylabel("U [Energy Units]")
plt.gca().set_xlabel("r [Distance Units]")
plt.gca().set_ylim([4 * min(-e0), 1])
plt.show()

# Minimum distance to use for each of these
r0_ss = rs[np.where(side_side <= 0)[0][0]]
r0_ff = rs[np.where(face_face <= 0)[0][0]]
r0_fs = rs[np.where(side_face <= 0)[0][0]]


# Accumulate every generated frame for a single Chemiscope dataset.
all_frames = []
all_shapes = []

# # Generate Example Sets

# Shared frame-building helpers used by the examples below.  These only remove
# repeated setup code; the energies, forces, torques, quaternions, frame info,
# filtering rules, and output filenames are kept the same as above.
def make_two_particle_frame(rx, ry, rz):
    return Atoms(
        cell=L * np.ones(3),
        positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
        numbers=np.zeros(2),
    )

def attach_shape_arrays(frame):
    frame.arrays["c_diameter\[1\]"] = a0 * np.ones(len(frame))
    frame.arrays["c_diameter\[2\]"] = b0 * np.ones(len(frame))
    frame.arrays["c_diameter\[3\]"] = c0 * np.ones(len(frame))
    return frame

def quaternions_from_matrices(A1, A2):
    raw_quaternions = np.array(
        [
            R.from_matrix(A1).as_quat(),
            R.from_matrix(A2).as_quat(),
        ]
    )
    quaternions = np.zeros(raw_quaternions.shape)
    quaternions[:, 0] = raw_quaternions[:, -1]
    quaternions[:, 1:] = raw_quaternions[:, :-1]
    return quaternions

def _as_float_list(values):
    """Convert numpy/scalar values to JSON-safe Python floats."""

    return np.asarray(values, dtype=float).tolist()


def _build_chemiscope_shapes(frames, force_scale=0.25, torque_scale=1):
    """Create Chemiscope shape groups for ellipsoids plus force/torque arrows.

    The ellipsoid semiaxes and orientations are atom-level shape entries, which is
    the Chemiscope format for per-particle shape information. Force and torque
    arrows are included as additional atom-level shape groups.
    """

    ellipsoid_atoms = []
    force_atoms = []
    torquex_atoms = []
    torquey_atoms = []
    torquez_atoms = []

    for frame in frames:
        raw_quaternions = np.asarray(frame.arrays["quaternions"], dtype=float)
        quaternions = np.zeros(raw_quaternions.shape)
        quaternions[:, -1] = raw_quaternions[:, 0]
        quaternions[:, :-1] = raw_quaternions[:, 1:]
        semiaxes = np.column_stack(
            [
                frame.arrays["c_diameter\[1\]"] / 2.0,
                frame.arrays["c_diameter\[2\]"] / 2.0,
                frame.arrays["c_diameter\[3\]"] / 2.0,
            ]
        )

        for atom_i in range(len(frame)):
            ellipsoid_atoms.append(
                {
                    "semiaxes": _as_float_list(semiaxes[atom_i]),
                    "orientation": _as_float_list(quaternions[atom_i]),
                }
            )
            force_atoms.append({"vector": _as_float_list(force_scale * frame.arrays["forces"][atom_i])})
            torquex_atoms.append({"vector": _as_float_list(torque_scale * np.array([frame.arrays["torques"][atom_i][0], 0, 0]))})
            torquey_atoms.append({"vector": _as_float_list(torque_scale * np.array([0, frame.arrays["torques"][atom_i][1], 0]))})
            torquez_atoms.append({"vector": _as_float_list(torque_scale * np.array([0, 0, frame.arrays["torques"][atom_i][2]]))})

    return {
        "ellipsoids": {
            "kind": "ellipsoid",
            "parameters": {
                "global": {"color": "lightgray"},
                "atom": ellipsoid_atoms,
            },
        },
        "forces": {
            "kind": "arrow",
            "parameters": {
                "global": {"baseRadius": 0.03, "headRadius": 0.08, "headLength": 0.12},
                "atom": force_atoms,
            },
        },
        "torques_x": {
            "kind": "arrow",
            "parameters": {
                "global": {"baseRadius": 0.03, "headRadius": 0.08, "headLength": 0.12, "color": "red"},
                "atom": torquex_atoms,
            },
        },
        "torques_y": {
            "kind": "arrow",
            "parameters": {
                "global": {"baseRadius": 0.03, "headRadius": 0.08, "headLength": 0.12, "color": "red"},
                "atom": torquey_atoms,
            },
        },
        "torques_z": {
            "kind": "arrow",
            "parameters": {
                "global": {"baseRadius": 0.03, "headRadius": 0.08, "headLength": 0.12, "color": "red"},
                "atom": torquez_atoms,
            },
        },
    }


def build_frame(rx, ry, rz, A1, A2, quaternions=None, separation_distance=None, angles=None):
    frame = make_two_particle_frame(rx, ry, rz)

    if quaternions is None:
        quaternions = quaternions_from_matrices(A1, A2)

    frame.arrays["quaternions"] = quaternions
    attach_shape_arrays(frame)

    if separation_distance is not None:
        frame.info["separation_distance"] = separation_distance
    else:
        frame.info['separation_distance'] = frame.get_all_distances(mic=True)[0, 1]
    if angles is not None:
        frame.arrays["angles"] = np.array(angles)

    add_energy_forces_torques(frame, A1, A2, S0, S0, np.array([rx, ry, rz]), e0, sigma0)
    if frame.info['energy'] < ENERGY_THRESHOLD and np.linalg.norm(frame.arrays['forces'], axis=1).max() < FORCE_THRESHOLD:
        return frame


def write_distance_scan(filename, r_start, r_stop, A1, A2):
    frames = []
    for rx in np.linspace(r_start, r_stop, 100):
        frame = build_frame(
                rx,
                ry,
                rz,
                A1,
                A2,
                separation_distance=rx,
            )
        if frame is not None:
            frames.append(frame)
    verbose_write(filename, frames)


def write_single_rotating_scan(filename, rotation_function, include_separation_and_angles):
    frames = []
    for rx in np.linspace(r0_ss, 3 * r0_ss, 10):
        for angle in np.linspace(0, np.pi, 10):
            A2 = rotation_function(A0, angle)
            frame = build_frame(
                rx,
                ry,
                rz,
                A0,
                A2,
                separation_distance=rx if include_separation_and_angles else None,
                angles=[0, angle] if include_separation_and_angles else None,
            )
            if frame is not None:
                frames.append(frame)
    verbose_write(filename, frames)


# ## Generate Side-to-Side Frames
write_distance_scan("side_to_side.xyz", r0_ss, 2 * r0_ss, A0, A0)


# ## Generate Face-to-Face Frames
write_distance_scan(
    "face_to_face.xyz",
    r0_ff,
    1.5 * r0_ff,
    rot_z(A0, np.pi / 2),
    rot_z(A0, np.pi / 2),
)


write_distance_scan(
    "side_to_face.xyz",
    r0_fs,
    1.5 * r0_fs,
    A0,
    rot_z(A0, np.pi / 2),
)


# ## Single Rotating Neighbor
write_single_rotating_scan(
    "single_rotating_in_z.xyz",
    rot_z,
    include_separation_and_angles=True,
)

write_single_rotating_scan(
    "single_rotating_in_y.xyz",
    rot_y,
    include_separation_and_angles=True,
)


# ## Both Rotating
frames = []
for rx in np.linspace(sigma0, 2.4, 10):
    for angle1 in np.linspace(0, np.pi, 10):
        for angle2 in np.linspace(0, np.pi, 10):
            A1 = rot_z(A0, angle1)
            A2 = rot_z(A0, angle2)
            frame = build_frame(
                rx,
                ry,
                rz,
                A1,
                A2,
                separation_distance=rx,
                angles=[angle1, angle2],
            )
            if frame is not None:
                frames.append(frame)

verbose_write("both_rotating_in_z.xyz", frames)


# ## Random Rotations and Distances
frames = []
for _ in range(1000):
    rx, ry_random, rz_random = np.random.uniform(sigma0, 2 * sigma0, size=3)

    A1 = R.random().as_matrix()
    A2 = R.random().as_matrix()
    frame = build_frame(rx, ry_random, rz_random, A1, A2)
    if frame is not None:
        frames.append(frame)

verbose_write("random_rotations.xyz", frames)

# ## Combined Chemiscope Dataset
chemiscope.write_input("gay_berne_all_frames.chemiscope.json", 
                        all_frames,
                        properties=chemiscope.extract_properties(
                            all_frames, only=["torques", "forces"],
                            environments = chemiscope.all_atomic_environments(structures=all_frames, cutoff=3.5)
                        ),
                        shapes = _build_chemiscope_shapes(all_frames),
                        environments = chemiscope.all_atomic_environments(structures=all_frames, cutoff=3.5),
                        settings= {
                            "target": "atom",
                            "map": {
                                "x": {"property": "torques[1]", "scale": "linear"},
                                "y": {"property": "torques[3]", "scale": "linear"},
                            },
                            "structure": [
                                {
                                    "atoms": False,
                                    "bonds": False,
                                    "shape": "ellipsoids,forces,torques_x,torques_y,torques_z",
                                }
                            ],
                        },
                        )

verbose_write("all_frames.xyz", all_frames)