import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from ase.io import write
from ase import Atoms


# # Default Parameters and Functions

a0 = c0 = 1.0
b0 = 1.5
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


# Silly function to make a print-out when writing
def verbose_write(filename, frames):
    print("Writing {} frames to {}.".format(len(frames), filename))
    write(filename, frames)


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


# # Generate Example Sets

# ## Generate Side-to-Side Frames
frames = []
for rx in np.linspace(r0_ss, 2 * r0_ss, 100):
    frame = Atoms(
        cell=L * np.ones(3),
        positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
        numbers=np.zeros(2),
    )

    quaternions = np.zeros((len(frame), 4))
    quaternions[:, 0] = 1

    frame.arrays["quaternions"] = quaternions
    frame.arrays["c_diameter[1]"] = a0 * np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = b0 * np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = c0 * np.ones(len(frame))

    frame.info["separation_distance"] = rx
    frame.info["energy"] = gay_berne(A0, A0, S0, S0, np.array([rx, ry, rz]), e0, sigma0)
    frames.append(frame)


verbose_write("side_to_side.xyz", frames)


# ## Generate Face-to-Face Frames
frames = []
for rx in np.linspace(r0_ff, 2 * r0_ff, 100):
    frame = Atoms(
        cell=L * np.ones(3),
        positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
        numbers=np.zeros(2),
    )

    quaternions = np.zeros((len(frame), 4))
    quaternions[:, 2] = np.sqrt(0.5)
    quaternions[:, 3] = np.sqrt(0.5)

    frame.arrays["quaternions"] = quaternions
    frame.arrays["c_diameter[1]"] = a0 * np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = b0 * np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = c0 * np.ones(len(frame))

    frame.info["separation_distance"] = rx
    frame.info["energy"] = gay_berne(
        rot_z(A0, np.pi / 2),
        rot_z(A0, np.pi / 2),
        S0,
        S0,
        np.array([rx, ry, rz]),
        e0,
        sigma0,
    )
    frames.append(frame)


verbose_write("face_to_face.xyz", frames)


# ## Generate Side-to-Face Frames
frames = []
for rx in np.linspace(r0_fs, 2 * r0_fs, 100):
    frame = Atoms(
        cell=L * np.ones(3),
        positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
        numbers=np.zeros(2),
    )

    quaternions = np.zeros((len(frame), 4))
    quaternions[0, 0] = 1
    quaternions[1, 2] = np.sqrt(0.5)
    quaternions[1, 3] = np.sqrt(0.5)

    frame.arrays["quaternions"] = quaternions
    frame.arrays["c_diameter[1]"] = a0 * np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = b0 * np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = c0 * np.ones(len(frame))

    frame.info["separation_distance"] = rx
    frame.info["energy"] = gay_berne(
        A0,
        rot_z(A0, np.pi / 2),
        S0,
        S0,
        np.array([rx, ry, rz]),
        e0,
        sigma0,
    )
    frames.append(frame)


verbose_write("side_to_face.xyz", frames)


# ## Single Rotating Neighbor
frames = []
for rx in np.linspace(r0_ss, 3 * r0_ss, 10):
    for angle in np.linspace(0, np.pi, 10):
        frame = Atoms(
            cell=L * np.ones(3),
            positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
            numbers=np.zeros(2),
        )

        A2 = rot_z(A0, angle)
        raw_quaternions = np.array(
            [
                R.from_matrix(A0).as_quat(),
                R.from_matrix(A2).as_quat(),
            ]
        )
        quaternions = np.zeros((len(frame), 4))
        quaternions[:, 0] = raw_quaternions[:, -1]
        quaternions[:, 1:] = raw_quaternions[:, :-1]

        frame.arrays["quaternions"] = quaternions
        frame.arrays["c_diameter[1]"] = a0 * np.ones(len(frame))
        frame.arrays["c_diameter[2]"] = b0 * np.ones(len(frame))
        frame.arrays["c_diameter[3]"] = c0 * np.ones(len(frame))
        frame.info["separation_distance"] = rx
        frame.arrays["angles"] = np.array([0, angle])

        frame.info["energy"] = gay_berne(
            A0, A2, S0, S0, np.array([rx, ry, rz]), e0, sigma0
        )
        if frame.info["energy"] < 0.1:
            frames.append(frame)

verbose_write("single_rotating_in_z.xyz", frames)

frames = []
for rx in np.linspace(r0_ss, 3 * r0_ss, 10):
    for angle in np.linspace(0, np.pi, 10):
        frame = Atoms(
            cell=L * np.ones(3),
            positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
            numbers=np.zeros(2),
        )

        A2 = rot_y(A0, angle)
        raw_quaternions = np.array(
            [
                R.from_matrix(A0).as_quat(),
                R.from_matrix(A2).as_quat(),
            ]
        )
        quaternions = np.zeros(raw_quaternions.shape)
        quaternions[:, 0] = raw_quaternions[:, -1]
        quaternions[:, 1:] = raw_quaternions[:, :-1]

        frame.arrays["quaternions"] = quaternions
        frame.arrays["c_diameter[1]"] = a0 * np.ones(len(frame))
        frame.arrays["c_diameter[2]"] = b0 * np.ones(len(frame))
        frame.arrays["c_diameter[3]"] = c0 * np.ones(len(frame))

        frame.info["energy"] = gay_berne(
            A0, A2, S0, S0, np.array([rx, ry, rz]), e0, sigma0
        )
        if frame.info["energy"] < 0.1:
            frames.append(frame)

verbose_write("single_rotating_in_y.xyz", frames)


# ## Both Rotating
frames = []
for rx in np.linspace(sigma0, 2.4, 10):
    for angle1 in np.linspace(0, np.pi, 10):
        for angle2 in np.linspace(0, np.pi, 10):
            frame = Atoms(
                cell=L * np.ones(3),
                positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
                numbers=np.zeros(2),
            )

            A1 = rot_z(A0, angle1)
            A2 = rot_z(A0, angle2)
            raw_quaternions = np.array(
                [
                    R.from_matrix(A1).as_quat(),
                    R.from_matrix(A2).as_quat(),
                ]
            )
            quaternions = np.zeros(raw_quaternions.shape)
            quaternions[:, 0] = raw_quaternions[:, -1]
            quaternions[:, 1:] = raw_quaternions[:, :-1]
            frame.arrays["quaternions"] = quaternions
            frame.arrays["c_diameter[1]"] = a0 * np.ones(len(frame))
            frame.arrays["c_diameter[2]"] = b0 * np.ones(len(frame))
            frame.arrays["c_diameter[3]"] = c0 * np.ones(len(frame))
            frame.info["separation_distance"] = rx
            frame.arrays["angles"] = np.array([angle1, angle2])

            frame.info["energy"] = gay_berne(
                A1, A2, S0, S0, np.array([rx, ry, rz]), e0, sigma0
            )
            if frame.info["energy"] < 0.1:
                frames.append(frame)

verbose_write("both_rotating_in_z.xyz", frames)

# ## Random Rotations and Distances
frames = []
for _ in range(1000):
    rx, ry, rz = np.random.uniform(sigma0, 2 * sigma0, size=3)
    frame = Atoms(
        cell=L * np.ones(3),
        positions=[[L / 2, L / 2, L / 2], [L / 2 + rx, L / 2 + ry, L / 2 + rz]],
        numbers=np.zeros(2),
    )

    A1 = R.random().as_matrix()
    A2 = R.random().as_matrix()
    raw_quaternions = np.array(
        [
            R.from_matrix(A1).as_quat(),
            R.from_matrix(A2).as_quat(),
        ]
    )
    quaternions = np.zeros(raw_quaternions.shape)
    quaternions[:, 0] = raw_quaternions[:, -1]
    quaternions[:, 1:] = raw_quaternions[:, :-1]
    frame.arrays["quaternions"] = quaternions
    frame.arrays["c_diameter[1]"] = a0 * np.ones(len(frame))
    frame.arrays["c_diameter[2]"] = b0 * np.ones(len(frame))
    frame.arrays["c_diameter[3]"] = c0 * np.ones(len(frame))

    frame.info["energy"] = gay_berne(A1, A2, S0, S0, np.array([rx, ry, rz]), e0, sigma0)
    if frame.info["energy"] < 0.1:
        frames.append(frame)

verbose_write("random_rotations.xyz", frames)
