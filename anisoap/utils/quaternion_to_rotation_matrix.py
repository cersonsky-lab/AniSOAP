import numpy as np


# Function to transform the quaternion representation of a rotation
# to its matrix form.
# The formula for the rotation matrix in terms of the four components
# of the quaternion can be obtained by simply applying a unit
# quaternion to an arbitrary vector and adding together the linear
# terms, see e.g.
# https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
def quaternion_to_rotation_matrix(q):
    assert abs(1 - np.linalg.norm(q)) < 1e-6
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    rot = np.zeros((3, 3))
    rot[0, 0] = 1 - 2 * (qy**2 + qz**2)
    rot[1, 1] = 1 - 2 * (qx**2 + qz**2)
    rot[2, 2] = 1 - 2 * (qx**2 + qy**2)
    for n in range(3):
        for m in range(3):
            if n != m:
                rot[n, m] += 2 * q[n + 1] * q[m + 1]

    rot[0, 1] -= 2 * qw * qz
    rot[1, 0] += 2 * qw * qz
    rot[2, 0] -= 2 * qw * qy
    rot[0, 2] += 2 * qw * qy
    rot[1, 2] -= 2 * qw * qx
    rot[2, 1] += 2 * qw * qx

    return rot
