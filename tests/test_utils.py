from re import I
import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import the different versions of the moment generators
from anisoap.utils import quaternion_to_rotation_matrix

class TestQuaternionToRotationMatrix:
    """
    Test that the conversion of rotational orientation from the
    quaternions to rotation matrices behaves correctly.
    """
    # Define some desired rotation angles
    angles = np.linspace(-3*np.pi,3*np.pi,47)

    # Compare the obtained rotation matrices for the special
    # case of rotations around the x-axis.
    @pytest.mark.parametrize('angle', angles)
    def test_exact_x_rotation(self, angle):
        rot_exact = np.zeros((3,3))
        rot_exact[1,1] = rot_exact[2,2] = np.cos(angle)
        rot_exact[2,1] = np.sin(angle)
        rot_exact[1,2] = -np.sin(angle)
        rot_exact[0,0] = 1.
        
        quat = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        rot = quaternion_to_rotation_matrix(quat)
        
        assert_allclose(rot, rot_exact, rtol=1e-15, atol=1e-15)

    # Compare the obtained rotation matrices for the special
    # case of rotations around the y-axis.
    @pytest.mark.parametrize('angle', angles)
    def test_exact_y_rotation(self, angle):
        rot_exact = np.zeros((3,3))
        rot_exact[0,0] = rot_exact[2,2] = np.cos(angle)
        rot_exact[2,0] = -np.sin(angle)
        rot_exact[0,2] = np.sin(angle)
        rot_exact[1,1] = 1.
        
        quat = np.array([np.cos(angle/2), 0, np.sin(angle/2), 0])
        rot = quaternion_to_rotation_matrix(quat)
        
        assert_allclose(rot, rot_exact, rtol=1e-15, atol=1e-15)

    # Compare the obtained rotation matrices for the special
    # case of rotations around the z-axis.
    @pytest.mark.parametrize('angle', angles)
    def test_exact_z_rotation(self, angle):
        rot_exact = np.zeros((3,3))
        rot_exact[0,0] = rot_exact[1,1] = np.cos(angle)
        rot_exact[1,0] = np.sin(angle)
        rot_exact[0,1] = -np.sin(angle)
        rot_exact[2,2] = 1.
        
        quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        rot = quaternion_to_rotation_matrix(quat)
        
        assert_allclose(rot, rot_exact, rtol=1e-15, atol=1e-15)

    # Generate a random set of unit quaternions.
    np.random.seed(2348120)
    quaternions = np.random.normal(size=(4, 47))
    quaternions /= np.linalg.norm(quaternions, axis=0)
    # Make sure that the obtained matrices are indeed orthogonal
    @pytest.mark.parametrize('quaternion', quaternions.T)
    def test_orthogonality(self, quaternion):
        rot = quaternion_to_rotation_matrix(quaternion)
        Id = np.eye(3)
        assert_allclose(rot.T @ rot, Id, rtol=3e-15, atol=1e-15)

    # The general form of a unit quaternion that corresponds to a rotation
    # by angle theta around an axis u (which is a unit vector) is:
    # q = (cos(theta/2), sin(theta/2)u)
    # Thus, if we apply the rotation matrix to the vector u,
    # the coordinate frame should be invariant. 
    @pytest.mark.parametrize('quaternion', quaternions.T)
    def test_invariant_axis(self, quaternion):
        cosine = quaternion[0]
        axis = quaternion[1:] / np.sqrt(1 - cosine**2)
        rot = quaternion_to_rotation_matrix(quaternion)
        axis_rotated = rot @ axis

        # If the rotation matrix is indeed a rotation around
        # "axis", then applying the matrix should not change
        # the vector.
        axis_rotated = rot @ axis
        assert_allclose(axis, axis_rotated, rtol=3e-14, atol=1e-15)