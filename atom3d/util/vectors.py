"""Utility functions for unit vectors."""
import math

import numpy as np
import pyrr as pr


def fibonacci(num_points):
    """
    Create an approximately uniform grid on the unit sphere.

    Adapted from
    http://matplotlib.1069221.n5.nabble.com/
    Generating-N-regularly-spaced-points-on-a-sphere-td20763.html

    Args:
        num_points (int):
            the number of grid points to place.
    Returns:
        num_points x 3 numpy array where each row is x, y, and z coordinates of
        each unit vector.
    """
    inc = np.pi * (3 - np.sqrt(5))
    off = 2. / num_points
    k = np.arange(0, num_points)
    y = k * off - 1. + 0.5 * off
    r = np.sqrt(1 - y * y)
    phi = k * inc
    x = np.cos(phi) * r
    z = np.sin(phi) * r

    unit_vectors = np.zeros((num_points, 3), dtype='f4')
    unit_vectors[:, 0] = x
    unit_vectors[:, 1] = y
    unit_vectors[:, 2] = z
    return unit_vectors


def get_quaternion_alignment(uv1, up1, uv2, up2):
    """
    Return quaternion that aligns 2 pairs of direction vectors and up vectors.

    We align the (uv1, up1) to (uv2, up2).  The rotation is unique, assuming
    direction vectors and up vectors are not the same.  All vectors are assumed
    to be of unit length.  Note however that for any given rotation, two
    quaternions can represent it.

    We want:

        q * uv1 = uv2
        q * up1 = up2

    Args:
        uv1 (np.array of float):
            3 entry array specifying direction we are aligning from.
        up1 (np.array of float):
            3 entry array specifying up vector we are aligning from.
        uv2 (np.array of float):
            3 entry array specifying direction we are aligning to.
        up2 (np.array of float):
            3 entry array specifying up vector we are aligning to.

    Returns:
        pr.Quaternion specifying rotation.
    """
    uv1 = np.array(uv1)
    uv2 = np.array(uv2)
    up1 = np.array(up1)
    up2 = np.array(up2)
    # First we get quaternion to rotate uv1 to uv2.  Code from
    # http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    if np.all(np.isclose(uv1, uv2, atol=1e-5)):
        qt = pr.Quaternion()
    elif np.all(np.isclose(-uv1, uv2, atol=1e-5)):
        qt = pr.Quaternion.from_axis_rotation(up1, math.pi)
        qt.normalise()
    else:
        w = np.cross(uv1, uv2)
        qt = np.insert(w, 3, [1.0 + np.dot(uv1, uv2)])
        qt = pr.Quaternion(qt)
        qt.normalise()

    # Then, we get the quaternion to rotate the transformed up1 to up2.
    up1t = qt * pr.Vector3(up1)
    if np.all(np.isclose(up1t, up2, atol=1e-5)):
        qr = pr.Quaternion()
    elif np.all(np.isclose(-up1t, up2, atol=1e-5)):
        qr = pr.Quaternion.from_axis_rotation(uv2, math.pi)
        qr.normalise()
    else:
        w = np.cross(up1t, up2)
        qr = np.insert(w, 3, [1.0 + np.dot(up1t, up2)])
        qr = pr.Quaternion(qr)
        qr.normalise()

    # Finally, we put the two rotations together to get the final rotation.
    q = qr * qt
    return q


def rotate_v_by_q(v, q):
    """
    Rotate vector by quaternion.

    Args:
        v (np.array of float):
            3 entry array specifying vector we are rotating.
        q (pr.Quaternion):
            quaternion representing rotation we are applying.

    Returns:
        np.array of 3 floats representing rotated v.
    """
    if np.all(np.isclose(v, [0, 0, 0])):
        return v
    mag_v = np.linalg.norm(v)
    v_unit = v / mag_v
    tv_unit = q * pr.Vector3(v_unit)
    tv = tv_unit * mag_v
    return np.array(tv)


def generate_up_vectors(uv, num_rolls, up_seed=None, rev=False):
    """
    Generate evenly distributed up vectors for the suppllied direction vector.

    Args:
        uv (np.array of float):
            3 entry array specifying direction we are getting up vectors for.
        num_rolls (int):
            number of different up vectors to create.

    Keyword Args:
        up_seed (np.array of float):
            3 entry array specifying starting up direction.  If unspecified will
            be generated automatically.  Used to ensure that up vectors for two
            matching surfacelets start pointing in the same direction.
        rev (boolean):
            whether to proceed counter-clockwise as opposed to clockwise for the
            generation of the up vectors.

    Returns:
        num_rolls x 3 np.array of floats, representing the num_rolls different
        up vectors.
    """
    if up_seed is None:
        if uv[0] == 0 and uv[1] == 0:
            c = [0, -1, 0]
        else:
            c = [0, 0, -1]
        up_seed = np.cross(uv, c)
        up_seed = up_seed / np.linalg.norm(up_seed)

    if not rev:
        q = pr.Quaternion.from_axis_rotation(uv, 2 * math.pi / num_rolls)
    else:
        q = pr.Quaternion.from_axis_rotation(uv, -2 * math.pi / num_rolls)
    up = np.zeros((num_rolls, 3), dtype='f4')

    up[0] = up_seed
    for i in range(1, num_rolls):
        up[i] = rotate_v_by_q(up[i - 1], q)
        up[i] = up[i] / np.linalg.norm(up[i])
    return up


def generate_all_up_vectors(uvs, num_rolls):
    """Generate all up vectors for provided unit vectors."""
    num_uvs = uvs.shape[0]
    ups = np.zeros((num_uvs, num_rolls, 3))
    for i, uv in enumerate(uvs):
        ups[i] = generate_up_vectors(uv, num_rolls)
    return ups


def get_all_rot_mats(uvs, ups):
    """
    Generate rotation matrices for multiple unit vectors.

    TODO: Add args.
    """
    num_uvs = uvs.shape[0]
    num_ups = ups.shape[1]

    rot_mats = np.zeros((num_uvs, num_ups, 3, 3))
    for i in range(num_uvs):
        rot_mats[i] = get_rot_mats(uvs[i], ups[i])

    return rot_mats


def get_rot_mats(uv, ups):
    """
    Generate rotation matrices to transform base coordinate system to provided.

    TODO: Add docs.
    """
    num_ups = ups.shape[0]

    rot_mats = np.zeros((num_ups, 3, 3))
    for i in range(num_ups):
        rot_mats[i, :, :] = get_rot_mat(uv, ups[i])

    return rot_mats


def get_rot_mat(uv, up):
    """
    Generate rotation matrix to transforms provided coordinate system to base.

    R * uv = uv_b
    R * up = up_b
    """
    base_uv = [1., 0., 0.]
    base_up = [0., 1., 0.]

    rot_mat = np.zeros((3, 3))
    q = get_quaternion_alignment(uv, up, base_uv, base_up)
    rot_mat = np.array(pr.Matrix33(q))
    return rot_mat
