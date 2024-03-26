from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


def krios_to_head_coordinate(
    elc: NDArray[np.float64],
    rpa: NDArray[np.float64],
    lpa: NDArray[np.float64],
    nz: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Transform Krios coordinates to head coordinates.

    Parameters
    ----------
    elc : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the EEG electrodes.
    rpa : array of shape (3,)
        (x, y, z) coordinates of the right pre-auricular point.
    lpa : array of shape (3,)
        (x, y, z) coordinates of the left pre-auricular point.
    nz : array of shape (3,)
        (x, y, z) coordinates of the nasion.

    Returns
    -------
    elc : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the EEG electrodes in the head coordinate system.
    fid : array of shape (3, 3)
        (x, y, z) coordinates of the fiducials in the head coordinate system, ordered as
        RPA, LPA, NZ.
    T : array of shape (3, 3)
        Transformation matrix from the Krios to the head coordinate system.
    """
    for var in (elc, rpa, lpa, nz):  # sanity-checks
        assert var.dtype == np.float64
        assert var.shape[-1] == 3
    assert elc.ndim == 2
    assert rpa.ndim == 1
    assert lpa.ndim == 1
    assert nz.ndim == 1
    # define coordinate system
    x = rpa - lpa  # left to right axis
    xy = nz - lpa  # vector on the xy plan
    z = np.cross(x, xy)  # bottom to top axis = normal to xy plan
    y = np.cross(z, x)  # back to front axis = normal to zx plan
    x = x / np.linalg.norm(x, 2)
    y = y / np.linalg.norm(y, 2)
    z = z / np.linalg.norm(z, 2)
    # create transformation matrix from Krios to head coordinate system
    T = np.zeros(shape=(3, 3), dtype=np.float64)
    T[0, :] = x
    T[1, :] = y
    T[2, :] = z
    # apply transformation to electrodes and fiducials
    elc = np.matmul(T, elc.T).T
    fid = np.matmul(T, np.array([rpa, lpa, nz], dtype=np.float64).T).T
    # change origin
    origin_shift = [fid[2, 0], fid[0, 1], fid[2, 2]]
    elc -= np.tile(origin_shift, (elc.shape[0], 1))
    fid -= np.tile(origin_shift, (fid.shape[0], 1))
    # sanity-checks
    assert np.allclose(fid[0, 1:], np.zeros(2))  # RPA
    assert np.allclose(fid[1, 1:], np.zeros(2))  # LPA
    assert np.allclose(fid[2, np.array([0, 2], dtype=np.int8)], np.zeros(2))  # NZ
    return elc, fid, T


def reorder_electrodes(
    elc: NDArray[np.float64], template: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Re-order electrodes according to the template.

    Parameters
    ----------
    elc : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the EEG electrodes.
    template : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the EEG electrodes in the template.

    Returns
    -------
    elc_reordered : array of shape (n_electrodes, 3)
        Reordered electrode locations matching the template order.
    """
    elc_reordered = np.zeros(template.shape, dtype=np.float64)
    cost_matrix = distance_matrix(template, elc, p=2)
    r_idx, c_idx = linear_sum_assignment(cost_matrix)
    elc_reordered[r_idx, :] = elc[c_idx, :]
    # check if there is less scanned electrodes than template electrode, in which case
    # add electrode location from the template directly
    if r_idx.size < template.shape[0]:
        z = np.in1d(np.arange(template.shape[0]), r_idx, assume_unique=False)
        idx = np.nonzero(~z)[0]
        elc_reordered[idx, :] = template[idx, :]
        warn(
            "Some electrodes are missing in the scanned data and have been filled from "
            f"the template. Missing electrodes idx: {idx}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return elc_reordered


def fit_matched_points(
    p: NDArray[np.float64],
    x: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    scale: bool = False,
) -> tuple[NDArray[np.float64], float]:
    """Fit matched points using an analytical formula.

    Notes
    -----
    Follow notation of P.J. Besl and N.D. McKay, A Method for Registration of 3-D
    Shapes, IEEE Trans. Patt. Anal. Machine Intell., 14, 239 - 255, 1992.

    The original method is actually by Horn, Closed-form solution of absolute
    orientation using unit quaternions, J Opt. Soc. Amer. A vol 4 no 4 pp 629-642, Apr.
    1987. This paper describes how weights can be easily incorporated, and a uniform
    scale factor can be computed.

    Caution: This can be dangerous if there are 3 points, or 4 points in a symmetric
             layout, as the geometry can be explained equivalently under 180 degree
             rotations.
    """
    assert p.shape == x.shape
    assert p.ndim == 2
    assert p.shape[1] == 3
    # (weighted) centroids
    weights_ = np.full((p.shape[0], 1), 1.0 / max(p.shape[0], 1))
    if weights is not None:
        weights_[:] = np.reshape(weights / weights.sum(), (weights.size, 1))
    mu_p = np.dot(weights_.T, p)[0]
    mu_x = np.dot(weights_.T, x)[0]
    dots = np.dot(p.T, weights_ * x)
    Sigma_px = dots - np.outer(mu_p, mu_x)  # eq 24
    # x and p should no longer be used
    A_ij = Sigma_px - Sigma_px.T
    Delta = np.array([A_ij[1, 2], A_ij[2, 0], A_ij[0, 1]])
    tr_Sigma_px = np.trace(Sigma_px)
    # "N" in Horn:
    Q = np.empty((4, 4))
    Q[0, 0] = tr_Sigma_px
    Q[0, 1:] = Delta
    Q[1:, 0] = Delta
    Q[1:, 1:] = Sigma_px + Sigma_px.T - tr_Sigma_px * np.eye(3)
    _, v = np.linalg.eigh(Q)  # sorted ascending
    quat = np.empty(6)
    quat[:3] = v[1:, -1]
    if v[0, -1] != 0:
        quat[:3] *= np.sign(v[0, -1])
    rot = quat_to_rot(quat[:3])
    # scale factor is easy once we know the rotation
    if scale:  # p is "right" (from), x is "left" (to) in Horn 1987
        dev_x = x - mu_x
        dev_p = p - mu_p
        dev_x *= dev_x
        dev_p *= dev_p
        if weights is not None:
            dev_x *= weights_
            dev_p *= weights_
        s = np.sqrt(np.sum(dev_x) / np.sum(dev_p))
    else:
        s = 1.0
    # translation is easy once rotation and scale are known
    quat[3:] = mu_x - s * np.dot(rot, mu_p)
    return quat, s


def quat_to_rot(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a set of quaternions to rotations.

    Parameters
    ----------
    quat : array, shape (..., 3)
        The q1, q2, and q3 (x, y, z) parameters of a unit quaternion.

    Returns
    -------
    rot : array, shape (..., 3, 3)
        The corresponding rotation matrices.
    """
    # z = a + bi + cj + dk
    b, c, d = quat[..., 0], quat[..., 1], quat[..., 2]
    bb, cc, dd = b * b, c * c, d * d
    # use max() here to be safe in case roundoff errs put us over
    aa = np.maximum(1.0 - bb - cc - dd, 0.0)
    a = np.sqrt(aa)
    ab_2 = 2 * a * b
    ac_2 = 2 * a * c
    ad_2 = 2 * a * d
    bc_2 = 2 * b * c
    bd_2 = 2 * b * d
    cd_2 = 2 * c * d
    rotation = np.empty(quat.shape[:-1] + (3, 3))
    rotation[..., 0, 0] = aa + bb - cc - dd
    rotation[..., 0, 1] = bc_2 - ad_2
    rotation[..., 0, 2] = bd_2 + ac_2
    rotation[..., 1, 0] = bc_2 + ad_2
    rotation[..., 1, 1] = aa + cc - bb - dd
    rotation[..., 1, 2] = cd_2 - ab_2
    rotation[..., 2, 0] = bd_2 - ac_2
    rotation[..., 2, 1] = cd_2 + ab_2
    rotation[..., 2, 2] = aa + dd - bb - cc
    return rotation


def quat_to_affine(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion to 4x4 affine transformation."""
    assert quat.shape == (6,)
    affine = np.eye(4)
    affine[:3, :3] = quat_to_rot(quat[:3])
    affine[:3, 3] = quat[3:]
    return affine
