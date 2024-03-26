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
