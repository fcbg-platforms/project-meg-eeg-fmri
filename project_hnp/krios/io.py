from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
from mne.channels import make_dig_montage, make_standard_montage
from mne.channels.montage import _get_montage_in_head
from mne.defaults import HEAD_SIZE_DEFAULT as _HEAD_SIZE_DEFAULT
from pycpd import RigidRegistration

from ..utils._checks import ensure_path
from ._transform import krios_to_head_coordinate, reorder_electrodes
from ._utils import remove_duplicates

if TYPE_CHECKING:
    from pathlib import Path

    from mne.channels import DigMontage
    from numpy.typing import NDArray


_LPA_LABEL: str = "LEFT_PP_MARKER"
_RPA_LABEL: str = "RIGHT_PP_MARKER"
_NZ_LABEL: str = "NASION_MARKER"
_TEMPLATE_FNAME: Path = (
    files("project_hnp.krios") / "assets" / "EGI 257.Geneva Average 13.10-10.xyz"
)


def read_krios(fname: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read electrode locations from a Krios file.

    Parameters
    ----------
    fname : Path | str
        Path to the Krios file.

    Returns
    -------
    elc : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the EEG electrodes in the head coordinate system.
    fid : array of shape (3, 3)
        (x, y, z) coordinates of the fiducials in the head coordinate system, ordered as
        RPA, LPA, NZ.
    """
    fname = ensure_path(fname, must_exist=True)
    df = np.loadtxt(
        fname,
        delimiter=",",
        dtype=np.dtype([("Ref", "U7"), ("xyz", np.float64, (3,)), ("type", "U15")]),
        skiprows=1,
        usecols=(2, 3, 4, 5, 6),
    )
    # let's start to filter out some duplicates
    refs = [elt[0] for elt in df]
    types = [elt[2] for elt in df]
    df = np.array([elt[1] for elt in df], dtype=np.float64)
    fid_idx = np.array(
        [types.index(_RPA_LABEL), types.index(_LPA_LABEL), types.index(_NZ_LABEL)],
        dtype=int,
    )
    eeg_idx = np.array(
        [
            k
            for k, elt in enumerate(zip(refs, types, strict=True))
            if elt[0].lower().strip() == "scanned" and elt[1].lower().strip() == "eeg"
        ],
        dtype=int,
    )
    # extract (x, y, z) coordinates as array of shape (n_scanned, 3)
    elc = df[eeg_idx, :]
    elc = remove_duplicates(elc)
    fid = df[fid_idx, :]
    del df
    elc, fid, _ = krios_to_head_coordinate(elc, fid)
    # read template file
    elc_template = np.loadtxt(
        _TEMPLATE_FNAME, skiprows=1, usecols=(0, 1, 2), max_rows=257, dtype=np.float64
    )
    # co-register with template
    # RigidRegistration(...).register() returns 2 elements:
    # - TY : array of shape (n_points, 3), the registered and transformed source points
    # - registration parameters : tuple of 3 elements
    #   - s_reg : float, the scale of the registration
    #   - R_reg : array of shape (3, 3), the rotation matrix of the registration
    #   - t_reg : array of shape (3,), the translation vector of the registration
    elc_TY, (s_reg, _, _) = RigidRegistration(X=elc_template, Y=elc).register()
    fid *= s_reg  # apply the same scaling to fiducials
    # reorder electrodes according to template
    elc_reordered = reorder_electrodes(elc_TY, elc_template)
    return elc_reordered, fid


def read_krios_montage(fname: Path | str) -> DigMontage:
    """Read electrode locations from a Krios file in a DigMontage.

    Parameters
    ----------
    fname : Path | str
        Path to the Krios file.

    Returns
    -------
    montage : DigMontage
        MNE channel montage for the EEG electrodes in the head coordinate frame.
    """
    elc, fid = read_krios(fname)
    # apply scaling to meters
    standard = make_standard_montage("standard_1020", head_size=_HEAD_SIZE_DEFAULT)
    standard = _get_montage_in_head(standard).get_positions()
    assert standard["coord_frame"] == "head"  # sanity-check
    fid_standard = np.array(
        [standard["rpa"], standard["lpa"], standard["nasion"]], dtype=np.float64
    )
    scaling = np.average(
        np.linalg.norm(fid_standard, axis=1) / np.linalg.norm(fid, axis=1)
    )
    elc *= scaling
    fid *= scaling
    # figure out labels from template
    ch_names = np.loadtxt(
        _TEMPLATE_FNAME, skiprows=1, usecols=3, max_rows=257, dtype=str
    )
    if elc.shape[0] != ch_names.size:
        raise ValueError(
            f"Number of electrodes ({elc.shape[0]}) does not match the "
            f"number of channel names ({ch_names.size}) in the template."
        )
    ch_pos = dict(zip(ch_names, elc, strict=True))
    return make_dig_montage(
        ch_pos=ch_pos,
        nasion=fid[2, :],
        lpa=fid[1, :],
        rpa=fid[0, :],
        coord_frame="head",
    )
