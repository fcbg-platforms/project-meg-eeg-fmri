from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from numpy.typing import NDArray


def remove_duplicates(
    points: NDArray[np.float64], threshold: float = 0.5
) -> NDArray[np.float64]:
    """Remove duplicate electrode locations.

    Parameters
    ----------
    points : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the points to filter.
    threshold : float
        Distance threshold to consider two points as duplicates.

    Returns
    -------
    points : array of shape (n_electrodes, 3)
        (x, y, z) coordinates of the filtered points.
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    distance_matrix = cdist(points, points)
    np.fill_diagonal(distance_matrix, np.inf)
    mask = np.any(distance_matrix < threshold, axis=1)
    if np.all(~mask):
        return points  # by-pass slow duplicate removal
    groups = list()
    for k in np.where(mask)[0]:
        if len(groups) == 0:
            groups.append([k])
            continue
        for group in groups:
            dist = [np.linalg.norm(points[k] - points[elt]) for elt in group]
            dist = np.average(dist)
            if dist < threshold:
                group.append(k)
                break
        else:
            groups.append([k])
    # determine average location for duplicate electrodes
    groups = [
        np.average(np.array([points[k] for k in group]), axis=0) for group in groups
    ]
    return np.vstack([groups, points[~mask]])
