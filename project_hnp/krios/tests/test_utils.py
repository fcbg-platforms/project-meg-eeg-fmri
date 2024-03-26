import numpy as np
from numpy.testing import assert_allclose

from project_hnp.krios._utils import remove_duplicates


def test_remove_duplicates():
    """Test function which removes duplicates in a set of points."""
    points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    points_filtered = remove_duplicates(points, threshold=0.1)
    assert_allclose(points, points_filtered)

    points2 = np.array(
        [[1, 1, 1], [1.1, 1.1, 1.1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
    )
    points_filtered = remove_duplicates(points2, threshold=0.5)
    assert_allclose(points, desired=points_filtered)
    points_filtered = remove_duplicates(points2, threshold=0.001)
    assert_allclose(points2, desired=points_filtered)

    points3 = np.array(
        [
            [1, 1, 1],
            [1.1, 1.1, 1.1],
            [2, 2, 2],
            [3, 3, 3],
            [0.9, 0.9, 0.9],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    points_filtered = remove_duplicates(points3, threshold=0.5)
    assert_allclose(points, desired=points_filtered)
    points_filtered = remove_duplicates(points3, threshold=0.001)
    assert_allclose(points3, desired=points_filtered)

    points4 = np.array(
        [
            [1, 1, 1],
            [1.1, 1.1, 1.1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [4.1, 4.1, 4.1],
            [5, 5, 5],
        ]
    )
    points_filtered = remove_duplicates(points4, threshold=0.5)
    assert_allclose(points, desired=points_filtered)
    points_filtered = remove_duplicates(points4, threshold=0.001)
    assert_allclose(points4, desired=points_filtered)

    points5 = np.array(
        [
            [1, 1, 1],
            [1.1, 1.1, 1.1],
            [1.11, 1.1, 1.1],
            [2, 2, 2],
            [4.1, 4.11, 4.1],
            [3, 3, 3],
            [1.1, 1.1, 1.11],
            [4, 4, 4],
            [4.1, 4.1, 4.1],
            [5, 5, 5],
            [4.1, 4.1, 4.11],
        ]
    )
    points_filtered = remove_duplicates(points5, threshold=0.5)
    assert_allclose(points, desired=points_filtered)
    points_filtered = remove_duplicates(points5, threshold=0.001)
    assert_allclose(points5, desired=points_filtered)
