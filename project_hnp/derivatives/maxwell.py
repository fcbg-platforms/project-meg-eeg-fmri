from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils._checks import ensure_path, ensure_subject_int
from ..utils._docs import fill_doc

if TYPE_CHECKING:
    from pathlib import Path


@fill_doc
def run_maxwell_filter(root: Path | str, derivative: Path | str, subject: int):
    """Run SSS on th MEG recording for the given subject.

    Parameters
    ----------
    %(bids_root)s
    %(bids_derivative)s
    %(bids_subject)s
    """
    root = ensure_path(root, must_exist=True)
    derivative = ensure_path(derivative, must_exist=True)
    subject = ensure_subject_int(subject)
