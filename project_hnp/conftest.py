from __future__ import annotations  # c.f. PEP 563, PEP 649

import os
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from pytest import Config


def pytest_configure(config: Config) -> None:
    """Configure pytest options."""
    warnings_lines = r"""
    error::
    """
    for warning_line in warnings_lines.split("\n"):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith("#"):
            config.addinivalue_line("filterwarnings", warning_line)

    testing_dataset = os.getenv("PROJECT_MEG_EEG_fMRI_TESTING_DATASET")
    if testing_dataset is None:
        warn(
            "Missing testing dataset environment variable "
            "'PROJECT_MEG_EEG_fMRI_TESTING_DATASET'.",
            RuntimeWarning,
            stacklevel=1,
        )
    elif not Path(testing_dataset).exists():
        warn(
            f"Testing dataset '{testing_dataset}' does not exist.",
            RuntimeWarning,
            stacklevel=1,
        )
