EXPECTED_EEG: set[str] = {"aud", "feet", "fing", "rest", "vis"}
EXPECTED_MEG: set[str] = {"aud", "feet", "fing", "noise", "rest", "vis"}
OPTIONAL_MEG: set[str] = {"resteeg"}
EXPECTED_MRI: set[str] = {"DICOM", "NIfTI"}
EXPECTED_fMRI_NIFTI: set[str] = {"feet", "hand", "face", "sparseAuditory"}
EXPECTED_fMRI_T1: list[str] = ["t1_mprage"]
MAPPING_fMRI: dict[str, str] = {
    "feet": "feet",
    "hand": "fing",
    "face": "vis",
    "sparseAuditory": "aud",
}

TRIGGERS_VIS: dict[str, dict[str, int] | str] = {
    "events": {"face": 1, "scramble": 2, "house": 3},
    "egi_stim_channel": "STI 014",
    "meg_stim_channel": "STI102",
}
TRIGGER_AUD: dict[str, dict[str, int] | str] = {
    "events": {"sine": 1, "piano1": 2, "piano4": 3, "guiro": 4},
    "egi_stim_channel": "STI 014",
    "meg_stim_channel": "STI102",
}
TRIGGER_FING: dict[str, dict[str, int] | str] = {
    "events": {"finger": 1},
    "egi_stim_channel": "STI 014",
    "meg_stim_channel": "STI101",
}
TRIGGER_FEET: dict[str, dict[str, int] | str] = {
    "events": {"right": 1, "left": 2},
    "egi_stim_channel": "STI 014",
    "meg_stim_channel": "STI101",
}
TRIGGERS: dict[str, dict[str, dict[str, int] | str]] = {
    "vis": TRIGGERS_VIS,
    "aud": TRIGGER_AUD,
    "fing": TRIGGER_FING,
    "feet": TRIGGER_FEET,
    "rest": dict(),
    "resteeg": dict(),
    "noise": dict(),
}
