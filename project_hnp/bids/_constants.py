EXPECTED_EEG: set[str] = {"aud", "feet", "fing", "rest", "vis"}
EXPECTED_MEG: set[str] = {"aud", "feet", "fing", "noise", "rest", "vis"}
OPTIONAL_MEG: set[str] = {"resteeg"}
EXPECTED_MRI: set[str] = {"DICOM", "NIfTI"}
EXPECTED_fMRI_NIFTI: set[str] = {"feet", "hand", "face", "sparseAuditory"}
EXPECTED_fMRI_T1: list[str] = ["t1_mprage"]
MAPPING_fMRI = {"feet": "feet", "hand": "fing", "face": "vis", "sparseAuditory": "aud"}
