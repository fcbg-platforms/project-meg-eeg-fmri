EXPECTED_EEG: set[str] = {"aud", "feet", "fing", "rest", "vis"}
EXPECTED_MEG: set[str] = {"aud", "feet", "fing", "noise", "rest", "vis"}
OPTIONAL_MEG: set[str] = {"resteeg"}
EXPECTED_MRI: set[str] = {"DICOM", "NIfTI"}
EXPECTED_fMRI_NIFTI: set[str] = {"feet", "hand", "face", "sparseAuditory"}
EXPECTED_fMRI_T1: list[str] = ["t1_mprage"]
# fmt: off
EGI_CH_TO_DROP: list[str] = [
    "31", "241", "242", "244", "245", "246", "247", "248", "249", "250", "251", "F9",
    "253", "254", "255", "256", "FT9", "73", "82", "91", "92", "102", "111", "120",
    "133", "145", "165", "174", "187", "199", "208", "209", "216", "217", "218", "FT10",
    "229", "228", "227", "225", "F10", "230", "231", "232", "233", "234", "235", "236",
    "237", "238", "239", "240"
]
# fmt: on
