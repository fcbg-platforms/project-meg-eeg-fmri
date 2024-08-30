## The `mri2bids.py` Script

The `mri2bids.py` script facilitates converting data from the `MRI_analyses` directory into BIDS format and saving it in the `dataset_bids` directory. The script includes multiple steps, each controlled by a boolean variable. By default, all steps are enabled. To run specific steps, modify the corresponding variables in the script. For each step, if the corresponding BIDS-formatted files already exist for a subject, the conversion is skipped. The progress is recorded in the `logger-mri2bids.log` file.

You can launch it using the following command:

```python
ipython path/to/mri2bids.py
```


**Available Steps and Control Variables:**

1. **T1w in SPM Template Space (BIDS format)** → `RUN_T1TEMP2BIDS`
   - Converts T1w files in SPM template space from `MRI_analyses/Subj-xxx/NIfTI/w*_t1_mprage_sag_p2_iso_Fov256.nii` to `dataset_bids/derivatives/{sub}/anat/{sub}_space-spmMNI_T1w.nii.gz`.

2. **Convert fMRI SPM Contrasts to BIDS Format** → `RUN_FMRICONTRAST2BIDS`
   - Converts SPM contrast maps from `MRI_analyses/Subj-xxx/Results/TASK/spmT_XXXX.nii` to `dataset_bids/derivatives/{sub}/func/task-{task}/{sub}_task-{task}_contrast-{contrast}_desc-stat-t_statmap.nii.gz`.
   - Requires the `Corresp_SPM_con_{task}.xlsx` files from `data_info` to map the SPM numerical identifier for the contrast to its corresponding name during the conversion.

3. **Convert SPM 'con' Output to BIDS Format** → `RUN_SPMCON2BIDS`
   - Converts SPM 'con' files from `MRI_analyses/Subj-xxx/Results/TASK/con_XXXX.nii` to `dataset_bids/derivatives/{sub}/func/task-{task}/{sub}_task-{task}_contrast-{contrast}_desc-stat-con_statmap.nii.gz`.
   - Requires `Corresp_SPM_con_{task}.xlsx` files from `data_info`.

4. **Convert SPM Beta Output to BIDS Format** → `RUN_SPMBETA2BIDS`
   - Converts beta maps from `MRI_analyses/Subj-xxx/Results/TASK/beta_XXXX.nii` to `dataset_bids/derivatives/{sub}/func/task-{task}/{sub}_task-{task}_desc-regressor-{regressor}_betamap.nii.gz`.
   - Requires the `Corresp_beta_regressors_{task}.xlsx` files from `data_info` to map the SPM numerical identifier of the regressor to its corresponding name during the conversion.

5. **Convert SPM Residual Mean Square (ResMS) Output to BIDS Format** → `RUN_RESMS2BIDS`
   - Converts ResMS files from `MRI_analyses/Subj-xxx/Results/TASK/ResMS.nii` to `dataset_bids/derivatives/{sub}/func/task-{task}/{sub}_task-{task}_desc-stat-ResMS_statmap.nii.gz`.

6. **Convert SPM Estimated RESELS per Voxels (RPV) Output to BIDS Format** → `RUN_RPV2BIDS`
   - Converts RPV files from `MRI_analyses/Subj-xxx/Results/TASK/RPV.nii` to `dataset_bids/derivatives/{sub}/func/task-{task}/{sub}_task-{task}_desc-stat-RPV_statmap.nii.gz`.

7. **Convert SPM Brain Mask Output to BIDS Format** → `RUN_MASK2BIDS`
   - Converts brain masks from `MRI_analyses/Subj-xxx/Results/TASK/mask.nii` to `dataset_bids/derivatives/{sub}/func/task-{task}/{sub}_task-{task}_desc-brainmask.nii.gz`.