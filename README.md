# Average receptive-field mapping on 2P data

Python version of the lab's average RF mapping (sparse-noise stimulus, ridge regression), run on sessions converted to the IBL ONE format.

## Repositories needed

Put the repositories side by side in one folder:

```
FANCiRepos/
├── Average_RF_mapping/        https://github.com/FANCilab/RF_mappers: this project (clone it into a folder named Average_RF_mapping)
├── ONE/                       https://github.com/int-brain-lab/ONE: IBL's loader, finds and loads the converted files
└── OneConverter/              https://github.com/FANCilab/OneConverter: converts raw data + Suite2p output to ONE files
```

ONE and OneConverter must be installed in your Python environment. Installation steps are in [AGENTS.md](AGENTS.md).

## Using `map_average_rf`

[mapAverageRF.py](mapAverageRF.py) has one function:

```python
from one.api import One
from mapAverageRF import map_average_rf

one = One("./ExampleRoot/", mode="local")          # the ONE root the converter wrote to
session_path = "./ExampleRoot/NM023/2026-04-28/001"

settings = {
    "sparse_noise_shape": (7, 27),                       # required: (rows, columns), as in the conversion
    "stimulus_edges_deg": (-135.0, 135.0, -40.0, 40.0),  # required: (left, right, bottom, top), as in the conversion
    "right_hemisphere_fovs": ["FOV_01"],                 # FOVs in the right hemisphere; the others are left
    "signal": "deconvolved",                             # "traces" (neuropil-corrected F) or "deconvolved"
    "n_shifts": 500,                                     # 5 for a quick test, 500 for results
}
results_path = map_average_rf(one, session_path, settings)
```

For every FOV of the session it:

1. builds the stimulus design matrix (each pixel, ON and OFF, at each delay in `rf_time_limits`), optionally cropped to part of the screen and with running speed as extra predictors;
2. prepares the activity of curated neurons (smoothed and neuropil-corrected F, or Suite2p deconvolved), sampled at stimulus times and z-scored;
3. fits each neuron with smoothness-regularised ridge regression (optionally plus L1), choosing the penalty by cross-validation;
4. tests significance by refitting on circularly shifted traces (p-value) and keeps RFs with p < `max_p_val` and explained variance > `min_ev`;
5. for significant RFs, picks the subfield (ON, OFF or ON+OFF) and the best delay, fits a 2D Gaussian, and marks RFs with peak-to-noise > `min_peak_to_noise` as good.

All settings and their defaults are listed in `DEFAULT_SETTINGS` at the top of the file; any you pass override them.

**Output.** Results are saved inside the session, one revision per run:

```
alf/FOV_00/rf_deconv/#2026-10-27#/        rf_traces/ when signal = "traces"; #2026-10-27a#, b, ... for later runs that day
├── _ss_averageRF.*.npy                   one row per fitted neuron: maps, pValues, explainedVariance, gaussPars, goodRF, ...
└── _ss_averageRFRun.meta.json            settings, inputs and counts of the run
```

If a saved revision already has the same settings, that FOV is skipped and the saved run is reused; add `"force": True` to refit. The function returns `{FOV collection: path to that run's meta.json}`. Load results with ONE (latest revision by default):

```python
rf = one.load_object(session_path, "averageRF", collection="alf/FOV_00/rf_deconv")
rf.goodRF, rf.gaussPars, rf.maps
```

`explain_dataset(one, "_ss_averageRF.maps.npy")` (from OneConverter) prints what each file contains.

[main.py](main.py) is a short script doing conversion + mapping for one session.

## The notebook: [RF_mapping_tutorial.ipynb](RF_mapping_tutorial.ipynb)

A walk-through from raw data to plots, on session NM023/2026-04-28/001:

1. **Converting a session (OneConverter).** Create an `OneSessionConverter` (modality, subject, date, session number, raw / Suite2p / output folders), look at the detected exporters (2P imaging, sparse noise, wheel) and their settings, export, and see where the files go. `explain_dataset` prints what a file contains.
2. **Finding and loading data (ONE).** How ALF file names work (session, collection, object, attribute, namespace, revision); the shared session clock for imaging, wheel and stimulus; listing collections, datasets and revisions; loading a dataset, an object and a whole collection, with example F traces and the mean image.
3. **Mapping receptive fields.** The settings that matter (required, important, optional); two runs on deconvolved activity, without and then with the running regressor; where the results are saved and how to load a given revision; comparing significant RFs between the two runs.
4. **Plotting.** `plot_session` ([single_session_plots.py](single_session_plots.py)) plots one run per FOV and pooled per session: Gaussian coverage, RF size, explained variance, ROI overlays, RF maps, predicted vs. actual traces, time courses. `plot_group` ([grouped_session_plots.py](grouped_session_plots.py)) pools FOVs across sessions: RF centres, coverage, explained variance and size. 
5. **Recap on a second session.** NM034/2026-05-06/001 from conversion to group plots in one cell, using traces instead of deconvolved activity.

Notes:
- Converting needs access to the lab NAS (`//10.233.25.135/FANCiNAS1`). Without it, use an already converted ONE root and skip the conversion cells.
- Some cells name the revisions `2026-10-27` and `2026-10-27a`, the dates the notebook was first run. When you rerun, your results get today's date; change those names accordingly.
- The notebook uses `n_shifts=5` to run fast. Use 500 for real results.
