"""Convert one session to ONE format and map its average receptive fields."""
from pathlib import Path

import numpy as np

from OneConverter import OneSessionConverter
from one.api import One
from mapAverageRF import map_average_rf

ONE_ROOT = Path(__file__).parent / "ExampleSession"

# Settings in step order. sparse_noise_shape and stimulus_edges_deg have no default; the others repeat
# mapAverageRF.DEFAULT_SETTINGS for readability. encoder_counts_per_revolution is used only with running_regressor.
SETTINGS = {
    "sparse_noise_shape": (7, 27),                       # conversion + stimulus: (rows, columns)
    "stimulus_edges_deg": (-135.0, 135.0, -40.0, 40.0),  # conversion + stimulus + step 9: (left, right, bottom, top)
    "rf_time_limits": (0.2, 1.0),                        # step 1a: RF delays (s)
    "azimuth_range_contra_deg": None,                    # step 1a: stimulus crop, (ipsilateral, contralateral) limits in degrees towards the FOV's contralateral side (0 = midline); None = no crop, or no limit on that side
    "elevation_range_deg": None,                         # step 1a: stimulus crop, (lower, upper) screen elevation (deg); None = no crop / no limit
    "right_hemisphere_fovs": [],                         # step 1a: stimulus crop, FOVs in the right hemisphere, e.g. ["FOV_01"]; others are left
    "running_regressor": False,                          # step 1b: add running speed as extra predictors
    "run_time_limits": (-5.0, 5.0),                      # step 1b: running-speed lags around each sample (s)
    "signal": "traces",                                  # step 2: "traces" (F, neuropil-corrected) or "deconvolved" (Suite2p spikes)
    "smoothing_sigma": 2,                                # step 2b: Gaussian smoothing of F and Fneu (imaging samples); traces only
    "padding": 10,                                       # step 2c: s of calcium kept around the stimulus period
    "max_nan_fraction": 0.1,                             # step 2d: neurons with at most this NaN fraction are zero-filled
    "lambdas": np.logspace(-4, -1, 4),                   # step 3: ridge penalties tried
    "l1_enabled": False,                                 # step 3: add an L1 penalty on all coefficients (PyProximal FISTA; much slower)
    "l1_lambdas": np.logspace(-5, -2, 4),                # step 3: L1 penalties tried (with l1_enabled)
    "l1_max_iter": 500,                                  # step 3: maximum FISTA iterations (with l1_enabled)
    "l1_tol": 1e-4,                                      # step 3: FISTA stops when the objective changes by less than this fraction
    "cross_folds": 3,                                    # step 4: cross-validation folds
    "n_shifts": 500,                                     # step 6: circular shifts in the significance test
    "random_seed": 0,                                    # step 6: seed for the shifts
    "max_p_val": 0.05,                                   # step 6: significance threshold
    "min_ev": 0.01,                                      # step 6: minimum cross-validated explained variance
    "thresh_subfield": 0.7,                              # step 7: ON+OFF chosen if its peak reaches this fraction
    "min_peak_to_noise": 7.7,                            # step 9: "good RF" threshold
    "force": False,                                      # rerun a FOV even if results with the same settings are saved
}

# %% Conversion
session = OneSessionConverter(
    "2P", "NM023", "20260428", 1,
    raw_data_root="//10.233.25.135/FANCiNAS1/Data",
    processed_data_root="//10.233.25.135/FANCiNAS1/Processed",
    one_root=ONE_ROOT,
)
session.exporters.SparseNoiseExporter.sparse_noise_shape = SETTINGS["sparse_noise_shape"]
session.exporters.SparseNoiseExporter.stimulus_edges_deg = SETTINGS["stimulus_edges_deg"]
session.export()  # also re-indexes ONE_ROOT (converted data and earlier RF results)

# %% RF mapping: results are written into the session, alf/FOV_XX/rf_<traces|deconv>/#<date>#/_ss_averageRF.*
one = One(ONE_ROOT, mode="local")
map_average_rf(one, session.session_path, SETTINGS)
