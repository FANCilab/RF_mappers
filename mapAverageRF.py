"""Average receptive-field mapping of one session, with optional L1 regularisation.
"""
import json
from datetime import date, datetime
from itertools import product
from math import ceil, floor
from string import ascii_lowercase

import numpy as np
from one.alf.cache import make_parquet_db
from one.alf.io import save_object_npy
from one.alf.path import ALFPath
from one.alf.spec import to_alf
from OneConverter import update_dataset_types
from scipy.linalg import block_diag

from utils import (estimate_neuropil_lfr, explained_variance, fit, fit_2d_gauss_rf, gaussian_smooth, interp_columns,
                   interpolate_rf_to_visual_degrees, make_lambda_matrix, make_stim_toeplitz, shift_test)

# Default settings, in step order.
# sparse_noise_shape and stimulus_edges_deg have no default and must be given.
DEFAULT_SETTINGS = {
    "rf_time_limits": (0.2, 1.0),       # step 1a: RF delays (s)
    "azimuth_range_contra_deg": None,   # step 1a: stimulus crop, (ipsilateral, contralateral) limits in degrees towards
                                        # the FOV's contralateral side (0 = midline); None = no crop, or no limit on that side
    "elevation_range_deg": None,        # step 1a: stimulus crop, (lower, upper) screen elevation (deg); None = no crop / no limit
    "right_hemisphere_fovs": [],        # step 1a: stimulus crop, FOVs in the right hemisphere, e.g. ["FOV_01"]; others are left
    "running_regressor": False,         # step 1b: add running speed as extra predictors
    "run_time_limits": (-5.0, 5.0),     # step 1b: running-speed lags around each sample (s)
    "signal": "traces",                 # step 2: "traces" (F, neuropil-corrected) or "deconvolved" (Suite2p spikes)
    "smoothing_sigma": 2,               # step 2b: Gaussian smoothing of F and Fneu (imaging samples); traces only
    "padding": 10,                      # step 2c: s of calcium kept around the stimulus period
    "max_nan_fraction": 0.1,            # step 2d: neurons with at most this NaN fraction are zero-filled
    "lambdas": np.logspace(-4, -1, 4),  # step 3: ridge penalties tried
    "l1_enabled": False,                # step 3: add an L1 penalty on all coefficients (PyProximal FISTA; much slower)
    "l1_lambdas": np.logspace(-5, -2, 4),  # step 3: L1 penalties tried (with l1_enabled)
    "l1_max_iter": 500,                 # step 3: maximum FISTA iterations (with l1_enabled)
    "l1_tol": 1e-4,                     # step 3: FISTA stops when the objective changes by less than this fraction
    "cross_folds": 3,                   # step 4: cross-validation folds
    "n_shifts": 500,                    # step 6: circular shifts in the significance test
    "random_seed": 0,                   # step 6: seed for the shifts
    "max_p_val": 0.05,                  # step 6: significance threshold
    "min_ev": 0.01,                     # step 6: minimum cross-validated explained variance
    "thresh_subfield": 0.7,             # step 7: ON+OFF chosen if its peak reaches this fraction
    "min_peak_to_noise": 7.7,           # step 9: "good RF" threshold
    "force": False,                     # rerun a FOV even if results with the same settings are saved
}



def map_average_rf(one, session_path, settings):
    """Fit sparse-noise RFs of every neuron in every FOV of the session at session_path.

    session_path is the session folder inside the ONE root (subject/date/number); ONE
    looks it up in the index of `one`, so no eid is needed.

    settings must contain sparse_noise_shape (rows, columns) and stimulus_edges_deg
    (left, right, bottom, top); any other key overrides DEFAULT_SETTINGS.
    settings["signal"] chooses the activity that is fit: "traces" (F, smoothed and
    neuropil-corrected) or "deconvolved" (Suite2p spikes, no preprocessing).
    settings["running_regressor"] adds running speed as extra predictors; RF
    significance is then judged on the stimulus part of the model only.
    settings["l1_enabled"] adds an L1 penalty (settings["l1_lambdas"], chosen by cross-validation
    together with the ridge penalties) and fits with FISTA (PyProximal) instead of least squares.

    Results are saved next to the session data as the ALF object _ss_averageRF in
    alf/FOV_XX/rf_<traces|deconv>/#<date>[a, b, ...]#/: one row per fitted neuron
    (_ss_averageRF.*.npy), with the run metadata in its own object (_ss_averageRFRun.meta.json).
    A FOV is skipped when a saved revision has the same settings, unless settings["force"].

    Returns {FOV collection: ALFPath of its _ss_averageRFRun.meta.json}, e.g. {"alf/FOV_00": .../alf/FOV_00/rf_traces/
    #2026-10-26#/_ss_averageRFRun.meta.json}: the revision written by this run, or for a skipped FOV the existing
    revision with the same settings. one.alf.io.load_object(path.parent, "averageRF") loads that run's results.
    """
    settings = {**DEFAULT_SETTINGS, **settings}
    n_rows, n_cols = settings["sparse_noise_shape"]       # stimulus: (rows, columns)
    edges = np.array(settings["stimulus_edges_deg"])       # stimulus + step 9: (left, right, bottom, top)
    rf_time_limits = settings["rf_time_limits"]
    azimuth_range_contra = settings["azimuth_range_contra_deg"]
    elevation_range = settings["elevation_range_deg"]
    right_hemisphere_fovs = settings["right_hemisphere_fovs"]
    running_regressor = settings["running_regressor"]      
    run_time_limits = settings["run_time_limits"]          
    signal = settings["signal"]                            
    smoothing_sigma = settings["smoothing_sigma"]          
    padding = settings["padding"]                          
    max_nan_fraction = settings["max_nan_fraction"]        
    lambdas = np.asarray(settings["lambdas"])
    l1_enabled = settings["l1_enabled"]
    l1_lambdas = np.asarray(settings["l1_lambdas"])
    l1_max_iter = settings["l1_max_iter"]
    l1_tol = settings["l1_tol"]
    cross_folds = settings["cross_folds"]                  
    n_shifts = settings["n_shifts"]                        
    rng = np.random.default_rng(settings["random_seed"])   
    max_p_val = settings["max_p_val"]                      
    min_ev = settings["min_ev"]                            
    thresh_subfield = settings["thresh_subfield"]          
    min_peak_to_noise = settings["min_peak_to_noise"]      
    if signal not in ("traces", "deconvolved"):
        raise ValueError(f'settings["signal"] must be "traces" or "deconvolved", got {signal!r}')

    session_path = ALFPath(session_path)
    variant = "rf_" + ("traces" if signal == "traces" else "deconv")
    meta_name = to_alf("averageRFRun", "meta", "json", namespace="ss")

    # Settings as saved in meta.json (JSON round trip, so they compare equal to a loaded file); force is not saved.
    saved_settings = json.loads(json.dumps({
        **{k: np.asarray(v).tolist() if isinstance(v, (tuple, np.ndarray)) else v
           for k, v in settings.items() if k != "force"},
        "sparse_noise_shape": {"rows": n_rows, "columns": n_cols},
        "stimulus_edges_deg": dict(zip(("left", "right", "bottom", "top"), edges.tolist())),
    }))

    # %% Stimulus
    # Raw int8 bytes are [frame, col, row] in C order; -128 gray, -1 black, 0 white.
    # Column 0 is the left edge of the screen, row 0 its bottom.
    t_stim = one.load_dataset(session_path, "_ibl_passiveRFM.times.npy", collection="alf")
    raw_path = one.load_dataset(session_path, "_iblrig_RFMapStim.raw.bin", collection="raw_passive_data")
    frames = np.fromfile(raw_path, dtype=np.int8).reshape(len(t_stim), n_cols, n_rows)
    frames = np.select([frames == -1, frames == 0], [-1.0, 1.0], 0.0)  # [frame x cols x rows]
    col_edges = np.linspace(edges[0], edges[1], n_cols + 1)  # azimuth borders of the stimulus columns
    row_edges = np.linspace(edges[2], edges[3], n_rows + 1)  # elevation borders of the stimulus rows
    col_centres = (col_edges[:-1] + col_edges[1:]) / 2
    row_centres = (row_edges[:-1] + row_edges[1:]) / 2

    # RF delays in stimulus frames
    frame_dt = np.median(np.diff(t_stim))
    rf_bins = np.arange(floor(rf_time_limits[0] / frame_dt), ceil(rf_time_limits[1] / frame_dt) + 1)
    t_rf = rf_bins * frame_dt

    # %% Step 1a. Stimulus design [time x (delays * pixels)], shared by all FOVs; each FOV keeps its cropped pixels
    toeplitz, t_toeplitz = make_stim_toeplitz(frames, t_stim, rf_bins)

    # %% Step 1b. Running design [time x lags]: z-scored running speed around each sample
    if running_regressor:
        wheel = one.load_object(session_path, "wheel", collection="alf")
        model_dt = np.median(np.diff(t_toeplitz))
        run_bins = np.arange(floor(run_time_limits[0] / model_dt), ceil(run_time_limits[1] / model_dt) + 1)
        t_run = run_bins * model_dt
        speed = np.interp(t_toeplitz, wheel.timestamps, np.abs(wheel.velocity), left=np.nan, right=np.nan)
        speed = (speed - np.nanmean(speed)) / np.nanstd(speed, ddof=1)
        # Column r holds the speed run_bins[r] samples away; 0 outside the recording or where NaN.
        running = np.zeros((len(speed), len(run_bins)))
        for r, lag in enumerate(run_bins):
            source = np.arange(len(speed)) + lag
            inside = (source >= 0) & (source < len(speed))
            running[inside, r] = speed[source[inside]]
        running = np.nan_to_num(running)
    else:
        running = np.zeros((len(t_toeplitz), 0))

    # %% Steps 2-9 for each FOV
    results = {}  # FOV collection -> ALFPath of the run's meta.json
    for fov in sorted(one.list_collections(session_path, collection="alf/FOV_??")):  # not the rf_* sub-collections
        # Skip if a saved revision already has these settings.
        collection = f"{fov}/{variant}"
        revisions = one.list_revisions(session_path, collection=collection)
        same = [r for r in one.list_revisions(session_path, collection=collection, filename=meta_name)
                if not settings["force"] and one.load_dataset(
                    session_path, meta_name, collection=collection, revision=r)["settings"] == saved_settings]
        if same:
            print(f"{fov}: results with the same settings already saved in {collection}/#{same[0]}#; "
                  f"skipped (set force=True to rerun)")
            results[fov] = session_path / collection / f"#{same[0]}#" / meta_name
            continue

        # Stimulus crop: azimuth limits are relative to the contralateral side, so they flip sign for the right
        # hemisphere; a pixel is kept when its centre lies inside both ranges.
        right_hemisphere = fov.split("/")[-1] in right_hemisphere_fovs
        ipsi, contra = azimuth_range_contra or (None, None)
        ipsi = -np.inf if ipsi is None else ipsi
        contra = np.inf if contra is None else contra
        azimuth_range = (-contra, -ipsi) if right_hemisphere else (ipsi, contra)
        lower, upper = elevation_range or (None, None)
        elevation_range_fov = (-np.inf if lower is None else lower, np.inf if upper is None else upper)
        kept_cols = np.flatnonzero((col_centres >= azimuth_range[0]) & (col_centres <= azimuth_range[1]))
        kept_rows = np.flatnonzero((row_centres >= elevation_range_fov[0]) & (row_centres <= elevation_range_fov[1]))
        if kept_cols.size == 0 or kept_rows.size == 0:
            raise ValueError(f"{fov}: the stimulus crop removes every column or row")
        stim_size = (len(kept_cols), len(kept_rows))
        fov_edges = np.array([col_edges[kept_cols[0]], col_edges[kept_cols[-1] + 1],
                              row_edges[kept_rows[0]], row_edges[kept_rows[-1] + 1]])
        kept_pixels = np.zeros((n_cols, n_rows), bool)
        kept_pixels[np.ix_(kept_cols, kept_rows)] = True
        fov_toeplitz = toeplitz[:, np.tile(kept_pixels.ravel(), len(rf_bins))]  # pixel order as in frames

        mpci = one.load_object(session_path, "mpci", collection=fov)
        rois = one.load_object(session_path, "mpciROIs", collection=fov, attribute="mpciROITypes")

        # Step 2. Calcium traces
        if signal == "traces":
            # a. Curated neurons with a non-zero F trace.
            roi_index = np.flatnonzero((rois.mpciROITypes == 1) & mpci.ROIActivityF.any(0))
            # b. Smooth F and Fneu, then LFR neuropil correction.
            F = gaussian_smooth(mpci.ROIActivityF[:, roi_index].astype(float), smoothing_sigma)
            Fneu = gaussian_smooth(mpci.ROINeuropilActivityF[:, roi_index].astype(float), smoothing_sigma)
            ca_traces = estimate_neuropil_lfr(F, Fneu)
        else:
            # a-b. Curated neurons; deconvolved activity is used as it is.
            roi_index = np.flatnonzero(rois.mpciROITypes == 1)
            ca_traces = mpci.ROIActivityDeconvolved[:, roi_index].astype(float)
        n_neurons = len(roi_index)
        print(f"{fov}: {n_neurons} neurons ({signal}, running regressor {'on' if running_regressor else 'off'}, "
              f"L1 {'on' if l1_enabled else 'off'})")

        # c. Crop to the stimulus period plus padding, sample at stimulus times, z-score.
        in_window = (mpci.times > t_stim[0] - padding) & (mpci.times < t_stim[-1] + padding)
        ca_traces = interp_columns(mpci.times[in_window], ca_traces[in_window], t_toeplitz)
        z_traces = (ca_traces - np.nanmean(ca_traces, 0)) / np.nanstd(ca_traces, 0, ddof=1)

        # d. Drop samples where every neuron is NaN; zero-fill neurons with few NaNs; skip the rest.
        valid_times = ~np.isnan(z_traces).all(1)
        z_traces, design, run_design = z_traces[valid_times], fov_toeplitz[valid_times], running[valid_times]
        few_nans = np.isnan(z_traces).mean(0) <= max_nan_fraction
        z_traces[:, few_nans] = np.nan_to_num(z_traces[:, few_nans])
        valid_units = ~np.isnan(z_traces).any(0)

        # e. Drop samples with no stimulus in any delay (gaps filled by make_stim_toeplitz).
        blank = (design == 0).all(1)
        valid_times[np.flatnonzero(valid_times)[blank]] = False
        z_traces, design, run_design = z_traces[~blank], design[~blank], run_design[~blank]

        # f. ON predictors (positive entries) then OFF predictors (negative entries), globally normalised.
        stim = np.hstack([np.clip(design, 0, None), np.clip(design, None, 0)])
        valid_stim = ~(stim == 0).all(0)
        stim = (stim - stim.mean()) / stim.std(ddof=1)
        if running_regressor and run_design.std(ddof=1) > 0:
            run_design = (run_design - run_design.mean()) / run_design.std(ddof=1)
        n_samples, n_predictors = stim.shape
        X = np.hstack([stim, run_design])

        # Step 3. Regularisation: one smoothness block per (stimulus lambda, running lambda, L1 lambda) combination.
        lam_stim = np.sqrt(lambdas * valid_times.sum() * n_predictors)
        lam_matrix = make_lambda_matrix([len(rf_bins), *stim_size], np.ones(len(stim_size) + 1))  # predictor order
        lam_matrix = block_diag(lam_matrix, lam_matrix)  # ON and OFF smoothed independently
        if running_regressor:
            lam_run = np.sqrt(lambdas * valid_times.sum() * run_design.shape[1])
            lam_matrix_run = make_lambda_matrix([run_design.shape[1]], [1])  # smooth across lags
        else:
            lam_run = np.array([np.nan])  # single dummy value: stimulus-only model
            lam_matrix_run = np.zeros((0, 0))
        l1_grid = list(l1_lambdas) if l1_enabled else [None]  # None: least squares, no L1
        combos = list(product(range(len(lam_stim)), range(len(lam_run)), range(len(l1_grid))))
        penalties = [block_diag(lam_matrix * lam_stim[i], lam_matrix_run * lam_run[k]) for i, k, _ in combos]
        l1_per_combo = [l1_grid[m] for _, _, m in combos]
        zero_pad = np.zeros((len(penalties[0]), valid_units.sum()))

        # Step 4. Cross-validated lambda selection (fold f tests samples f, f + cross_folds, ...).
        # The combination is chosen on the full model; RF inclusion uses the stimulus-only EV at that combination.
        cv_ev_full = np.full((n_neurons, len(combos), cross_folds), np.nan)
        cv_ev_rf = np.full((n_neurons, len(combos), cross_folds), np.nan)
        y = z_traces[:, valid_units]
        for fold in range(cross_folds):
            test = np.zeros(n_samples, bool)
            test[fold::cross_folds] = True
            train = ~test if cross_folds > 1 else test

            y_train = np.vstack([y[train], zero_pad])
            y_mean = y[train].mean(0)
            for p, penalty in enumerate(penalties):
                B = fit(np.vstack([X[train], penalty]), y_train, l1_per_combo[p], l1_max_iter, l1_tol)
                pred_full = X[test] @ B
                pred_rf = stim[test] @ B[:n_predictors]
                cv_ev_full[valid_units, p, fold] = 1 - ((y[test] - pred_full) ** 2).sum(0) / ((y[test] - y_mean) ** 2).sum(0)
                cv_ev_rf[valid_units, p, fold] = 1 - ((y[test] - pred_rf) ** 2).sum(0) / ((y[test] - y_mean) ** 2).sum(0)

        mean_ev_full, mean_ev_rf = cv_ev_full.mean(2), cv_ev_rf.mean(2)  # [neurons x combos]; NaN for invalid neurons
        best_combo = np.zeros(n_neurons, int)
        best_combo[valid_units] = mean_ev_full[valid_units].argmax(1)
        max_ev_full = mean_ev_full[np.arange(n_neurons), best_combo]
        max_ev = mean_ev_rf[np.arange(n_neurons), best_combo]
        best_lambdas = np.where(valid_units, lambdas[[combos[p][0] for p in best_combo]], np.nan)
        best_run_lambdas = np.where(valid_units, lambdas[[combos[p][1] for p in best_combo]], np.nan)
        l1_best_lambdas = np.where(valid_units, l1_lambdas[[combos[p][2] for p in best_combo]], np.nan)

        # Step 5. Final fit on all samples with each neuron's best lambda combination.
        B_all = np.full((X.shape[1], n_neurons), np.nan)
        for p, penalty in enumerate(penalties):
            neurons = (best_combo == p) & valid_units
            if neurons.any():
                b = np.vstack([z_traces[:, neurons], zero_pad[:, : neurons.sum()]])
                B_all[:, neurons] = fit(np.vstack([X, penalty]), b, l1_per_combo[p], l1_max_iter, l1_tol)
        coefficients, running_kernels = B_all[:n_predictors], B_all[n_predictors:]

        # Step 6. Shift test: stimulus-only EV of each RF vs. RFs refit on circularly shifted traces.
        ev = explained_variance(z_traces, stim @ coefficients)
        if not l1_enabled:
            ev_shifted = shift_test(z_traces, stim, run_design, penalties, np.where(valid_units, best_combo, -1),
                                    n_shifts, rng)
        else:
            # As utils.shift_test (same shifts), but each neuron's shifted traces are refit with FISTA.
            n_t = len(z_traces)
            ev_shifted = np.full((n_neurons, n_shifts), np.nan)
            shifts = rng.integers(1, n_t + 1, n_shifts)
            shifted_index = (np.arange(n_t)[:, None] - shifts) % n_t  # column s is np.roll(z, shifts[s])
            for p, penalty in enumerate(penalties):
                neurons = np.flatnonzero((best_combo == p) & valid_units)
                if neurons.size == 0:
                    continue
                A = np.vstack([X, penalty])
                l1_gram = A.T @ A / len(A)  # shared by all neurons with this combination
                l1_lipschitz = np.linalg.eigvalsh(l1_gram)[-1]
                for n in neurons:
                    shifted = z_traces[shifted_index, n]
                    Y = np.vstack([shifted, np.zeros((len(penalty), n_shifts))])
                    B = fit(A, Y, l1_per_combo[p], l1_max_iter, l1_tol, l1_gram, l1_lipschitz)
                    ev_shifted[n] = explained_variance(shifted, stim @ B[:n_predictors])
        p_values = (ev_shifted > ev[:, None]).mean(1)
        p_values[np.isnan(ev)] = np.nan
        valid_rf = np.flatnonzero((p_values < max_p_val) & (max_ev > min_ev))
        print(f"{fov}: {valid_rf.size} of {n_neurons} neurons have a significant RF")

        # Unstimulated predictors -> NaN; maps are [neuron x cols x rows x delays x ON/OFF].
        coefficients[~valid_stim] = np.nan
        maps = coefficients.T.reshape(n_neurons, 2, len(rf_bins), *stim_size).transpose(0, 3, 4, 2, 1)

        # Steps 7-9. Subfield, optimal delay and 2D Gaussian for each significant RF
        best_subfields = np.full(n_neurons, -1, np.int8)  # 0 ON, 1 OFF, 2 ON+OFF, -1 not significant
        subfield_signs = np.full((n_neurons, 2), np.nan)
        optimal_delays = np.full(n_neurons, -1, np.int8)  # index into t_rf, -1 not significant
        gauss_pars = np.full((n_neurons, 6), np.nan)     # [amplitude, x0, sigma_x, y0, sigma_y, rotation]
        peak_to_noise = np.full(n_neurons, np.nan)
        for n in valid_rf:
            # Step 7a. Flip OFF so positive means driven by a black square.
            rf = maps[n].copy()
            rf[..., 1] *= -1

            # b-c. Strongest delay-averaged peak and its sign per subfield.
            mean_rf = rf.mean(2)
            peaks = [mean_rf[..., s].flat[np.nanargmax(np.abs(mean_rf[..., s]))] for s in range(2)]
            signs = np.sign(peaks)

            # d-e. ON+OFF wins if its peak reaches thresh_subfield of the strongest.
            combined = (mean_rf[..., 0] * signs[0] + mean_rf[..., 1] * signs[1]) / 2
            strengths = [abs(peaks[0]), abs(peaks[1]), np.nanmax(combined)]
            subfield = 2 if strengths[2] > thresh_subfield * max(strengths) else int(np.argmax(strengths))
            best_subfields[n], subfield_signs[n] = subfield, signs

            # f. Collapse to the chosen subfield.
            if subfield < 2:
                rf = rf[..., subfield] * signs[subfield]
            else:
                rf = (rf[..., 0] * signs[0] + rf[..., 1] * signs[1]) / 2

            # Step 8. Delay with the highest spatial peak.
            delay = int(np.argmax(np.nanmax(rf, axis=(0, 1))))
            optimal_delays[n] = delay

            # Step 9. Rotated Gaussian on a 1-degree grid (rotation sign flipped).
            rf_deg, x_deg, y_deg = interpolate_rf_to_visual_degrees(rf[:, :, delay], fov_edges)
            pars, fitted = fit_2d_gauss_rf(rf_deg, x_deg, y_deg)
            pars[5] = -pars[5]
            gauss_pars[n] = pars
            peak_to_noise[n] = pars[0] / np.std(rf_deg - fitted, ddof=1)

        good_rf = peak_to_noise > min_peak_to_noise
        print(f"{fov}: {good_rf.sum()} good RFs (peak/noise > {min_peak_to_noise})")

        # Save: ALF object _ss_averageRF, one row per fitted neuron, in a new revision of alf/FOV_XX/<variant>/.
        rf = {
            "mpciROIs": roi_index,  # cross-reference: row of alf/FOV_XX/mpciROIs.*
            "maps": maps,
            "explainedVariance": max_ev,
            "pValues": p_values,
            "lambdas": best_lambdas,
            "bestSubfields": best_subfields,
            "subfieldSigns": subfield_signs,
            "optimalDelays": optimal_delays,
            "gaussPars": gauss_pars,
            "peakToNoise": peak_to_noise,
            "goodRF": good_rf,
        }
        if l1_enabled:
            rf["l1Lambdas"] = l1_best_lambdas
        if running_regressor:
            rf["runningKernels"] = running_kernels.T
            rf["explainedVarianceFullModel"] = max_ev_full
            rf["runLambdas"] = best_run_lambdas
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "inputs": {
                "activity": [f"{fov}/mpci.{a}" for a in (("ROIActivityF", "ROINeuropilActivityF")
                                                         if signal == "traces" else ("ROIActivityDeconvolved",))],
                "times": f"{fov}/mpci.times",
                "rois": f"{fov}/mpciROIs.mpciROITypes == 1",
                "stimulus": ["alf/_ibl_passiveRFM.times", "raw_passive_data/_iblrig_RFMapStim.raw.bin"],
                "wheel": "alf/_ibl_wheel" if running_regressor else None,
            },
            "settings": saved_settings,
            "stimulusCrop": {
                "hemisphere": "right" if right_hemisphere else "left",
                "keptColumns": kept_cols.tolist(),  # original stimulus columns kept, in maps order
                "keptRows": kept_rows.tolist(),     # original stimulus rows kept, in maps order
                "edgesDeg": dict(zip(("left", "right", "bottom", "top"), fov_edges.tolist())),
            },
            "rfDelays": t_rf.tolist(),
            "runLags": t_run.tolist() if running_regressor else None,
            "counts": {"fitted": int(valid_units.sum()), "significant": int(valid_rf.size), "good": int(good_rf.sum())},
        }

        # Revision: today's date; a, b, ... appended for further runs on the same day.
        revision = date.today().isoformat()
        same_day = [r for r in revisions if r.startswith(revision)]
        if same_day:
            revision += ascii_lowercase[len(same_day) - 1]
        folder = session_path / collection / f"#{revision}#"
        folder.mkdir(parents=True)
        save_object_npy(folder, rf, "averageRF", namespace="ss")
        with open(folder / meta_name, "w") as f:
            json.dump(meta, f, indent=1)
        print(f"{fov}: saved results to {folder}")
        results[fov] = folder / meta_name

        
    # Documentation of the output datasets, written to one_root/dataset_types.json
    # Rows of every _ss_averageRF.*.npy are the fitted neurons: ROIs with mpciROITypes == 1 (traces: also non-zero F).
    # "Not fitted" = more than max_nan_fraction NaN samples; "significant" = pValues < max_p_val and
    # explainedVariance > min_ev (thresholds in meta.json settings).
    DATASET_DOCS = {
        "_ss_averageRF.mpciROIs.npy": """Dataset: _ss_averageRF.mpciROIs.npy
    Units: N/A
    Type: int64
    Dimensions: [nNeurons]
    Format: npy
    Description: Index of each fitted neuron in the FOV's mpciROIs datasets (cross-reference). Row n of every other _ss_averageRF dataset is ROI mpciROIs[n] of alf/FOV_XX.
    0-based row of alf/FOV_XX/mpciROIs.* and column of alf/FOV_XX/mpci.ROIActivity*.""",

        "_ss_averageRF.maps.npy": """Dataset: _ss_averageRF.maps.npy
    Units: z-scored activity per unit of the normalised stimulus predictor
    Type: float64
    Dimensions: [nNeurons, nColumns, nRows, nDelays, 2]
    Format: npy
    Description: Receptive field of each neuron: regression coefficients (ridge, plus L1 with settings l1_enabled) per stimulus pixel, delay and subfield.
    Axis 1 is the stimulus column (0 = leftmost kept column), axis 2 the row (0 = lowest kept row), axis 3 the delay
    (seconds in meta.json rfDelays), axis 4 the subfield (0 = ON/white, 1 = OFF/black). OFF keeps the sign of the fit:
    negative means the response increases to black. Only the columns and rows kept by the stimulus crop are present
    (meta.json stimulusCrop keptColumns/keptRows; all of them without a crop). Pixel centres in degrees follow from
    meta.json stimulusCrop edgesDeg: azimuth = left + (column + 0.5) * (right - left) / nColumns, likewise for elevation.
    NaN for pixels/delays never stimulated and for neurons that were not fitted.""",

        "_ss_averageRF.explainedVariance.npy": """Dataset: _ss_averageRF.explainedVariance.npy
    Units: fraction of variance
    Type: float64
    Dimensions: [nNeurons]
    Format: npy
    Description: Cross-validated explained variance of the stimulus part of the model, at the chosen ridge penalty.
    Mean over cross-validation folds; can be negative. With the running regressor the penalty is chosen on the
    full model, but this value is still the stimulus-only EV. NaN for neurons that were not fitted.""",

        "_ss_averageRF.pValues.npy": """Dataset: _ss_averageRF.pValues.npy
    Units: probability
    Type: float64
    Dimensions: [nNeurons]
    Format: npy
    Description: Significance of each RF: fraction of circular-shift refits whose stimulus explained variance is above the real one.
    Resolution is 1/n_shifts (meta.json settings); 0 means p < 1/n_shifts. NaN for neurons that were not fitted.""",

        "_ss_averageRF.lambdas.npy": """Dataset: _ss_averageRF.lambdas.npy
    Units: N/A
    Type: float64
    Dimensions: [nNeurons]
    Format: npy
    Description: Ridge penalty chosen for the stimulus predictors of each neuron by cross-validation.
    One of meta.json settings lambdas. NaN for neurons that were not fitted.""",

        "_ss_averageRF.bestSubfields.npy": """Dataset: _ss_averageRF.bestSubfields.npy
    Units: N/A
    Type: int8
    Dimensions: [nNeurons]
    Format: npy
    Description: Subfield that best describes each significant RF.
    0 = ON, 1 = OFF, 2 = ON+OFF (chosen when the combined peak reaches thresh_subfield of the strongest),
    -1 = not significant.""",

        "_ss_averageRF.subfieldSigns.npy": """Dataset: _ss_averageRF.subfieldSigns.npy
    Units: N/A
    Type: float64
    Dimensions: [nNeurons, 2]
    Format: npy
    Description: Sign of the delay-averaged ON and OFF peaks of each significant RF.
    Columns are ON, OFF; +1 or -1, with OFF flipped so that +1 means driven by black. NaN = not significant.""",

        "_ss_averageRF.optimalDelays.npy": """Dataset: _ss_averageRF.optimalDelays.npy
    Units: N/A
    Type: int8
    Dimensions: [nNeurons]
    Format: npy
    Description: Delay at which each significant RF has its highest spatial peak, in the chosen subfield.
    Index into meta.json rfDelays (seconds) and axis 3 of maps; -1 = not significant.""",

        "_ss_averageRF.gaussPars.npy": """Dataset: _ss_averageRF.gaussPars.npy
    Units: see columns
    Type: float64
    Dimensions: [nNeurons, 6]
    Format: npy
    Description: 2D Gaussian fitted to each significant RF.
    Columns: amplitude (map units), x0 (azimuth, degrees), sigma_x (degrees), y0 (elevation, degrees),
    sigma_y (degrees), rotation (radians, sign flipped). Fitted to the map of the chosen subfield at
    the optimal delay, sign-corrected so the peak is positive and interpolated to a 1-degree grid. NaN = not significant.""",

        "_ss_averageRF.peakToNoise.npy": """Dataset: _ss_averageRF.peakToNoise.npy
    Units: N/A
    Type: float64
    Dimensions: [nNeurons]
    Format: npy
    Description: Peak-to-noise ratio of each significant RF: Gaussian amplitude divided by the standard deviation of the fit residual.
    Computed on the 1-degree map used for gaussPars. NaN = not significant.""",

        "_ss_averageRF.goodRF.npy": """Dataset: _ss_averageRF.goodRF.npy
    Units: N/A
    Type: bool
    Dimensions: [nNeurons]
    Format: npy
    Description: Whether each RF is good: significant and peakToNoise > min_peak_to_noise (meta.json settings).""",

        "_ss_averageRF.runningKernels.npy": """Dataset: _ss_averageRF.runningKernels.npy
    Units: z-scored activity per z-scored running speed
    Type: float64
    Dimensions: [nNeurons, nLags]
    Format: npy
    Description: Running-speed coefficients of each neuron per lag; only saved with the running regressor.
    Lags in seconds are meta.json runLags. NaN for neurons that were not fitted.""",

        "_ss_averageRF.explainedVarianceFullModel.npy": """Dataset: _ss_averageRF.explainedVarianceFullModel.npy
    Units: fraction of variance
    Type: float64
    Dimensions: [nNeurons]
    Format: npy
    Description: Cross-validated explained variance of the stimulus + running model; only saved with the running regressor.
    The value used to choose the ridge penalties. NaN for neurons that were not fitted.""",

        "_ss_averageRF.runLambdas.npy": """Dataset: _ss_averageRF.runLambdas.npy
    Units: N/A
    Type: float64
    Dimensions: [nNeurons]
    Format: npy
    Description: Ridge penalty chosen for the running predictors of each neuron; only saved with the running regressor.
    One of meta.json settings lambdas. NaN for neurons that were not fitted.""",

        "_ss_averageRF.l1Lambdas.npy": """Dataset: _ss_averageRF.l1Lambdas.npy
    Units: N/A
    Type: float64
    Dimensions: [nNeurons]
    Format: npy
    Description: L1 penalty chosen for each neuron by cross-validation; only saved with settings l1_enabled.
    One of meta.json settings l1_lambdas. The model then minimises ||A B - Y||^2 / (2 N) + l1Lambda * sum|B|,
    with A the design plus smoothness rows and N its number of rows (FISTA). NaN for neurons that were not fitted.""",

        "_ss_averageRFRun.meta.json": """Dataset: _ss_averageRFRun.meta.json
    Units: N/A
    Type: dict
    Dimensions: N/A
    Format: json
    Description: Metadata of one RF-mapping run.
    Keys: created (date and time of the run); inputs (datasets used); settings (all settings after defaults,
    stimulus_edges_deg and sparse_noise_shape as named fields); rfDelays (seconds after each stimulus frame, one per
    maps delay); runLags (seconds, one per runningKernels column, null without the running regressor);
    stimulusCrop (hemisphere of the FOV, original stimulus columns and rows kept, edges of the kept area in degrees);
    counts (fitted, significant and good neurons).""",
    }


        # Document the output datasets next to the converter's, and re-index the ONE root so One instances created
        # from now on find the results; `one` keeps its old snapshot.

    update_dataset_types(one.cache_dir, DATASET_DOCS)
    make_parquet_db(one.cache_dir)
    one.load_cache()
    return results
