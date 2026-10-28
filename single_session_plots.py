"""Plots of the average receptive-field mapping of one session (ALF object _ss_averageRF, written by map_average_rf).

plot_session: plots of one RF-mapping run, each FOV on its own and all FOVs pooled, saved as PNG and SVG in
<plot root>/FOV_XX/<rf_traces|rf_deconv>/<revision>/ and
<plot root>/session/<rf_traces|rf_deconv>/<newest revision among the FOVs>/.
The plot root is <output_dir>/<subject>/<date>/<number> (settings["output_dir"], required).
"""
import shutil
from math import ceil, floor
from pathlib import Path

import numpy as np
import sparse
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from one.alf.io import load_object
from one.alf.path import ALFPath
from pycolormap_2d import ColorMap2DBremm

from utils import estimate_neuropil_lfr, explained_variance, gaussian_smooth, interp_columns, make_stim_toeplitz

rcParams["svg.fonttype"] = "none"  # SVG text stays editable text

DEFAULT_PLOT_SETTINGS = {
    "signal": "latest",            # "traces", "deconvolved", or "latest" (signal of the session's most recent run)
    "revision": "latest",          # "latest", or a revision folder name such as "2026-10-26" or "2026-10-26a"
    "output_dir": None,            # plots go to <output_dir>/<subject>/<date>/<number>/ (required)
    "azimuth_range_contra_deg": None,  # (ipsilateral, contralateral) azimuth shown, in degrees towards the FOV's
                                       # contralateral side (0 = midline), e.g. (-10, 90); None = whole screen, or
                                       # no limit on that side. Zooms coverage and RF maps; the RF ROI overlay
                                       # shows only RFs centred in this range, coloured across it
    # Plots made per FOV and for the pooled session
    "gaussian_coverage": True,     # summed Gaussian RFs over the visual field, with and without ellipses
    "rf_size": True,               # RF size of good RFs
    "explained_variance": True,    # RF explained variance, p < max_p_val vs. the rest (session: rebuilds the traces)
    "roi_overlays": True,          # ROIs on the mean image (session: neurons, filled and outlined)
    "rf_roi_overlay": True,        # neurons coloured by RF centre on the mean image; also good RFs only
    # Plots made per FOV only
    "rf_maps": True,               # one RF map per neuron
    "n_rf_maps_good": 10,          # good RFs with the highest peak/noise; None = all
    "n_rf_maps_significant": 10,   # significant but not good RFs with the highest peak/noise; None = all
    "predicted_vs_actual": True,   # traces vs. model prediction: 5 best good RFs and the neuron with median EV
    "peak_time_courses": True,     # RF time course at the Gaussian centre of the 5 best good RFs
}

META_NAME = "_ss_averageRFRun.meta.json"
VARIANTS = {"traces": "rf_traces", "deconvolved": "rf_deconv"}
SUBFIELD_NAMES = ("ON", "OFF", "ON+OFF")
# Per-neuron RF maps: negative light gray, zero white, positive green to dark green
RF_MAP_COLORS = LinearSegmentedColormap.from_list(
    "rf_map", [(0, "#b8b8b8"), (0.5, "#ffffff"), (0.83, "#00ff00"), (1, "#126d36")])
AZIMUTH_LABEL = "Azimuth (deg, contralateral > 0)"


def load_runs(one, session_path, signal="latest", revision="latest"):
    """Load the RF-mapping run of each FOV of the session, chosen by signal and revision.

    signal: "traces" or "deconvolved" takes rf_traces or rf_deconv; "latest" takes the signal of the most
        recently created run in the session, so that every FOV has the same signal.
    revision: "latest" takes the last revision of each FOV; a revision folder name takes, in each FOV, the
        last revision up to that name (ONE's rule). FOVs without one are skipped.
    Only the files of the chosen revision are loaded (ONE would fill in files missing from it, such as
    runningKernels, from older revisions). FOVs mapped as right hemisphere (meta.json stimulusCrop) are mirrored,
    so azimuth > 0 is contralateral.

    Returns (signal, runs): runs has one dict per FOV (see load_run).
    """
    fovs = sorted(one.list_collections(session_path, collection="alf/FOV_??"))

    def chosen_revision(collection):
        revisions = sorted(one.list_revisions(session_path, collection=collection, filename=META_NAME))
        if revision != "latest":
            revisions = [r for r in revisions if r <= revision]
        return revisions[-1] if revisions else None

    if signal == "latest":
        created = {}  # signal -> creation time of its newest chosen run
        for candidate, variant in VARIANTS.items():
            for fov in fovs:
                fov_revision = chosen_revision(f"{fov}/{variant}")
                if fov_revision:
                    meta = one.load_dataset(session_path, META_NAME, collection=f"{fov}/{variant}",
                                            revision=fov_revision)
                    created[candidate] = max(created.get(candidate, ""), meta["created"])
        if not created:
            raise FileNotFoundError(f"No RF-mapping results in {session_path}")
        signal = max(created, key=created.get)
    if signal not in VARIANTS:
        raise ValueError(f'signal must be "traces", "deconvolved" or "latest", got {signal!r}')

    runs = []
    for fov in fovs:
        fov_revision = chosen_revision(f"{fov}/{VARIANTS[signal]}")
        if fov_revision is None:
            print(f"{fov}: no {signal} results up to revision {revision}; skipped")
            continue
        runs.append(load_run(one, session_path, fov, signal, fov_revision))
    return signal, runs


def load_run(one, session_path, fov, signal, revision):
    """Load one RF-mapping run: the _ss_averageRF object of fov ("alf/FOV_00") / rf_<signal> / #revision#.

    Only the files of that revision are loaded (ONE would fill in files missing from it, such as runningKernels,
    from older revisions). A FOV mapped as right hemisphere (meta.json stimulusCrop) is mirrored, so azimuth > 0 is
    contralateral.

    Returns a dict with fov, name ("FOV_00"), signal, revision, rf (the _ss_averageRF object), meta, settings (of
    the run), maps, gauss and edges (mirrored for the right hemisphere; edges are left, right, bottom, top),
    significant, good, thresholds (text for titles).
    """
    files = one.load_object(session_path, "averageRF", collection=f"{fov}/{VARIANTS[signal]}", revision=revision,
                            download_only=True)
    rf = load_object([f for f in files if f.revision == revision])
    meta = one.load_dataset(session_path, META_NAME, collection=f"{fov}/{VARIANTS[signal]}", revision=revision)
    run = meta["settings"]

    # Right hemisphere: mirror azimuth (maps columns, edges, Gaussian centre and rotation).
    maps = rf["maps"]  # [neuron x cols x rows x delays x ON/OFF]
    gauss = rf["gaussPars"].copy()  # [amplitude, x0, sigma_x, y0, sigma_y, rotation]
    edges = meta["stimulusCrop"]["edgesDeg"]
    edges = (edges["left"], edges["right"], edges["bottom"], edges["top"])
    if meta["stimulusCrop"]["hemisphere"] == "right":
        maps = maps[:, ::-1]
        edges = (-edges[1], -edges[0], edges[2], edges[3])
        gauss[:, 1] *= -1
        gauss[:, 5] *= -1

    return {
        "fov": fov, "name": fov.split("/")[-1], "signal": signal, "revision": revision, "rf": rf, "meta": meta,
        "settings": run, "maps": maps, "gauss": gauss, "edges": edges,
        "significant": (rf["pValues"] < run["max_p_val"]) & (rf["explainedVariance"] > run["min_ev"]),
        "good": rf["goodRF"],
        "thresholds": (f"significant: p < {run['max_p_val']:g}, EV > {run['min_ev']:g}; "
                       f"good: peak/noise > {run['min_peak_to_noise']:g}"),
    }


def azimuth_window(edges, azimuth_range_contra):
    """Azimuth limits (left, right) shown: the screen edges, cut to (ipsilateral, contralateral) degrees if given.

    Plots are mirrored so that azimuth > 0 is contralateral, so the range applies to them directly.
    """
    left, right = edges[0], edges[1]
    if azimuth_range_contra is not None:
        ipsi, contra = azimuth_range_contra
        left = left if ipsi is None else max(left, ipsi)
        right = right if contra is None else min(right, contra)
    if left >= right:
        raise ValueError(f"azimuth_range_contra_deg {azimuth_range_contra} leaves nothing of the screen {edges[:2]}")
    return left, right


def plot_root(session_path, output_dir):
    """Plot folder of a session: <output_dir>/<subject>/<date>/<number>."""
    return Path(output_dir) / session_path.session_path_short()


def load_fov_image(one, session_path, fov):
    """Mean image of a FOV scaled to 0-1 (20th to 99.8th percentile), and its mpciROIs masks and types."""
    image = one.load_dataset(session_path, "mpciMeanImage.images.npy", collection=fov)[:, :, 0, 0]
    rois = one.load_object(session_path, "mpciROIs", collection=fov, attribute=["masks", "mpciROITypes"])
    low, high = np.percentile(image, [20, 99.8])
    return np.clip((image - low) / (high - low), 0, 1), rois


def plot_session(one, session_path, settings):
    """Plot one RF-mapping run of the session at session_path: each FOV on its own, then all FOVs pooled.

    settings override DEFAULT_PLOT_SETTINGS; output_dir is required. signal and revision choose the run (see
    load_runs). A plot made both
    per FOV and for the session has one flag for both. Thresholds (max_p_val, min_ev, min_peak_to_noise) are those
    of the run (meta.json). FOVs mapped as right hemisphere (meta.json) are mirrored, so azimuth > 0 is
    contralateral in every plot.

    Returns {FOV collection or "session": plot folder}.
    """
    settings = {**DEFAULT_PLOT_SETTINGS, **settings}
    if not settings["output_dir"]:
        raise ValueError('settings["output_dir"] is required')
    session_path = ALFPath(session_path)
    signal, runs = load_runs(one, session_path, settings["signal"], settings["revision"])
    if not runs:
        raise FileNotFoundError(f"No {signal} results in {session_path} up to revision {settings['revision']}")

    # %% Each FOV
    plot_folders = {}
    for r in runs:
        fov, revision, rf, meta = r["fov"], r["revision"], r["rf"], r["meta"]
        maps, gauss, edges, significant, good = r["maps"], r["gauss"], r["edges"], r["significant"], r["good"]
        thresholds = r["thresholds"]
        azimuth_limits = azimuth_window(edges, settings["azimuth_range_contra_deg"])
        by_peak_to_noise = np.argsort(-rf["peakToNoise"])  # NaN (not significant) last
        good_ranked = [n for n in by_peak_to_noise if good[n]]
        significant_ranked = [n for n in by_peak_to_noise if significant[n] and not good[n]]

        folder = plot_root(session_path, settings["output_dir"]) / r["name"] / VARIANTS[signal] / revision
        if folder.exists():
            shutil.rmtree(folder)  # old plots of this run, e.g. RF maps of neurons no longer selected
        folder.mkdir(parents=True)
        title = (f"{session_path.session_path_short()} {r['name']} {signal} #{revision}# "
                 f"({meta['stimulusCrop']['hemisphere']} hemisphere)")
        print(f"{fov}: plotting {fov}/{VARIANTS[signal]}/#{revision}# "
              f"({significant.sum()} significant, {good.sum()} good RFs)")

        if settings["gaussian_coverage"]:
            plot_gaussian_coverage(folder, title, thresholds, gauss, significant, good, edges, azimuth_limits)
        if settings["rf_size"]:
            plot_rf_size_histogram(folder, title, thresholds, gauss[good])
        if settings["explained_variance"]:
            plot_ev_histogram(folder, title, rf["explainedVariance"], rf["pValues"], r["settings"]["max_p_val"])
        if settings["rf_maps"]:
            for kind, neurons in (("good", good_ranked[:settings["n_rf_maps_good"]]),
                                  ("significant", significant_ranked[:settings["n_rf_maps_significant"]])):
                for n in neurons:
                    plot_rf_map(folder / "rfMaps" / kind, title, n, rf, maps, gauss, edges, azimuth_limits, meta)
        if settings["predicted_vs_actual"] or settings["explained_variance"]:
            # Traces as fit, rebuilt once; the stimulus EV over all samples (as in the shift test) is also pooled below.
            t_samples, z_traces, prediction, stim_prediction = model_traces(one, session_path, fov, rf, meta)
            r["ev_all"] = ev_all = explained_variance(z_traces, stim_prediction)
        if settings["predicted_vs_actual"]:
            median_ev = int(np.nanargmin(np.abs(ev_all - np.nanmedian(ev_all))))
            neurons = good_ranked[:5] + ([median_ev] if median_ev not in good_ranked[:5] else [])
            for n in neurons:
                note = "neuron with median EV" if n == median_ev else "best peak/noise"
                plot_predicted_vs_actual(folder / "predictedVsActual", title, note, n, rf, t_samples, z_traces,
                                         prediction, ev_all)
        if settings["peak_time_courses"] and good_ranked:
            plot_peak_time_courses(folder, title, good_ranked[:5], rf, maps, gauss, edges, meta)
        if settings["roi_overlays"] or settings["rf_roi_overlay"]:
            r["image"], r["rois"] = image, rois = load_fov_image(one, session_path, fov)  # also pooled below
            if settings["roi_overlays"]:
                plot_roi_overlays(folder, title, image, rois["masks"], rois["mpciROITypes"])
            if settings["rf_roi_overlay"]:
                for good_only in (False, True):
                    plot_rf_roi_overlay(folder, title, image, rois["masks"], rf["mpciROIs"], gauss, significant,
                                        good, edges, azimuth_limits, good_only)
        plot_folders[fov] = folder

    # %% All FOVs pooled
    folder = (plot_root(session_path, settings["output_dir"]) / "session" / VARIANTS[signal]
              / max(r["revision"] for r in runs))
    if folder.exists():
        shutil.rmtree(folder)  # old plots of this run
    folder.mkdir(parents=True)
    title = (f"{session_path.session_path_short()} session {signal} ("
             + ", ".join(f"{r['name']} #{r['revision']}# {r['meta']['stimulusCrop']['hemisphere']}" for r in runs)
             + ")")
    thresholds = " | ".join(sorted({r["thresholds"] for r in runs}))  # one entry unless the runs' thresholds differ
    print(f"session: plotting {', '.join(r['name'] for r in runs)} -> {folder}")

    # Pooled RFs; the visual field spans the kept area of every FOV.
    gauss = np.vstack([r["gauss"] for r in runs])
    significant = np.concatenate([r["significant"] for r in runs])
    good = np.concatenate([r["good"] for r in runs])
    edges = (min(r["edges"][0] for r in runs), max(r["edges"][1] for r in runs),
             min(r["edges"][2] for r in runs), max(r["edges"][3] for r in runs))
    azimuth_limits = azimuth_window(edges, settings["azimuth_range_contra_deg"])

    if settings["gaussian_coverage"]:
        plot_gaussian_coverage(folder, title, thresholds, gauss, significant, good, edges, azimuth_limits)
    if settings["rf_size"]:
        plot_rf_size_line(folder, title, thresholds, gauss[good])
    if settings["explained_variance"]:
        is_significant = np.concatenate([r["rf"]["pValues"] < r["settings"]["max_p_val"] for r in runs])
        p_threshold = " / ".join(sorted({f"{r['settings']['max_p_val']:g}" for r in runs}))
        plot_ev_lines(folder, title, np.concatenate([r["ev_all"] for r in runs]), is_significant, p_threshold)
    if settings["roi_overlays"] or settings["rf_roi_overlay"]:
        # Mean images side by side; ROI masks and indices shifted to match.
        fov_labels, roi_offsets, coords, weights, roi_types = [], [], [], [], []
        x_offset, roi_offset = 0, 0
        for r in runs:
            fov_labels.append((x_offset, r["image"].shape[1], r["name"]))
            roi_offsets.append(roi_offset)
            coords.append(r["rois"]["masks"].coords + np.array([[roi_offset], [0], [x_offset]]))
            weights.append(r["rois"]["masks"].data)
            roi_types.append(r["rois"]["mpciROITypes"])
            x_offset += r["image"].shape[1]
            roi_offset += len(r["rois"]["mpciROITypes"])
        height = max(r["image"].shape[0] for r in runs)
        mosaic = np.zeros((height, x_offset))
        for r, (x, width, _) in zip(runs, fov_labels):
            mosaic[:r["image"].shape[0], x:x + width] = r["image"]
        masks = sparse.COO(np.hstack(coords), np.concatenate(weights), shape=(roi_offset, height, x_offset))
        figsize = (min(24, 4 + 8 * x_offset / height), 8)
        if settings["roi_overlays"]:
            plot_neuron_overlays(folder, title, mosaic, masks, np.flatnonzero(np.concatenate(roi_types) == 1),
                                 fov_labels, figsize)
        if settings["rf_roi_overlay"]:
            roi_index = np.concatenate([r["rf"]["mpciROIs"] + offset for r, offset in zip(runs, roi_offsets)])
            for good_only in (False, True):
                plot_rf_roi_overlay(folder, title, mosaic, masks, roi_index, gauss, significant, good, edges,
                                    azimuth_limits, good_only, fov_labels, figsize)
    plot_folders["session"] = folder
    return plot_folders


def save(fig, folder, name):
    """Save fig as folder/name.png and folder/name.svg (no dots in name: not an ALF dataset)."""
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{name}.{extension}", dpi=200)


def chosen_rf(rf, maps, n):
    """RF of neuron n in its chosen subfield, sign-corrected so the peak is positive [cols x rows x delays]."""
    subfield, signs = rf["bestSubfields"][n], rf["subfieldSigns"][n]
    on, off = maps[n, ..., 0], -maps[n, ..., 1]  # OFF flipped: positive = driven by black
    if subfield == 0:
        return on * signs[0]
    if subfield == 1:
        return off * signs[1]
    return (on * signs[0] + off * signs[1]) / 2


def draw_ellipses(ax, gauss, **style):
    """2-sigma ellipses of Gaussians [amplitude, x0, sigma_x, y0, sigma_y, rotation]."""
    t = np.linspace(0, 2 * np.pi, 100)
    for _, x0, sx, y0, sy, rotation in gauss:
        u, v = 2 * sx * np.cos(t), 2 * sy * np.sin(t)
        ax.plot(x0 + u * np.cos(rotation) - v * np.sin(rotation),
                y0 + u * np.sin(rotation) + v * np.cos(rotation), **style)


def plot_gaussian_coverage(folder, title, thresholds, gauss, significant, good, edges, azimuth_limits):
    """1. Each RF as its Gaussian (amplitude 1, cut at 2 sigma), summed over the visual field.

    Top row with the 2-sigma ellipses, bottom row the same without; azimuth_limits zooms the x axis.
    """
    left, right, bottom, top = edges
    azimuth, elevation = np.meshgrid(np.linspace(left, right, 500), np.linspace(bottom, top, 250))

    def coverage(neurons):
        total = np.zeros_like(azimuth)
        for _, x0, sx, y0, sy, rotation in gauss[neurons]:
            if sx <= 0 or sy <= 0:
                continue
            dx, dy = azimuth - x0, elevation - y0
            u = (dx * np.cos(rotation) + dy * np.sin(rotation)) / sx
            v = (-dx * np.sin(rotation) + dy * np.cos(rotation)) / sy
            total += np.where(u ** 2 + v ** 2 <= 4, np.exp(-(u ** 2 + v ** 2) / 2), 0)
        return total

    panels = [(f"Good RFs (n = {good.sum()})", good), (f"All significant RFs (n = {significant.sum()})", significant)]
    coverages = [coverage(neurons) for _, neurons in panels]
    vmax = max(c.max() for c in coverages) or 1

    fig = Figure(figsize=(14, 8), layout="constrained")
    for row, with_ellipses in zip(fig.subplots(2, 2), (True, False)):
        for ax, (name, neurons), c in zip(row, panels, coverages):
            image = ax.imshow(c, origin="lower", extent=edges, cmap="viridis", vmin=0, vmax=vmax)
            if with_ellipses:
                draw_ellipses(ax, gauss[neurons & ~good], color="#dddddd", linestyle="--", linewidth=0.8)
                draw_ellipses(ax, gauss[neurons & good], color="white", linewidth=1.1)
                if neurons is significant:
                    name += "; dashed: not good"
            ax.axvline(0, color="#333333", linewidth=0.75)
            ax.axhline(0, color="#333333", linewidth=0.75)
            ax.set(xlim=azimuth_limits, ylim=(bottom, top), xlabel=AZIMUTH_LABEL, ylabel="Elevation (deg)",
                   title=name if with_ellipses else f"{name}, no ellipses")
            fig.colorbar(image, ax=ax, label="Summed Gaussian RFs")
    fig.suptitle(f"{title}: Gaussian RF coverage\n{thresholds}")
    save(fig, folder, "gaussianCoverage")


def plot_rf_size_histogram(folder, title, thresholds, gauss_good):
    """2. RF size of good RFs: (|sigma_x| + |sigma_y|) / 2, 2-degree bins."""
    size = (np.abs(gauss_good[:, 2]) + np.abs(gauss_good[:, 4])) / 2
    fig = Figure(figsize=(6.4, 6.4), layout="constrained")
    ax = fig.subplots()
    if size.size:
        ax.hist(size, bins=np.arange(0, size.max() + 2, 2), color="0.75", edgecolor="0.45")
        ax.axvline(size.mean(), color="black", linewidth=2, label="mean")
        ax.legend(frameon=False)
        stats = f"n = {size.size}\nmean = {size.mean():.2f} deg\nmedian = {np.median(size):.2f} deg"
    else:
        stats = "n = 0: no good RFs"
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="RF size (deg) = (|sigma_x| + |sigma_y|) / 2", ylabel="Number of neurons")
    ax.set_title(f"{title}: good-RF size\n{thresholds}", fontsize=9)
    save(fig, folder, "rfSizeHistogram")


def plot_ev_histogram(folder, title, ev, p_values, max_p_val):
    """3. Cross-validated RF explained variance, p < max_p_val vs. the rest, each in % of its group."""
    fitted = ~np.isnan(ev)
    groups = [(f"p < {max_p_val:g}", fitted & (p_values < max_p_val), "-"),
              (f"p >= {max_p_val:g}", fitted & (p_values >= max_p_val), "--")]
    bins = np.linspace(ev[fitted].min(), ev[fitted].max(), 40)
    fig = Figure(figsize=(6.4, 6.4), layout="constrained")
    ax = fig.subplots()
    for name, neurons, linestyle in groups:
        if neurons.any():
            ax.hist(ev[neurons], bins=bins, weights=np.full(neurons.sum(), 100 / neurons.sum()), histtype="step",
                    color="0.35", linestyle=linestyle, linewidth=1.5, label=f"{name} (n = {neurons.sum()})")
    ax.axvline(ev[fitted].mean(), color="black", linewidth=2, label="mean, all neurons")
    ax.legend(frameon=False)
    ax.set(xlabel="Cross-validated RF explained variance", ylabel="Neurons (% of group)")
    ax.set_title(f"{title}\nRF explained variance", fontsize=9)
    save(fig, folder, "evHistogram")


def plot_rf_map(folder, title, n, rf, maps, gauss, edges, azimuth_limits, meta):
    """4. RF of neuron n in its chosen subfield at its optimal delay, with the 2-sigma Gaussian ellipse.

    azimuth_limits zooms the x axis.
    """
    delay = rf["optimalDelays"][n]
    frame = chosen_rf(rf, maps, n)[:, :, delay]  # [cols x rows]
    peak = np.nanmax(np.abs(frame))
    roi = rf["mpciROIs"][n]
    shown_aspect = (azimuth_limits[1] - azimuth_limits[0]) / (edges[3] - edges[2])  # width / height in degrees
    fig = Figure(figsize=(min(9, 2.5 + 3.4 * shown_aspect), 4.5), layout="constrained")  # narrower when zoomed
    ax = fig.subplots()
    image = ax.imshow(frame.T, origin="lower", extent=edges, cmap=RF_MAP_COLORS,
                      norm=TwoSlopeNorm(vmin=-peak, vcenter=0, vmax=peak))
    draw_ellipses(ax, gauss[[n]], color="black", linewidth=1.5)
    ax.set(xlim=azimuth_limits, ylim=edges[2:], xlabel=AZIMUTH_LABEL, ylabel="Elevation (deg)")
    fig.colorbar(image, ax=ax, label="RF amplitude (sign-corrected)")
    ax.set_title(f"{title}\nROI {roi} | {SUBFIELD_NAMES[rf['bestSubfields'][n]]} | "
                 f"delay {meta['rfDelays'][delay]:.3f} s | EV {rf['explainedVariance'][n]:.3f}, "
                 f"p {rf['pValues'][n]:.3f}, peak/noise {rf['peakToNoise'][n]:.2f}", fontsize=9)
    save(fig, folder, f"roi{roi:04d}")


def model_traces(one, session_path, fov, rf, meta):
    """Rebuild the traces the run was fit to and the model's prediction of them.

    Repeats the stimulus and steps 1-2 of map_average_rf with the run's settings (meta.json), so it must be kept in
    step with that code; the rebuilt neurons are checked against the saved mpciROIs.
    Returns t_samples, z_traces [samples x neurons], prediction (stimulus + running model) and stim_prediction
    (stimulus part only). Never-stimulated predictors have NaN coefficients in maps and are left out.
    """
    run = meta["settings"]
    n_rows, n_cols = run["sparse_noise_shape"]["rows"], run["sparse_noise_shape"]["columns"]

    # Stimulus and step 1a: design [time x (delays * kept pixels)]
    t_stim = one.load_dataset(session_path, "_ibl_passiveRFM.times.npy", collection="alf")
    raw_path = one.load_dataset(session_path, "_iblrig_RFMapStim.raw.bin", collection="raw_passive_data")
    frames = np.fromfile(raw_path, dtype=np.int8).reshape(len(t_stim), n_cols, n_rows)
    frames = np.select([frames == -1, frames == 0], [-1.0, 1.0], 0.0)  # [frame x cols x rows]
    frame_dt = np.median(np.diff(t_stim))
    rf_time_limits = run["rf_time_limits"]
    rf_bins = np.arange(floor(rf_time_limits[0] / frame_dt), ceil(rf_time_limits[1] / frame_dt) + 1)
    toeplitz, t_toeplitz = make_stim_toeplitz(frames, t_stim, rf_bins)
    kept_pixels = np.zeros((n_cols, n_rows), bool)
    kept_pixels[np.ix_(meta["stimulusCrop"]["keptColumns"], meta["stimulusCrop"]["keptRows"])] = True
    fov_toeplitz = toeplitz[:, np.tile(kept_pixels.ravel(), len(rf_bins))]

    # Step 1b: running design [time x lags]
    if run["running_regressor"]:
        wheel = one.load_object(session_path, "wheel", collection="alf")
        model_dt = np.median(np.diff(t_toeplitz))
        run_time_limits = run["run_time_limits"]
        run_bins = np.arange(floor(run_time_limits[0] / model_dt), ceil(run_time_limits[1] / model_dt) + 1)
        speed = np.interp(t_toeplitz, wheel.timestamps, np.abs(wheel.velocity), left=np.nan, right=np.nan)
        speed = (speed - np.nanmean(speed)) / np.nanstd(speed, ddof=1)
        running = np.zeros((len(speed), len(run_bins)))
        for r, lag in enumerate(run_bins):
            source = np.arange(len(speed)) + lag
            inside = (source >= 0) & (source < len(speed))
            running[inside, r] = speed[source[inside]]
        running = np.nan_to_num(running)
    else:
        running = np.zeros((len(t_toeplitz), 0))

    # Step 2: calcium traces
    mpci = one.load_object(session_path, "mpci", collection=fov)
    rois = one.load_object(session_path, "mpciROIs", collection=fov, attribute="mpciROITypes")
    if run["signal"] == "traces":
        roi_index = np.flatnonzero((rois.mpciROITypes == 1) & mpci.ROIActivityF.any(0))
        F = gaussian_smooth(mpci.ROIActivityF[:, roi_index].astype(float), run["smoothing_sigma"])
        Fneu = gaussian_smooth(mpci.ROINeuropilActivityF[:, roi_index].astype(float), run["smoothing_sigma"])
        ca_traces = estimate_neuropil_lfr(F, Fneu)
    else:
        roi_index = np.flatnonzero(rois.mpciROITypes == 1)
        ca_traces = mpci.ROIActivityDeconvolved[:, roi_index].astype(float)
    if not np.array_equal(roi_index, rf["mpciROIs"]):
        raise ValueError(f"{fov}: rebuilt neurons differ from the run's mpciROIs; data or code changed since the run")
    in_window = (mpci.times > t_stim[0] - run["padding"]) & (mpci.times < t_stim[-1] + run["padding"])
    ca_traces = interp_columns(mpci.times[in_window], ca_traces[in_window], t_toeplitz)
    z_traces = (ca_traces - np.nanmean(ca_traces, 0)) / np.nanstd(ca_traces, 0, ddof=1)
    valid_times = ~np.isnan(z_traces).all(1)
    z_traces, design, run_design = z_traces[valid_times], fov_toeplitz[valid_times], running[valid_times]
    few_nans = np.isnan(z_traces).mean(0) <= run["max_nan_fraction"]
    z_traces[:, few_nans] = np.nan_to_num(z_traces[:, few_nans])
    blank = (design == 0).all(1)
    valid_times[np.flatnonzero(valid_times)[blank]] = False
    z_traces, design, run_design = z_traces[~blank], design[~blank], run_design[~blank]
    stim = np.hstack([np.clip(design, 0, None), np.clip(design, None, 0)])
    stim = (stim - stim.mean()) / stim.std(ddof=1)
    if run["running_regressor"] and run_design.std(ddof=1) > 0:
        run_design = (run_design - run_design.mean()) / run_design.std(ddof=1)

    # Prediction: maps [neuron x cols x rows x delays x ON/OFF] back to coefficients [predictors x neurons]
    coefficients = rf["maps"].transpose(0, 4, 3, 1, 2).reshape(len(roi_index), -1).T
    fitted = ~np.isnan(rf["explainedVariance"])
    stim_prediction = np.full(z_traces.shape, np.nan)
    stim_prediction[:, fitted] = stim @ np.nan_to_num(coefficients[:, fitted])
    prediction = stim_prediction.copy()
    if run["running_regressor"]:
        prediction[:, fitted] += run_design @ rf["runningKernels"][fitted].T
    return t_toeplitz[valid_times], z_traces, prediction, stim_prediction


def plot_predicted_vs_actual(folder, title, note, n, rf, t_samples, z_traces, prediction, ev_all):
    """5. Calcium trace the model was fit to and the model's prediction, over the whole recording."""
    roi = rf["mpciROIs"][n]
    fig = Figure(figsize=(13, 4.8), layout="constrained")
    ax = fig.subplots()
    ax.plot(t_samples, z_traces[:, n], linewidth=0.8, label="trace (z-scored, as fit)")
    ax.plot(t_samples, prediction[:, n], linewidth=1.1, label="model prediction")
    ax.legend(frameon=False, loc="upper right")
    ax.set(xlabel="Time (s)", ylabel="z-scored activity")
    ax.set_title(f"{title}\nROI {roi} ({note}) | EV {ev_all[n]:.3f} all samples, {rf['explainedVariance'][n]:.3f} "
                 f"cross-validated | p {rf['pValues'][n]:.3f} | peak/noise {rf['peakToNoise'][n]:.2f}", fontsize=9)
    save(fig, folder, f"roi{roi:04d}")


def plot_peak_time_courses(folder, title, neurons, rf, maps, gauss, edges, meta):
    """6. ON, OFF (sign-flipped) and chosen RF over delays, at the stimulus pixel nearest the Gaussian centre."""
    left, right, bottom, top = edges
    n_cols, n_rows = maps.shape[1:3]
    col_centres = left + (np.arange(n_cols) + 0.5) * (right - left) / n_cols
    row_centres = bottom + (np.arange(n_rows) + 0.5) * (top - bottom) / n_rows
    delays = meta["rfDelays"]

    fig = Figure(figsize=(9.5, 2.6 * len(neurons)), layout="constrained")
    axes = np.atleast_1d(fig.subplots(len(neurons), 1, sharex=True))
    for ax, n in zip(axes, neurons):
        x0, y0 = gauss[n, 1], gauss[n, 3]
        col, row = np.argmin(np.abs(col_centres - x0)), np.argmin(np.abs(row_centres - y0))
        ax.plot(delays, maps[n, col, row, :, 0], alpha=0.5, label="ON")
        ax.plot(delays, -maps[n, col, row, :, 1], alpha=0.5, label="OFF (sign-flipped)")
        ax.plot(delays, chosen_rf(rf, maps, n)[col, row], linewidth=2,
                label=f"chosen ({SUBFIELD_NAMES[rf['bestSubfields'][n]]}, sign-corrected)")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_ylabel("RF amplitude")
        ax.set_title(f"ROI {rf['mpciROIs'][n]}: centre ({x0:.1f} az, {y0:.1f} el), "
                     f"peak/noise {rf['peakToNoise'][n]:.2f}", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[-1].set_xlabel("Delay after stimulus (s)")
    fig.suptitle(f"{title}\nRF time course at the Gaussian centre, best good RFs")
    save(fig, folder, "peakTimeCourses")


def roi_pixels(masks, roi):
    """(y, x) pixels of one ROI from the sparse masks [roi x y x x]."""
    in_roi = masks.coords[0] == roi
    return masks.coords[1][in_roi], masks.coords[2][in_roi]


def plot_roi_overlays(folder, title, image, masks, roi_types):
    """7. All ROIs and the curated neurons (mpciROITypes == 1) in green on the mean image."""
    fig = Figure(figsize=(16, 8), layout="constrained")
    for ax, (name, rois) in zip(fig.subplots(1, 2), [("All ROIs", np.arange(len(roi_types))),
                                                       ("Neurons", np.flatnonzero(roi_types == 1))]):
        layer = np.zeros((*image.shape, 4))
        for roi in rois:
            y, x = roi_pixels(masks, roi)
            layer[y, x] = (0, 1, 0, 0.3)
        ax.imshow(image, cmap="gray")
        ax.imshow(layer)
        ax.set_title(f"{name} (n = {len(rois)})")
        ax.set_axis_off()
    fig.suptitle(f"{title}: ROIs")
    save(fig, folder, "roiOverlays")


def draw_outline(ax, y, x, color, linewidth=1):
    """Outline of the ROI with pixels (y, x)."""
    footprint = np.zeros((np.ptp(y) + 3, np.ptp(x) + 3))
    footprint[y - y.min() + 1, x - x.min() + 1] = 1
    ax.contour(np.arange(x.min() - 1, x.max() + 2), np.arange(y.min() - 1, y.max() + 2), footprint,
               levels=[0.5], colors=[color], linewidths=linewidth)


def label_fovs(ax, fov_labels):
    """FOV names above the images of a mosaic, with a line between images; fov_labels: (x offset, width, name)."""
    for x_offset, width, name in fov_labels:
        ax.text(x_offset + width / 2, -5, name, ha="center", va="bottom")
        if x_offset > 0:
            ax.axvline(x_offset - 0.5, color="black", linewidth=1)


def plot_rf_roi_overlay(folder, title, image, masks, roi_index, gauss, significant, good, edges, azimuth_limits,
                        good_only=False, fov_labels=(), figsize=(14, 8)):
    """8. Neurons coloured by Gaussian RF centre (2D colour map): good RFs filled, other significant RFs outlined.

    good_only: only the good RFs (saved as rfRoiOverlayGood). Only RFs centred within azimuth_limits (the shown
    azimuth) are drawn, and the colour map spans azimuth_limits by the screen's elevation.
    Also used for the session mosaic (session plot 3): fov_labels names the FOV images in it (see label_fovs).
    """
    bottom, top = edges[2], edges[3]
    colormap = ColorMap2DBremm(range_x=(float(azimuth_limits[0]), float(azimuth_limits[1])),
                               range_y=(float(bottom), float(top)))
    inside = (gauss[:, 1] >= azimuth_limits[0]) & (gauss[:, 1] <= azimuth_limits[1])
    good = good & inside
    shown = good if good_only else significant & inside
    fig = Figure(figsize=figsize, layout="constrained")
    ax, ax_legend = fig.subplots(1, 2, width_ratios=[3, 1.3])

    layer = np.zeros((*image.shape, 4))
    for n in np.flatnonzero(shown):
        color = colormap(gauss[n, 1], gauss[n, 3]) / 255
        y, x = roi_pixels(masks, roi_index[n])
        if good[n]:
            layer[y, x] = (*color, 1)
        else:
            draw_outline(ax, y, x, color)
    ax.imshow(image, cmap="gray")
    ax.imshow(layer)
    label_fovs(ax, fov_labels)
    if good_only:
        name = f"Good RFs (n = {good.sum()})"
    else:
        name = f"Filled: good RFs (n = {good.sum()}); outlined: other significant RFs (n = {(shown & ~good).sum()})"
    ax.set_title(name, pad=18 if fov_labels else None)  # room for FOV names
    ax.set_axis_off()

    # Legend: the colour map over the shown visual field (stored as [x, y, RGB]) with the RF centres.
    ax_legend.imshow(colormap.get_cmap_data().transpose(1, 0, 2), origin="lower", extent=(*azimuth_limits, bottom, top),
                     aspect="auto")
    others = shown & ~good
    ax_legend.scatter(gauss[others, 1], gauss[others, 3], s=20, facecolors="none", edgecolors="black")
    ax_legend.scatter(gauss[good, 1], gauss[good, 3], s=20, color="black")
    ax_legend.set(xlim=azimuth_limits, ylim=(bottom, top), xlabel=AZIMUTH_LABEL, ylabel="Elevation (deg)",
                  title="RF centre colour", box_aspect=(top - bottom) / (azimuth_limits[1] - azimuth_limits[0]))
    if tuple(azimuth_limits) != tuple(edges[:2]):
        ax_legend.set_title(f"RF centre colour\nonly RFs centred within\nazimuth {azimuth_limits[0]:g} to "
                            f"{azimuth_limits[1]:g} deg")
    fig.suptitle(f"{title}: neurons coloured by RF centre" + (", good RFs only" if good_only else ""))
    save(fig, folder, "rfRoiOverlayGood" if good_only else "rfRoiOverlay")


# %% Session plots (all FOVs pooled), used by plot_session

def plot_rf_size_line(folder, title, thresholds, gauss_good):
    """Session 4. RF size of good RFs, (|sigma_x| + |sigma_y|) / 2, as counts per 2-degree bin."""
    size = (np.abs(gauss_good[:, 2]) + np.abs(gauss_good[:, 4])) / 2
    fig = Figure(figsize=(6.4, 6.4), layout="constrained")
    ax = fig.subplots()
    if size.size:
        counts, bin_edges = np.histogram(size, bins=np.arange(0, size.max() + 2, 2))
        ax.plot((bin_edges[:-1] + bin_edges[1:]) / 2, counts, color="0.45", linewidth=1.8)
        ax.axvline(size.mean(), color="black", linewidth=2, label="mean")
        ax.legend(frameon=False)
        stats = f"n = {size.size}\nmean = {size.mean():.2f} deg\nmedian = {np.median(size):.2f} deg"
    else:
        stats = "n = 0: no good RFs"
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="RF size (deg) = (|sigma_x| + |sigma_y|) / 2", ylabel="Number of neurons")
    ax.set_title(f"{title}\ngood-RF size, {thresholds}", fontsize=8)
    save(fig, folder, "rfSize")


def plot_ev_lines(folder, title, ev, is_significant, p_threshold):
    """Session 5. Stimulus explained variance over all samples (as in the shift test), p below vs. above
    threshold, each in % of its group per bin."""
    fitted = ~np.isnan(ev)
    bins = np.linspace(ev[fitted].min(), ev[fitted].max(), 40)
    centres = (bins[:-1] + bins[1:]) / 2
    fig = Figure(figsize=(6.4, 6.4), layout="constrained")
    ax = fig.subplots()
    for name, neurons, linestyle in [(f"p < {p_threshold}", fitted & is_significant, "-"),
                                     (f"p >= {p_threshold}", fitted & ~is_significant, "--")]:
        if neurons.any():
            counts, _ = np.histogram(ev[neurons], bins=bins)
            ax.plot(centres, 100 * counts / neurons.sum(), color="0.35", linestyle=linestyle, linewidth=1.8,
                    label=f"{name} (n = {neurons.sum()})")
    ax.axvline(ev[fitted].mean(), color="black", linewidth=2, label="mean, all neurons")
    ax.legend(frameon=False)
    ax.set(xlabel="Stimulus explained variance, all samples (as in the shift test)", ylabel="Neurons (% of group)")
    ax.set_title(f"{title}\nRF explained variance", fontsize=8)
    save(fig, folder, "evLines")


def plot_neuron_overlays(folder, title, image, masks, neurons, fov_labels, figsize):
    """Session 2. Curated neurons on the mean images of all FOVs: filled (top) and outlined (bottom)."""
    fig = Figure(figsize=(figsize[0], 2 * figsize[1]), layout="constrained")
    ax_filled, ax_outlined = fig.subplots(2, 1)
    layer = np.zeros((*image.shape, 4))
    for roi in neurons:
        y, x = roi_pixels(masks, roi)
        layer[y, x] = (0, 1, 0, 0.5)
        draw_outline(ax_outlined, y, x, "#00ff00", linewidth=0.7)
    for ax, name in [(ax_filled, "filled"), (ax_outlined, "outlined")]:
        ax.imshow(image, cmap="gray")
        label_fovs(ax, fov_labels)
        ax.set_title(f"Neurons (n = {len(neurons)}), {name}", pad=18)
        ax.set_axis_off()
    ax_filled.imshow(layer)
    fig.suptitle(f"{title}: neurons")
    save(fig, folder, "neuronOverlays")
