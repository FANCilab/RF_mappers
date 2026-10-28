"""Plots of the average receptive-field mapping pooled over a group of sessions (ALF object _ss_averageRF).

plot_group: RF centres, Gaussian RF density, explained variance and RF size of the RF-mapping runs of a group, saved
as PNG and SVG in <output_dir>/<group_name>/, with group_runs.txt listing the runs plotted. Each run is one FOV's
results (traces or deconvolved, any revision), loaded and mirrored as in single_session_plots.plot_session.
"""
import shutil
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from one.alf.path import ALFPath
from pycolormap_2d import ColorMap2DBremm

from single_session_plots import AZIMUTH_LABEL, META_NAME, VARIANTS, azimuth_window, load_run, model_traces, save
from utils import explained_variance

DEFAULT_GROUP_SETTINGS = {
    "group_name": None,            # name of the group, e.g. "groupA"; plots go to <output_dir>/<group_name>/
    "fov_runs": [],                # RF-mapping runs of single FOVs, as paths under the ONE root:
                                   #   subject/date/number/alf/FOV_XX/<rf_traces|rf_deconv>[/#revision#]
                                   #   without a revision, the latest one; e.g.
                                   #   ["NM023/2026-02-09/002/alf/FOV_00/rf_deconv",
                                   #    "NM023/2026-02-09/002/alf/FOV_01/rf_deconv/#2026-10-26#"]
    "output_dir": None,            # folder holding the group folders (required)
    "azimuth_range_contra_deg": None,  # (ipsilateral, contralateral) azimuth shown, in degrees towards each FOV's
                                       # contralateral side (0 = midline), e.g. (-10, 90); None = whole screen, or
                                       # no limit on that side. Zooms the RF centres and Gaussian density; RF
                                       # centres shows only RFs centred in this range, coloured across it
    # Plots
    "rf_centres": True,            # RF centres on the 2D colour map: significant and good, and good only
    "gaussian_coverage": True,     # summed Gaussian RFs over the visual field: good vs. all significant
    "explained_variance": True,    # stimulus EV, p < max_p_val vs. the rest: each session (grey) and their mean
                                   # (black); rebuilds the traces of every FOV
    "rf_size": True,               # good-RF size: each session (grey) and their mean (black)
}


def plot_group(one, settings):
    """Plot the RF-mapping runs of a group, all runs pooled; per-session plots combine the FOVs of each session.

    settings override DEFAULT_GROUP_SETTINGS; group_name, fov_runs and output_dir are required. Each path gives the
    session, FOV, signal and revision, so traces and deconvolved runs and different revisions can be pooled. A named
    revision must exist (no fallback to an earlier one). FOVs mapped as right hemisphere (meta.json) are mirrored, so
    azimuth > 0 is contralateral in every plot. Thresholds are those of each run (meta.json).

    Returns the plot folder.
    """
    settings = {**DEFAULT_GROUP_SETTINGS, **settings}
    for key in ("group_name", "fov_runs", "output_dir"):
        if not settings[key]:
            raise ValueError(f"settings[{key!r}] is required")

    signals = {variant: signal for signal, variant in VARIANTS.items()}  # "rf_deconv" -> "deconvolved"
    sessions = {}  # session label -> {"session_path", "runs"}, in the order of settings["fov_runs"]
    for path in settings["fov_runs"]:
        parts = path.strip("/").split("/")
        revision = parts.pop()[1:-1] if parts[-1].startswith("#") else None
        if len(parts) != 6 or parts[3] != "alf" or parts[5] not in signals:
            raise ValueError(f"{path}: expected subject/date/number/alf/FOV_XX/<rf_traces|rf_deconv>[/#revision#]")
        subject, date, number, _, fov_name, variant = parts
        session_path = ALFPath(one.cache_dir) / subject / date / f"{int(number):03d}"
        fov = f"alf/{fov_name}"
        revisions = sorted(one.list_revisions(session_path, collection=f"{fov}/{variant}", filename=META_NAME))
        if revision is None and revisions:
            revision = revisions[-1]
        if revision not in revisions:
            raise FileNotFoundError(f"{path}: no such RF-mapping run; revisions found: {revisions}")
        session = sessions.setdefault(session_path.session_path_short(), {"session_path": session_path, "runs": []})
        session["runs"].append(load_run(one, session_path, fov, signals[variant], revision))
    sessions = [{"label": label, **session} for label, session in sessions.items()]

    folder = Path(settings["output_dir"]) / settings["group_name"]
    if folder.exists():
        shutil.rmtree(folder)  # old plots of this group
    folder.mkdir(parents=True)
    runs = [r for s in sessions for r in s["runs"]]
    run_signals = [r["signal"] for r in runs]
    title = (f"{settings['group_name']} ({len(sessions)} sessions, {len(runs)} FOVs: "
             + ", ".join(f"{run_signals.count(signal)} {signal}" for signal in sorted(set(run_signals))) + ")")
    thresholds = " | ".join(sorted({r["thresholds"] for r in runs}))  # one entry unless the runs' thresholds differ
    listing = [f"{s['label']} {r['name']} {r['signal']} #{r['revision']}# {r['meta']['stimulusCrop']['hemisphere']} "
               f"hemisphere: {r['significant'].sum()} significant, {r['good'].sum()} good RFs"
               for s in sessions for r in s["runs"]]
    (folder / "group_runs.txt").write_text("\n".join(listing) + "\n")
    print(f"{settings['group_name']}: plotting\n  " + "\n  ".join(listing) + f"\n-> {folder}")

    # Pooled RFs; the visual field spans the kept area of every FOV.
    gauss = np.vstack([r["gauss"] for r in runs])
    significant = np.concatenate([r["significant"] for r in runs])
    good = np.concatenate([r["good"] for r in runs])
    edges = (min(r["edges"][0] for r in runs), max(r["edges"][1] for r in runs),
             min(r["edges"][2] for r in runs), max(r["edges"][3] for r in runs))
    azimuth_limits = azimuth_window(edges, settings["azimuth_range_contra_deg"])

    if settings["rf_centres"]:
        for good_only in (False, True):
            plot_rf_centres(folder, title, thresholds, gauss, significant, good, edges, azimuth_limits, good_only)
    if settings["gaussian_coverage"]:
        plot_gaussian_density(folder, title, thresholds, gauss, significant, good, edges, azimuth_limits)
    if settings["explained_variance"]:
        ev, is_significant = [], []  # per session, all FOVs
        for s in sessions:
            print(f"{s['label']}: rebuilding the traces for the explained variance")
            ev_fovs = []
            for r in s["runs"]:
                _, z_traces, _, stim_prediction = model_traces(one, s["session_path"], r["fov"], r["rf"], r["meta"])
                ev_fovs.append(explained_variance(z_traces, stim_prediction))
            ev.append(np.concatenate(ev_fovs))
            is_significant.append(np.concatenate([r["rf"]["pValues"] < r["settings"]["max_p_val"] for r in s["runs"]]))
        p_threshold = " / ".join(sorted({f"{r['settings']['max_p_val']:g}" for r in runs}))
        plot_ev_sessions(folder, title, ev, is_significant, p_threshold)
    if settings["rf_size"]:
        plot_rf_size_sessions(folder, title, thresholds,
                              [np.vstack([r["gauss"][r["good"]] for r in s["runs"]]) for s in sessions])
    return folder


def plot_rf_centres(folder, title, thresholds, gauss, significant, good, edges, azimuth_limits, good_only=False):
    """Group 1, 2. Gaussian RF centres on the 2D colour map of the shown visual field (as in the RF ROI overlay):
    good RFs filled, other significant RFs open.

    good_only: only the good RFs (saved as rfCentresGood). Only RFs centred within azimuth_limits are drawn.
    """
    bottom, top = edges[2], edges[3]
    colormap = ColorMap2DBremm(range_x=(float(azimuth_limits[0]), float(azimuth_limits[1])),
                               range_y=(float(bottom), float(top)))
    inside = (gauss[:, 1] >= azimuth_limits[0]) & (gauss[:, 1] <= azimuth_limits[1])
    good = good & inside
    others = np.zeros_like(good) if good_only else significant & inside & ~good
    shown_aspect = (azimuth_limits[1] - azimuth_limits[0]) / (top - bottom)  # width / height in degrees

    width = min(14, 1.5 + 3.6 * shown_aspect)
    fig = Figure(figsize=(width, (width - 1.2) / shown_aspect + 2.2), layout="constrained")  # room for the titles
    ax = fig.subplots()
    ax.imshow(colormap.get_cmap_data().transpose(1, 0, 2), origin="lower", extent=(*azimuth_limits, bottom, top))
    ax.scatter(gauss[others, 1], gauss[others, 3], s=22, facecolors="none", edgecolors="black", linewidths=0.9)
    ax.scatter(gauss[good, 1], gauss[good, 3], s=18, color="black")
    ax.set(xlim=azimuth_limits, ylim=(bottom, top), xlabel=AZIMUTH_LABEL, ylabel="Elevation (deg)")
    if good_only:
        name = f"Good RFs (n = {good.sum()})"
    else:
        name = f"Filled: good RFs (n = {good.sum()}); open: other significant RFs (n = {others.sum()})"
    if tuple(azimuth_limits) != tuple(edges[:2]):
        name += f"\nonly RFs centred within azimuth {azimuth_limits[0]:g} to {azimuth_limits[1]:g} deg"
    ax.set_title(name, fontsize=10)
    fig.suptitle(f"{title}: RF centres" + (", good RFs only" if good_only else "") + f"\n{thresholds}", fontsize=10)
    save(fig, folder, "rfCentresGood" if good_only else "rfCentres")


def plot_gaussian_density(folder, title, thresholds, gauss, significant, good, edges, azimuth_limits):
    """Group 3. Each RF as its Gaussian (amplitude 1, cut at 2 sigma), summed over the visual field: good RFs (top)
    and all significant RFs (bottom), each panel with its own colour scale. azimuth_limits zooms the x axis."""
    left, right, bottom, top = edges
    azimuth, elevation = np.meshgrid(np.linspace(left, right, 500), np.linspace(bottom, top, 250))
    shown_aspect = (azimuth_limits[1] - azimuth_limits[0]) / (top - bottom)  # width / height in degrees

    def density(neurons):
        total = np.zeros_like(azimuth)
        for _, x0, sx, y0, sy, rotation in gauss[neurons]:
            if sx <= 0 or sy <= 0:
                continue
            dx, dy = azimuth - x0, elevation - y0
            u = (dx * np.cos(rotation) + dy * np.sin(rotation)) / sx
            v = (-dx * np.sin(rotation) + dy * np.cos(rotation)) / sy
            total += np.where(u ** 2 + v ** 2 <= 4, np.exp(-(u ** 2 + v ** 2) / 2), 0)
        return total

    width = min(10, 2.2 + 2.3 * shown_aspect)  # narrower when zoomed
    fig = Figure(figsize=(width, 1.5 + 2 * (0.8 + (width - 2.2) / shown_aspect)), layout="constrained")
    for ax, (name, neurons) in zip(fig.subplots(2, 1),[(f"Good RFs (n = {good.sum()})", good),
                                                         (f"All significant RFs (n = {significant.sum()})", significant)]):
        summed = density(neurons)
        image = ax.imshow(summed, origin="lower", extent=edges, cmap="viridis", vmin=0, vmax=summed.max() or 1)
        ax.set(xlim=azimuth_limits, ylim=(bottom, top), xlabel=AZIMUTH_LABEL, ylabel="Elevation (deg)", title=name)
        fig.colorbar(image, ax=ax, label="Summed Gaussian RFs")
    fig.suptitle(f"{title}: Gaussian RF density\n{thresholds}", fontsize=10)
    save(fig, folder, "gaussianDensity")


def session_mean_legend(ax, session_style, mean_style, labels):
    """Legend: a grey line per session, a black line for their mean, for each linestyle in labels."""
    handles = []
    for linestyle, label in labels:
        handles.append(Line2D([], [], **session_style, linestyle=linestyle, label=f"{label}, each session"))
        handles.append(Line2D([], [], **mean_style, linestyle=linestyle, label=f"{label}, mean of sessions"))
    ax.legend(handles=handles, frameon=False, fontsize=8)


def plot_ev_sessions(folder, title, ev, is_significant, p_threshold):
    """Group 6. Stimulus explained variance over all samples (as in the shift test), p below vs. above threshold,
    in % of the session's group per bin: one grey line per session, black the mean of the sessions.

    ev, is_significant: one array per session (all its FOVs).
    """
    pooled = np.concatenate(ev)
    bins = np.linspace(np.nanmin(pooled), np.nanmax(pooled), 40)
    centres = (bins[:-1] + bins[1:]) / 2
    session_style, mean_style = {"color": "0.65", "linewidth": 1.1}, {"color": "black", "linewidth": 2.2}
    fig = Figure(figsize=(6.4, 6.4), layout="constrained")
    ax = fig.subplots()
    groups = [(f"p < {p_threshold}", "-", True), (f"p >= {p_threshold}", "--", False)]
    for _, linestyle, below in groups:
        curves = []
        for session_ev, session_significant in zip(ev, is_significant):
            values = session_ev[~np.isnan(session_ev) & (session_significant == below)]
            if values.size:  # no curve for a session without neurons in this group
                counts, _ = np.histogram(values, bins=bins)
                curves.append(100 * counts / values.size)
                ax.plot(centres, curves[-1], **session_style, linestyle=linestyle)
        if curves:
            ax.plot(centres, np.mean(curves, 0), **mean_style, linestyle=linestyle)
    session_mean_legend(ax, session_style, mean_style, [(linestyle, name) for name, linestyle, _ in groups])
    ax.set(xlabel="Stimulus explained variance, all samples (as in the shift test)",
           ylabel="Neurons (% of the session's group)")
    ax.set_title(f"{title}\nRF explained variance per session", fontsize=9)
    save(fig, folder, "evSessions")


def plot_rf_size_sessions(folder, title, thresholds, gauss_good):
    """Group 7. RF size of good RFs, (|sigma_x| + |sigma_y|) / 2, as counts per 2-degree bin: one grey line per
    session (zero for a session without good RFs), black the mean of the sessions.

    gauss_good: Gaussian parameters of the good RFs, one array per session (all its FOVs).
    """
    sizes = [(np.abs(g[:, 2]) + np.abs(g[:, 4])) / 2 for g in gauss_good]
    pooled = np.concatenate(sizes)
    session_style, mean_style = {"color": "0.65", "linewidth": 1.1}, {"color": "black", "linewidth": 2.2}
    fig = Figure(figsize=(6.4, 6.4), layout="constrained")
    ax = fig.subplots()
    if pooled.size:
        bins = np.arange(0, pooled.max() + 2, 2)
        centres = (bins[:-1] + bins[1:]) / 2
        curves = [np.histogram(size, bins=bins)[0] for size in sizes]
        for curve in curves:
            ax.plot(centres, curve, **session_style)
        ax.plot(centres, np.mean(curves, 0), **mean_style)
        session_mean_legend(ax, session_style, mean_style, [("-", "good RFs")])
        stats = (f"n = {pooled.size} in {len(sizes)} sessions\nmean = {pooled.mean():.2f} deg\n"
                 f"median = {np.median(pooled):.2f} deg")
    else:
        stats = "n = 0: no good RFs"
    ax.text(0.98, 0.80, stats, transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="RF size (deg) = (|sigma_x| + |sigma_y|) / 2", ylabel="Number of good RFs")
    ax.set_title(f"{title}\ngood-RF size per session, {thresholds}", fontsize=8)
    save(fig, folder, "rfSizeSessions")
