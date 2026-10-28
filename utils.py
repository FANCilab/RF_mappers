"""Helpers for mapAverageRF.py and the plotting modules.
"""
from math import ceil, floor

import numpy as np
import statsmodels.api as sm
from pylops import MatrixMult
from pyproximal import L1, Quadratic
from pyproximal.optimization.primal import ProximalGradient
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve1d
from scipy.optimize import least_squares


def gaussian_smooth(traces, sigma):
    """Gaussian smoothing along time (sigma in samples, kernel +-4 sigma), ignoring NaNs."""
    half_width = ceil(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half_width, half_width + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    valid = np.isfinite(traces)
    numerator = convolve1d(np.where(valid, traces, 0), kernel, axis=0, mode="constant")
    denominator = convolve1d(valid.astype(float), kernel, axis=0, mode="constant")
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def estimate_neuropil_lfr(F, Fneu, n_bins=20, neuropil_range=(10, 90), cell_percentile=5):
    """Neuropil correction F - r * Fneu - b (s2pUtils.estimateNeuropil_LFR).

    Fneu between its 10th and 90th percentiles is split into n_bins; the 5th percentile
    of F in each bin is robustly fit (bisquare) against the bin's Fneu value, giving
    intercept b and slope r. F and Fneu are [time x neurons].
    """
    corrected = np.full_like(F, np.nan)
    bins = np.arange(1, n_bins + 1)
    for n in range(F.shape[1]):
        f, neuropil = F[:, n], Fneu[:, n]
        if np.isnan(f).all() or np.isnan(neuropil).all():
            continue
        low, high = np.nanpercentile(neuropil, neuropil_range, method="hazen")
        neuropil_bin = np.round(n_bins * (neuropil - low) / (high - low))
        low_f = np.array([np.nanpercentile(f[neuropil_bin == b], cell_percentile, method="hazen")
                          if np.any(neuropil_bin == b) else np.nan for b in bins])
        bin_values = bins * (high - low) / n_bins + low
        has_data = ~np.isnan(low_f)
        fit = sm.RLM(low_f[has_data], sm.add_constant(bin_values[has_data]),
                     M=sm.robust.norms.TukeyBiweight(4.685)).fit()
        intercept, slope = fit.params
        corrected[:, n] = f - slope * neuropil - intercept
    return corrected


def interp_columns(x, traces, xq):
    """Linearly interpolate each column of traces at xq; NaN outside x."""
    return np.column_stack([np.interp(xq, x, col, left=np.nan, right=np.nan) for col in traces.T])


def make_lambda_matrix(dims, smooth_per_dim):
    """Smoothness regulariser for a kernel of shape dims (C order).

    Each column holds +1 for its parameter and -w for its neighbours along every
    dimension, scaled so an interior parameter's neighbours sum to -1.
    """
    n = int(np.prod(dims))
    index = np.arange(n).reshape(dims)
    matrix = np.zeros((n, n))
    for axis, weight in enumerate(smooth_per_dim):
        lo = np.take(index, range(dims[axis] - 1), axis).ravel()
        hi = np.take(index, range(1, dims[axis]), axis).ravel()
        matrix[lo, hi] -= weight
        matrix[hi, lo] -= weight
    return matrix / abs(np.median(matrix.sum(0))) + np.eye(n)


def make_stim_toeplitz(frames, times, rf_bins):
    """Design matrix [time x (delays * pixels)]: each row holds the frames rf_bins before it.

    Gaps longer than two frames are filled with blank frames. Returns (toeplitz, times).
    """
    stim = frames.reshape(len(frames), -1)
    frame_dt = np.median(np.diff(times))
    for g in np.flatnonzero(np.diff(times) > 2 * frame_dt)[::-1]:
        n_blank = floor((times[g + 1] - times[g]) / frame_dt)
        stim = np.insert(stim, [g + 1] * n_blank, 0, axis=0)
        times = np.insert(times, g + 1, times[g] + np.arange(1, n_blank + 1) * frame_dt)

    # Consecutive delays starting at rf_bins[0]; each block is stim shifted down by its delay.
    blocks = []
    for lag in range(rf_bins[0], rf_bins[0] + len(rf_bins)):
        shifted = np.zeros_like(stim)
        shifted[lag:] = stim[: len(stim) - lag]
        blocks.append(shifted)
    return np.hstack(blocks), times


def explained_variance(y, prediction):
    """Fraction of variance of each column of y explained by prediction."""
    return 1 - ((y - prediction) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def fit(A, Y, l1_lambda, l1_max_iter, l1_tol, l1_gram=None, l1_lipschitz=None):
    """Coefficients B of A B = Y (A includes the smoothness rows, Y their zero targets).

    l1_lambda None: least squares. Otherwise PyProximal FISTA minimises
    ||A B - Y||^2 / (2 N) + l1_lambda * sum|B| (N = rows of A), written as the quadratic
    B.T (A.T A / N) B / 2 - (A.T Y / N) B plus the L1 term; all columns of Y are fit together.
    l1_gram = A.T @ A / N and l1_lipschitz (its largest eigenvalue) can be passed when already computed.
    """
    if l1_lambda is None:
        return np.linalg.lstsq(A, Y, rcond=None)[0]
    if l1_gram is None:
        l1_gram = A.T @ A / len(A)
        l1_lipschitz = np.linalg.eigvalsh(l1_gram)[-1]
    n_coefs, n_targets = A.shape[1], Y.shape[1]
    l1_quadratic = Quadratic(Op=MatrixMult(l1_gram, otherdims=n_targets), b=-(A.T @ Y / len(A)).ravel())
    B = ProximalGradient(l1_quadratic, L1(sigma=l1_lambda), x0=np.zeros(n_coefs * n_targets), tau=1 / l1_lipschitz,
                         niter=l1_max_iter, acceleration="fista", tol=l1_tol)
    return B.reshape(n_coefs, n_targets)


def shift_test(z_traces, stim, run_design, penalties, neuron_penalty, n_shifts, rng):
    """Explained variance of RFs refit on circularly shifted traces [neurons x n_shifts].

    The model [stim, run_design] is refit with each neuron's penalty block
    (penalties[neuron_penalty[n]]; -1 skips the neuron); EV uses the stimulus part only.
    """
    n_t, n_stim = len(z_traces), stim.shape[1]
    X = np.hstack([stim, run_design])
    ev_shifted = np.full((z_traces.shape[1], n_shifts), np.nan)
    shifts = rng.integers(1, n_t + 1, n_shifts)
    shifted_index = (np.arange(n_t)[:, None] - shifts) % n_t  # column s is np.roll(z, shifts[s])
    for p, penalty in enumerate(penalties):
        neurons = np.flatnonzero(neuron_penalty == p)
        if neurons.size == 0:
            continue
        # Ridge solution operator for [X; penalty] B = [y; 0], stimulus coefficients only
        solver = np.linalg.pinv(np.vstack([X, penalty]))[:n_stim, :n_t]
        for n in neurons:
            shifted = z_traces[shifted_index, n]
            ev_shifted[n] = explained_variance(shifted, stim @ (solver @ shifted))
    return ev_shifted


def interpolate_rf_to_visual_degrees(rf, edges):
    """Resample an RF [cols x rows] onto a 1-degree grid [x x y].

    edges: [left, right, bottom, top] in degrees; col 0 is the left, row 0 the bottom.
    Returns (rf_deg, x, y).
    """
    left, right, bottom, top = edges
    half_w = (right - left) / rf.shape[0] / 2
    half_h = (top - bottom) / rf.shape[1] / 2
    grid_x = np.linspace(left + half_w, right - half_w, rf.shape[0])
    grid_y = np.linspace(bottom + half_h, top - half_h, rf.shape[1])
    x = np.arange(ceil(grid_x[0]), floor(grid_x[-1]) + 1)
    y = np.arange(ceil(grid_y[0]), floor(grid_y[-1]) + 1)
    interp = RegularGridInterpolator((grid_x, grid_y), rf, bounds_error=False)
    xq, yq = np.meshgrid(x, y, indexing="ij")
    return interp(np.stack([xq, yq], -1)), x, y


def gauss_2d_rotated(p, x, y):
    """Rotated 2D Gaussian; p = [amplitude, x0, sigma_x, y0, sigma_y, rotation]."""
    amplitude, x0, sx, y0, sy, rotation = p
    cos, sin = np.cos(rotation), np.sin(rotation)
    xr, yr = x * cos - y * sin, x * sin + y * cos
    x0r, y0r = x0 * cos - y0 * sin, x0 * sin + y0 * cos
    return amplitude * np.exp(-((xr - x0r) ** 2 / (2 * sx ** 2) + (yr - y0r) ** 2 / (2 * sy ** 2)))


def fit_2d_gauss_rf(rf, x, y):
    """Least-squares fit of gauss_2d_rotated to rf [x x y] sampled at x, y.

    Returns (parameters, fitted map).
    """
    xx, yy = np.meshgrid(x, y, indexing="ij")
    peak_x, peak_y = np.unravel_index(np.nanargmax(rf), rf.shape)
    start = [1, x[peak_x], np.mean(np.diff(x)) * 5, y[peak_y], np.mean(np.diff(y)) * 5, 0]
    max_sigma = max(np.ptp(x), np.ptp(y)) / 4
    lower = [0, x.min(), 0, y.min(), 0, -np.pi / 4]
    upper = [2 * np.nanmax(rf), x.max(), max_sigma, y.max(), max_sigma, np.pi / 4]
    fit = least_squares(lambda p: (gauss_2d_rotated(p, xx, yy) - rf).ravel(),
                        np.clip(start, lower, upper), bounds=(lower, upper))  # scipy needs a feasible start
    return fit.x, gauss_2d_rotated(fit.x, xx, yy)
