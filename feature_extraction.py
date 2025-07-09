# Calculate Automated Feature Extraction (AFE) features from a hyperspectral image
#
# Australian Centre for Field Robotics
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2025 Alexander Lowe <alexander.lowe@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------------------------


import numpy as np
from scipy.signal import wiener
from scipy.integrate import simpson
import matplotlib.pyplot as plt


def extract_features(s, wavelengths, mean_refl_threshold=0.15, std_refl_threshold=0.03, with_plots=False, denoise_method=None):
    """
    Extracts AFE features from a hyperspectral cube suitable for detecting carbonates as per:

        Murphy, R. J., et al. (2017),High-resolution hyperspectral imaging of diagenesis and clays in fossil
        coral reef material: a nondestructive tool for improving environmental and climate reconstructions, Geochem.
        Geophys.Geosyst., 18, 3209–3230, doi:10.1002/2017GC006949.

    Args:
        s: MxNxD hyperspectral image cube of calibrated reflectance
        wavelengths: D-array of wavelengths
        mean_refl_threshold: pixels with mean(s) below this threshold are considered uninteresting - feature values for such pixels have np.nan value
        std_refl_threshold: pixels with std(s)/mean(s) below this threshold are considered uninteresting - feature values for such pixels have np.nan value
        with_plots: plot line-by-line preview of feature values as they're calculated
        denoise_method: eg "POLY" or "WIENER" or None

    Returns:
        feature_wl, feature_intensity, feature_fwhm, feature_asymmetry - all MxN arrays of feature values
    """

    M, N, K = s.shape

    feature_wl = np.full(s.shape[:2], np.nan, dtype=float)
    feature_intensity = np.full(s.shape[:2], np.nan, dtype=float)
    feature_fwhm = np.full(s.shape[:2], np.nan, dtype=float)
    feature_asymmetry = np.full(s.shape[:2], np.nan, dtype=float)

    if with_plots:
        plt.ion()
        fig, ax = plt.subplots(1, 5)
        im = [
            ax[0].imshow(feature_wl, cmap='viridis', vmin=np.min(wavelengths), vmax=np.max(wavelengths)),
            ax[1].imshow(feature_intensity, cmap='viridis', vmin=0., vmax=0.5),
            ax[2].imshow(feature_fwhm, cmap='viridis', vmin=0., vmax=500.),
            ax[3].imshow(feature_asymmetry, cmap='viridis', vmin=-1., vmax=1.)
        ]
        for i in im:
            plt.colorbar(i)

    for m in range(M):
        for n in range(N):
            spectrum0 = s[m, n, :]
            mean_spectrum = np.mean(spectrum0)
            std_spectrum = np.std(spectrum0)
            if not np.all(np.isfinite(spectrum0)) or mean_spectrum  < mean_refl_threshold or std_spectrum/mean_spectrum < std_refl_threshold:
                continue

            features = extract_features_single_spec(spectrum0, wavelengths, denoise_method)
            if features is None:
                continue

            wl_min, spec_min, fwhm_low, fwhm_high, asymmetry = features
            feature_wl[m, n] = wl_min
            feature_intensity[m, n] = 1. - spec_min
            feature_fwhm[m, n] = fwhm_high - fwhm_low
            feature_asymmetry[m, n] = asymmetry

        if with_plots:
            im[0].set_data(feature_wl)
            im[1].set_data(feature_intensity)
            im[2].set_data(feature_fwhm)
            im[3].set_data(feature_asymmetry)
            plt.draw()
            plt.pause(0.1)
            plt.show()

    return feature_wl, feature_intensity, feature_fwhm, feature_asymmetry


def get_continuum(y0, idx=None):
    """
    Compute the upper convex hull of a 1D signal using a custom approach.
    Returns the x and y coordinates of the upper hull.
    """

    if idx is None:
        idx = np.arange(len(y0))

    x0 = np.arange(len(y0))
    x = x0[idx]
    y = y0[idx]

    hull_x = [x[0]]
    hull_y = [y[0]]

    for i in range(1, len(x)):
        hull_x.append(x[i])
        hull_y.append(y[i])
        while len(hull_x) >= 3:
            x1, x2, x3 = hull_x[-3], hull_x[-2], hull_x[-1]
            y1, y2, y3 = hull_y[-3], hull_y[-2], hull_y[-1]
            # Check if the middle point is below the line formed by the outer points
            if (y2 - y1) * (x3 - x2) < (y3 - y2) * (x2 - x1):
                # Middle point is not part of the upper hull
                del hull_x[-2]
                del hull_y[-2]
            else:
                break

    return np.interp(x0, hull_x, hull_y)


def find_min_wavelength(spectrum, wavelengths):
    """
    Finds the wavelength of the minimum spectrum value for each pixel in a hyperspectral image cube
    using quadratic interpolation, and returns the interpolated spectrum value at that point.

    Parameters:
    s (numpy.ndarray): Hyperspectral image cube of shape (N, M, K).
    wavelengths (numpy.ndarray): 1D array of corresponding wavelengths of length K.

    Returns:
    tuple: Two 2D arrays of shape (N, M) - one with the wavelength of the minimum spectrum value for each pixel,
           and one with the interpolated spectrum value at that point.
    """
    K = len(spectrum)

    # Find the index of the minimum value in the spectrum
    idx_search = np.arange(K)
    min_idx = np.argmin(spectrum)
    while (min_idx == idx_search[0] or min_idx == idx_search[-1]) and len(idx_search) > K // 2:
        if min_idx == idx_search[0]:
            idx_search = idx_search[1:]
            min_idx = idx_search[np.argmin(spectrum[idx_search])]
        elif min_idx == idx_search[-1]:
            idx_search = idx_search[:-2]
            min_idx = idx_search[np.argmin(spectrum[idx_search])]
    if len(idx_search) <= K // 2:
        min_idx = np.argmin(spectrum)

    # Ensure we have adjacent samples for quadratic interpolation
    if min_idx == 0 or min_idx == K - 1:
        # If the minimum is at the boundary, use the boundary wavelength and value
        min_wavelength = wavelengths[min_idx]
        min_value = spectrum[min_idx]
    else:
        # Get the three points for quadratic interpolation
        x1, x2, x3 = wavelengths[min_idx - 1], wavelengths[min_idx], wavelengths[min_idx + 1]
        y1, y2, y3 = spectrum[min_idx - 1], spectrum[min_idx], spectrum[min_idx + 1]

        # Perform quadratic interpolation to find the vertex of the parabola
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        B = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (y2 - y3)) / denom
        C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

        # The vertex of the parabola (minimum point) is at x = -B / (2A)
        min_wavelength = -B / (2 * A)
        min_value = A * min_wavelength ** 2 + B * min_wavelength + C

        min_wavelength = min_wavelength
        min_value = min_value

    return min_wavelength, min_value


def find_interpolated_wavelengths(wavelengths, spectrum, target_value):
    """
    Finds the linearly interpolated wavelengths at which the linearly interpolated spectrum equals the given value.

    Parameters:
    spectrum (numpy.ndarray): 1D array of spectrum values.
    wavelengths (numpy.ndarray): 1D array of corresponding wavelengths.
    target_value (float): The target spectrum value to find the interpolated wavelengths for.

    Returns:
    numpy.ndarray: Array of interpolated wavelengths where the spectrum equals the target value.
    """
    interpolated_wavelengths = []

    for i in range(len(spectrum) - 1):
        if (spectrum[i] <= target_value <= spectrum[i + 1]) or (spectrum[i] >= target_value >= spectrum[i + 1]):
            # Linear interpolation formula
            x1, x2 = wavelengths[i], wavelengths[i + 1]
            y1, y2 = spectrum[i], spectrum[i + 1]
            interpolated_wavelength = x1 + (target_value - y1) * (x2 - x1) / (y2 - y1)
            if not np.isfinite(interpolated_wavelength):
                interpolated_wavelength = 0.5 * (x1 + x2)
            interpolated_wavelengths.append(interpolated_wavelength)

    return np.array(interpolated_wavelengths)


def find_wavelength_where_spectrum_closest_to(wavelengths, spectrum, target_value):
    d = np.abs(spectrum - target_value)
    idx_min = np.argmin(d)
    return np.array([wavelengths[idx_min]])


def get_lower_upper_wavelengths(spectrum, wavelengths, wl_central, target_value):

    f = find_interpolated_wavelengths
    #f = find_wavelength_where_spectrum_closest_to

    idx = np.argwhere(np.array(wavelengths) <= wl_central).flatten()
    wl_low = f(wavelengths[idx], spectrum[idx], target_value)
    wl_low = wl_low[-1] if len(wl_low)>0 else wavelengths[0]

    idx = np.argwhere(np.array(wavelengths) >= wl_central).flatten()
    wl_high = f(wavelengths[idx], spectrum[idx], target_value)
    wl_high = wl_high[0] if len(wl_high)>0 else wavelengths[-1]

    return wl_low, wl_high


def calculate_assymetry(wl, spec, wl_central):

    f_integral = simpson
    # f_integral = np.trapz

    idx_central = np.argmin(np.abs(wl - wl_central)).flatten()[0]
    try:
        area1 = f_integral(spec[0:idx_central+1], x=wl[0:idx_central+1])
        area2 = f_integral(spec[idx_central:-1], x=wl[idx_central:-1])
    except ValueError:
        return 0.
    asymmetry = np.log10(area2 / area1)
    return asymmetry


def denoise(s, method="WIENER", order=30):
    """
    denoise the given 1D spectrum.

    Parameters:
    - s: 1D numpy array representing the spectrum

    Returns:
    - 1D numpy array, denoised spectrum
    """

    if method == "POLY":
        x = np.arange(len(s))
        coeffs = np.polyfit(x, s, order)
        poly = np.poly1d(coeffs)
        return poly(x)

    elif method == "WIENER":
        return wiener(s)

    else:
        raise ValueError(f"unknown noise filter method={method}")


def extract_features_single_spec(spectrum0, wavelengths, denoise_method=None, debug=False):
    if denoise_method:
        spectrum_c = denoise(spectrum0, method=denoise_method)

    else:
        spectrum_c = spectrum0

    continuum = get_continuum(spectrum_c)
    spectrum = spectrum_c / continuum

    wl_min, spec_min = find_min_wavelength(spectrum, wavelengths)
    fwhm_low, fwhm_high = get_lower_upper_wavelengths(spectrum, wavelengths, wl_min, 0.5 * (1. + spec_min))
    wl_low, wl_high = get_lower_upper_wavelengths(spectrum, wavelengths, wl_min, 1.0)
    if wl_low is None or wl_high is None:
        return None

    idx = (wavelengths >= wl_low) & (wavelengths <= wl_high)
    asymmetry = calculate_assymetry(wavelengths[idx], 1. - spectrum[idx], wl_min)

    if debug:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f"wl={int(wl_min)}, intensity={1.-spec_min:.3f}, fwhm={int(fwhm_high-fwhm_low)}, asym={asymmetry:.3f}")
        ax.plot(wavelengths, spectrum0, label="reflectance", color='blue')
        ax.plot(wavelengths, spectrum_c, label="reflectance(denoised)", color='orange')
        ax.plot(wavelengths, continuum, label="continuum", color='green')
        ax.set_ylim(bottom=0.)
        ax.set_ylabel('reflectance')

        # Create a second y-axis for the 'normalised' spectrum
        ax2 = ax.twinx()
        ax2.plot(wavelengths, spectrum, label="normalised", color='red')
        ax2.set_ylim(bottom=0.)
        ax2.set_ylabel('normalised')

        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='lower right')

        ax.grid()
        plt.show()

    return wl_min, spec_min, fwhm_low, fwhm_high, asymmetry
