# Generate figures of carbonate classifications from hyperspectral imagery
#
# Australian Centre for Field Robotics
# Faculty of Engineering
# The University of Sydney
#
# SPDX-FileCopyrightText: 2025 Alexander Lowe <alexander.lowe@sydney.edu.au>
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------------------------

import feature_extraction
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.stats
import spectral
import spectral.io.envi

from pathlib import Path


def load_panel_factor(wavelengths, srt_data_file):
    data = np.loadtxt(srt_data_file, delimiter='\t')

    panel_factor_dn = scipy.signal.wiener(data[:, 1])
    panel_factor_dn = np.interp(wavelengths, data[:, 0], panel_factor_dn)

    panel_factor = np.interp(wavelengths, data[:, 0], data[:, 1])

    return panel_factor, panel_factor_dn


def image_filter(x, footsize, f_aggregate):
    x_filtered = np.array(x)
    footprint = np.sqrt(np.arange(-footsize, footsize+1)[:, np.newaxis]**2 + np.arange(-footsize, footsize+1)[np.newaxis, :]**2) <= footsize + 1e-6
    for m in range(footsize, x.shape[0] - footsize):
        for n in range(footsize, x.shape[1] - footsize):
            if np.isfinite(x[m, n]):
                vals = x[m - footsize:m + footsize + 1, n - footsize:n + footsize + 1][footprint].flatten()
                x_filtered[m, n] = f_aggregate(vals)
    return x_filtered

def my_subplots(N, ax_size_pix, cb_size_pix, title, dpi=100, margin=0.3, cb_margin=0.1):
    """
    Create a series of N pairs of axes all laid out horizontally one after the other.

    :param N: Number of pairs
    :param ax_size_pix: size in pixels of the first of each pair
    :param cb_size_pix: size in pixels of the second of each pair
    :param title: text title to place in the top margin
    :param dpi: dots per inch
    :param margin: margin around the edges of the figure and between pairs of axes (in inches)
    :return: N tuples containing pairs of axes
    """
    # Convert pixel sizes to inches
    ax_size_in = [x / dpi for x in ax_size_pix]
    cb_size_in = [x / dpi for x in cb_size_pix]

    # Total width and height of the figure in inches
    total_width = N * (ax_size_in[0] + cb_size_in[0] + margin + cb_margin) + 2*margin
    total_height = ax_size_in[1] + 2 * margin + 1.

    # Create the figure
    fig = plt.figure(figsize=(total_width, total_height), dpi=dpi)
    fig.suptitle(title)

    axes_pairs = []

    for i in range(N):
        # Compute left position for the main axis
        left_ax = margin + i * (ax_size_in[0] + cb_size_in[0] + margin + cb_margin)
        bottom = margin
        width_ax = ax_size_in[0]
        height_ax = ax_size_in[1]

        # Compute left position for the colorbar axis
        left_cb = left_ax + width_ax + cb_margin
        width_cb = cb_size_in[0]
        height_cb = cb_size_in[1]

        # Normalize positions to [0, 1] for add_axes
        fig_width, fig_height = fig.get_size_inches()
        ax_rect = [left_ax / fig_width, bottom / fig_height, width_ax / fig_width, height_ax / fig_height]
        cb_rect = [left_cb / fig_width, bottom / fig_height, width_cb / fig_width, height_cb / fig_height]

        ax = fig.add_axes(ax_rect)
        cb = fig.add_axes(cb_rect)
        axes_pairs.append((ax, cb))

    return fig, axes_pairs

def show_image(ax, x, vmin=None, vmax=None, title=None, colors=None, bounds=None, show_cb=True, aspect='equal'):
    if title is not None:
        ax[0].set_title(title)
    ax[0].axis('off')

    if colors is not None and bounds is not None:
        cmap = matplotlib.colors.ListedColormap(colors)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        im = ax[0].imshow(x, vmin=None, vmax=None, interpolation='none', aspect=aspect, cmap=cmap, norm=norm)
    else:
        im = ax[0].imshow(x, vmin=vmin, vmax=vmax, interpolation='none', aspect=aspect, cmap='jet')

    if show_cb:
        if colors is not None and bounds is not None:
            class_labels = ['unclassified', 'calcite', 'dolomite', 'aragonite', 'kaolinite']
            cbar = plt.colorbar(im, cax=ax[1], ticks=np.arange(len(class_labels)))
            cbar.ax.set_yticklabels(class_labels)

        else:
            plt.colorbar(im, cax=ax[1])
    else:
        ax[1].axis('off')

# @brief Prepare an image for plotting by applying a sigmoid and normalising
# @param s MxN image
def sigmoid_norm(s, th, scale):
    s = np.nan_to_num(s)

    def sig(x):
        z = (x-th) / scale
        return 1. / (1. + np.exp(-z))

    s = sig(s)

    s = s - np.min(s)
    if np.max(s) > 0:
        s = s / np.max(s)

    return s


# @brief Call SigmoidNorm on each channel of the image
# @param s MxNxC
def sigmoid_norm_per_channel(s, _th=None, scale=0.3):
    sn = np.zeros(s.shape)
    for n in range(0,s.shape[2]):
        th = _th
        if th is None:
            th = np.nanmedian(s[:,:,n])
        sn[:,:,n] = sigmoid_norm(s[:, :, n], th, scale)

    return sn

# @brief reduce a hyperspectral image cube to 3 bands, being the average of the 1st, 2nd and 3rd third of the full spectrum
# @param s hyperspectral cube MxNxC
# @return rgb image MxNx3
def triband(s):
    bands = s.shape[2]
    bins = [0, round(bands/3), round(2*bands/3), bands]
    s3 = np.zeros((s.shape[0],s.shape[1],3))
    for n in range(0,3):
        b1 = bins[n]
        b2 = max(b1+1,bins[n+1])
        s3[:,:,2-n] = np.nansum(s[:,:,b1:b2],axis=2)
        s3[:,:,2-n] -= np.nanmin(s3[:,:,2-n])
        smax = np.nanmax(s3[:,:,2-n])
        if smax > 0.:
            s3[:,:,2-n] /= smax

    return s3

def create_afe_maps(s, wavelengths, wl_low, wl_high, mean_refl_threshold, title, denoise_method=None, fn_base=None, interactive=True):
    wavelengths = np.array(wavelengths)
    idx = (wavelengths >= wl_low) & (wavelengths < wl_high)
    s = s[:, :, idx]
    wavelengths = wavelengths[idx]
    rgb = sigmoid_norm_per_channel(triband(s), None, 0.1)

    spectra = np.full((s.shape[0], s.shape[2]), np.nan, dtype=float)
    idx = np.nanmean(s[:, s.shape[1]//2, :], axis=1) > mean_refl_threshold
    for n in np.argwhere(idx).flatten():
        sn = s[n, s.shape[1]//2, :]
        sn = feature_extraction.denoise(sn, denoise_method)
        sc = feature_extraction.get_continuum(sn)
        spectra[n, :] = sn / sc

    fig, ax = my_subplots(8, [s.shape[1], s.shape[0]], [20, s.shape[0]], title)
    show_image(ax[0], rgb, title="False Colour", show_cb=False)
    show_image(ax[1], spectra, title="Reflectance Spectrum\n(normalised, middle column)", aspect='auto', vmax=1.)

    wl, intensity, fwhm, asym = feature_extraction.extract_features(
        s, wavelengths, mean_refl_threshold=mean_refl_threshold, denoise_method=denoise_method)
    #asym = image_filter(asym, 1, np.nanmedian)
    #wl = image_filter(wl, 2, np.nanmedian)

    is_calcite = (wl >= 2333.) & (wl <= 2340.) & (asym <= -0.35)
    is_dolomite = (wl >= 2312.) & (wl <= 2323.) & (asym <= -0.35)
    is_aragonite = (wl >= 2308.) & (wl <= 2322.) & (asym >= -0.35)
    #is_aragonite = (wl >= 2300.) & (wl <= 2328.) & (asym >= -0.35)

    carbonate_map = np.full(wl.shape, 0, dtype=float)
    carbonate_map[is_calcite] = 1
    carbonate_map[is_dolomite] = 2
    carbonate_map[is_aragonite] = 3

    carbonate_map_filter_radius = 1
    carbonate_map_f = image_filter(carbonate_map, carbonate_map_filter_radius, lambda x: scipy.stats.mode(x).mode)

    show_image(ax[2], wl, title="Feat. wavelen(nm)", vmin=np.nanpercentile(wl, 2), vmax=np.nanpercentile(wl, 98))
    show_image(ax[3], intensity, title="Feat. intensity", vmin=np.nanpercentile(intensity, 2), vmax=np.nanpercentile(intensity, 98))
    show_image(ax[4], fwhm, title='Feat. FWHM(nm)', vmin=np.nanpercentile(fwhm, 2), vmax=np.nanpercentile(fwhm, 98))
    show_image(ax[5], asym, title='Feat. asymmetry', vmin=np.nanpercentile(asym, 2), vmax=np.nanpercentile(asym, 98))

    colors = ['black', 'orange', 'pink', 'blue']
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    show_image(ax[6], np.ma.masked_invalid(carbonate_map), title="carbonate map", colors=colors, bounds=bounds, show_cb=False)
    show_image(ax[7], np.ma.masked_invalid(carbonate_map_f), title=f"carbonate map\n(filter, mode, radius={carbonate_map_filter_radius})", colors=colors, bounds=bounds)
    if fn_base is not None:
        fig.savefig(f"{fn_base}-AFE-{int(wl_low)}nm-{int(wl_high)}nm.png")

    def on_click_image(event):
        if event.key == "control":
            return
        x, y = int(event.xdata), int(event.ydata)
        if event.inaxes is ax[1][0]:
            x = s.shape[1] // 2
        feature_extraction.extract_features_single_spec(s[y, x], wavelengths, denoise_method, debug=True)
    fig.canvas.mpl_connect('button_press_event', lambda ev: on_click_image(ev))
    if interactive:
        plt.show()
    plt.close(fig)


def patch_wavelengths(hdr, s0, vimg_key, new_wavelengths, new_fwhm):
    """
    :brief Replace wavelengths.
        used if the image was acquired with erroneous wavelength calibrations,
        but you have the correct wavelengths on hand
    :param hdr: original header dict
    :param s0: original image cube
    :param vimg_key: 'vimg1' or 'vimg2'
    :param new_wavelengths: list of new wavelengths
    :param new_fwhm: list of new fwhm
    :return: new_hdr, new_s0
    """

    hdr = dict(hdr)

    def do_subst(arr, idx_from, idx_to, new_arr):
        idx_from = int(idx_from)
        idx_to = int(idx_to)
        for n in range(idx_from, idx_to):
            arr[n] = float(new_arr[n-idx_from]) if n-idx_from < len(new_arr) else np.nan

    if vimg_key == 'vimg1':
        do_subst(hdr['wavelength'], hdr['vimg1'][0]-1, hdr['vimg1'][1], new_wavelengths)
        do_subst(hdr['fwhm'], hdr['vimg1'][0] - 1, hdr['vimg1'][1], new_fwhm)
        total_keep = np.sum(np.isfinite(hdr['wavelength']))
        hdr['vimg1'] = [1., int(len(new_wavelengths))]
        hdr['vimg2'] = [int(len(new_wavelengths)+1), int(total_keep)]

    elif vimg_key == 'vimg2':
        do_subst(hdr['wavelength'], hdr['vimg2'][0]-1, hdr['vimg2'][1], new_wavelengths)
        do_subst(hdr['fwhm'], hdr['vimg2'][0] - 1, hdr['vimg2'][1], new_fwhm)
        total_keep = np.sum(np.isfinite(hdr['wavelength']))
        hdr['vimg2'][1] = int(total_keep)

    idx_keep = np.isfinite(hdr['wavelength'])
    hdr['wavelength'] = [float(x) for x in np.array(hdr['wavelength'])[idx_keep]]
    hdr['fwhm'] = [float(x) for x in np.array(hdr['fwhm'])[idx_keep]]
    hdr['vimg'][1] = int(total_keep)
    hdr['bands'] = int(total_keep)
    hdr['vroi'][1] = int(total_keep)
    s0 = s0[:, :, idx_keep]

    return hdr, s0


def load_and_plot_afe(
        path, srt_panel_factor_file, WHITE_CAL_LINES_SWIR=450, WHITE_CAL_LINES_VNIR=50,
        WAVELENGTH_PATCH_FILE=None, interactive=True):
    """
    Args:
        path: directory containing images to process
        srt_panel_factor_file: path containing spectrum of the white-calibration panel
        WHITE_CAL_LINES_SWIR: Number of lines at the start of the SWIR image to be used for white calibration
        WHITE_CAL_LINES_VNIR: Number of lines at the start of the VNIR image to be used for white calibration
        WAVELENGTH_PATCH_FILE: path to a file containing replacement wavelengths (for case where image was acquired with erroneous wavelength calibrations)
        interactive: show interactive figures, or just save them to disk
    """

    path = Path(path)

    for hdr_file in path.glob("*.hdr"):
        fn_base = str(hdr_file).replace(".hdr", "")
        if "panel" in fn_base or 'SWIR' in fn_base or 'VNIR' in fn_base:
            continue

        print(fn_base)
        _im = spectral.open_image(hdr_file)
        hdr = _im.metadata
        s0 = _im.load()

        hdr['vimg1'] = [int(x) for x in hdr['vimg1']]
        hdr['vimg2'] = [int(x) for x in hdr['vimg2']]
        hdr['wavelength'] = [float(x) for x in hdr['wavelength']]
        hdr['autodarkstartline'] = int(hdr['autodarkstartline'])

        if WAVELENGTH_PATCH_FILE is not None:
            hdr_patch = spectral.io.envi.open(WAVELENGTH_PATCH_FILE, image=f"{fn_base}.raw").metadata
            if 'wavelength1' in hdr_patch and 'fwhm1' in hdr_patch:
                hdr, s0 = patch_wavelengths(hdr, s0, 'vimg1', hdr_patch['wavelength1'], hdr_patch['fwhm1'])
            if 'wavelength2' in hdr_patch and 'fwhm2' in hdr_patch:
                hdr, s0 = patch_wavelengths(hdr, s0, 'vimg2', hdr_patch['wavelength2'], hdr_patch['fwhm2'])

        # white calibration from front of current image
        panel_factor, panel_factor_dn = load_panel_factor(hdr['wavelength'], srt_panel_factor_file)
        s_white_vnir = np.nanmean(s0[:WHITE_CAL_LINES_VNIR, :, :], axis=0)
        s_white_swir = np.nanmean(s0[:WHITE_CAL_LINES_SWIR, :, :], axis=0)
        s_white = np.concatenate((s_white_vnir[:, hdr['vimg1'][0]-1:hdr['vimg1'][1]], s_white_swir[:, hdr['vimg2'][0]-1:hdr['vimg2'][1]]), axis=1)

        # dark calibration from end of current image
        from_idx = hdr['autodarkstartline'] + 15
        s_dark = np.nanmean(s0[from_idx:, :, :], axis=0)

        def plot_calibration_spectra(wl, swhite, sdark, panel_factor, panel_factor_dn):
            fig, ax = plt.subplots(1, 3, figsize=[12., 4.])
            ax[0].plot(wl, swhite.T)
            ax[0].set_title("per-column white calibration")
            ax[1].plot(wl, sdark.T)
            ax[1].set_title("per-column dark current")
            ax[2].plot(wl, panel_factor)
            ax[2].plot(wl, panel_factor_dn)
            ax[2].set_title("Panel Factor")
            [x.grid() for x in ax]
            plt.show()

        if interactive:
            plot_calibration_spectra(hdr['wavelength'], s_white, s_dark, panel_factor, panel_factor_dn)

        # white/dark calibration and panel factor
        s = panel_factor_dn[np.newaxis, np.newaxis, :] * (s0 - s_dark[np.newaxis, :, :]) / (s_white - s_dark)[np.newaxis, :, :]

        # fix some broken pixels
        for col, wl in ((200, 2176.5), (220, 2436.9)):
            idx = np.argmin(np.abs(np.array(hdr['wavelength']) - wl)).flatten()[0]
            s[:, col, idx] = np.nanmedian(s[:, col, idx-1:idx+2], axis=1)

        for wl_low, wl_high, mean_refl_threhold in (
            #(1050., 1500., 0.05),
            #(1500., 2000., 0.05),
            (2000., 2450., 0.15),
        ):
            create_afe_maps(s[0:, :, :], hdr['wavelength'], wl_low, wl_high, mean_refl_threhold,
                            title=fn_base, denoise_method="WIENER", fn_base=fn_base, interactive=interactive)


if __name__ == "__main__":
    load_and_plot_afe('data/',
                      'data/SRT-99-100.txt',
                      WAVELENGTH_PATCH_FILE='data/swir_wavelength_patch.txt',
                      interactive=True)
