"""
2023-10-25

Functions for computing feature indices of ground-motion data.
"""

from typing import Tuple

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from scipy import integrate
from scipy.signal import argrelmax, argrelmin
from scipy.fft import fft, fftfreq, ifft


def duration(data: np.ndarray, dt: float = 0.01, *args) -> np.ndarray:
    """
    Calculate the significant duration of ground-motion data.

    If only `data` and `dt` are provided, the function returns the times `t5`, `t45`, and `t95`
    (Required to calculate the D_{5-95} and D_{5-45} values).
    Here, `t_n` denotes the time (in seconds) at which the waveform’s cumulative power reaches n% of the total cumulative power.
    By passing desired percentages via `args`, the function returns the corresponding `t_n` values for those percentages.

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data.
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    args : float
        Percent levels (from 0.0 to 1.0) for which to calculate the duration.
        For example, `0.05, 0.75, 0.95`.

    Returns
    -------
    np.ndarray
        - If `args` is not provided: An array containing the values of t5, t45, and t95.
          The shape is (3,) for a single record or (N, 3) for N records.
        - If `args` is provided: An array containing the calculated durations for each percentage level.
          The shape is (len(args),) for a single record or (N, len(args)) for N records.
    """

    # Check data dimension
    data = check_data_size(data)

    power = np.square(data)
    all_power = integrate.trapezoid(power, dx=dt, axis=1)
    cum_sum = integrate.cumulative_trapezoid(power, dx=dt, axis=1, initial=0) / all_power[:, None]

    if not args:
        # Return t5, t45, and t95
        t5 = np.count_nonzero(cum_sum < 0.05, axis=1) * dt
        t45 = np.count_nonzero(cum_sum < 0.45, axis=1) * dt
        t95 = np.count_nonzero(cum_sum < 0.95, axis=1) * dt

        return np.stack([t5, t45, t95], axis=1).squeeze()

    out_list = []

    for arg in args:
        if not (0.0 <= arg <= 1.0):
            raise ValueError("args must be between 0 and 1.")

        temp = np.count_nonzero(cum_sum < arg, axis=1) * dt
        out_list.append(temp)

    out_mat = np.stack(out_list, axis=1)

    return out_mat.squeeze()


def arias_intensity(data: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Arias intensity.

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data (unit: cm/s^2).
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    const = np.pi / (2 * 980.665)
    data = data * np.sqrt(const)

    arias = integrate.trapezoid(np.square(data), dx=dt, axis=1)

    return arias


def arias_intensity_range(data: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Compute Arias intensity using only the D_{5-95} duration window.

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data (unit: cm/s^2).
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    power = np.square(data)
    all_power = integrate.trapezoid(power, dx=dt, axis=1)
    cum_sum = integrate.cumulative_trapezoid(power, dx=dt, axis=1, initial=0) / all_power[:, np.newaxis]

    t5_ind = np.count_nonzero(cum_sum < 0.05, axis=1)
    t95_ind = np.count_nonzero(cum_sum < 0.95, axis=1)

    const = np.pi / (2 * 980.665)
    data = data * np.sqrt(const)

    arias = []

    for i in range(data.shape[0]):
        temp = integrate.trapezoid(np.square(data[i, t5_ind[i] : t95_ind[i] + 1]), dx=dt)
        arias.append(temp)

    return np.array(arias)


def cav_std(data: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Cumulative absolute velocity (CAVstd)

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data (unit: cm/s^2).
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    interval = int(round(1.0 / dt))
    thresh = 980.665 * 0.025
    data_len = data.shape[1]

    n_full = data_len // interval

    if n_full != 0:
        data = data[:, : n_full * interval]

    data = data.reshape(data.shape[0], -1, interval)
    max_list = np.max(np.abs(data), axis=2)
    use = max_list >= thresh

    integ = integrate.trapezoid(np.abs(data), dx=dt, axis=2)
    out = np.sum(integ * use, axis=1)

    return out


def spectral_intensity(data: np.ndarray, period: np.ndarray) -> np.ndarray:
    """
    Spectral intensity.

    Parameters
    ----------
    data : np.ndarray
        Velocity response spectra.
        Either a 1D array of length T (single record) or a 2D array of shape (N, T) (N records).
    period : np.ndarray
        Array of natural periods (unit: s) to be considered (length T).
    """

    # Check data dimension
    data = check_data_size(data)

    ind = np.where((period >= 0.1) & (period <= 2.5))[0]
    period = period[ind]
    data = data[:, ind]

    out_list = []

    for kk in range(data.shape[0]):
        temp = integrate.trapezoid(data[kk, :], x=period)
        out_list.append(temp / 2.4)

    return np.array(out_list)


def zero_crossing_rate(data: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Zero-crossing rate.

    This function defines the zero-crossing rate as the average of the zero-level up-crossing rate
    and the zero-level down-crossing rate.
    Crossings that merely touch zero (i.e., the signal turns exactly at 0 without changing sign) are not counted.

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data.
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    out = []

    for ii in range(data.shape[0]):
        s1 = data[ii, :-1]
        s2 = data[ii, 1:]

        up = np.count_nonzero((s1 < 0) & (s2 > 0))
        down = np.count_nonzero((s1 > 0) & (s2 < 0))

        out.append(0.5 * (up + down) / ((data.shape[1] - 1) * dt))

    out = np.array(out)

    return out.squeeze()


def zero_crossing_rate_range(data: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Compute zero-crossing rate using only the D_{5-95} duration window.

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data.
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    all_power = integrate.trapezoid(np.square(data), dx=dt, axis=1)
    cum_sum = integrate.cumulative_trapezoid(np.square(data), dx=dt, axis=1, initial=0) / all_power[:, np.newaxis]

    t5_ind = np.count_nonzero(cum_sum < 0.05, axis=1)
    t95_ind = np.count_nonzero(cum_sum < 0.95, axis=1)

    out = []

    for i in range(data.shape[0]):
        acc = data[i, t5_ind[i] : t95_ind[i] + 1]
        out.append(zero_crossing_rate(acc, dt))

    return np.array(out)


def seismic_intensity(data: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    JMA instrumental seismic intensity (Shindo).

    Note (limitation)
    -----------------
    - This implementation accepts only 1D ground-motion acceleration time-history data.
    - The official JMA definition targets 3-component records and requires vector composition
      of two horizontal components and one vertical component; that part is not implemented here.
    - Therefore, the seismic intensity values calculated by this function do not correspond to the
      official JMA seismic intensity scale table.

    Method
    ------
    The computation follows, in principle, the description on the JMA website (Japanese):
    https://www.jma.go.jp/jma/kishou/know/jishin/kyoshin/kaisetsu/calc_sindo.html

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data (unit: cm/s^2).
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    data_len = data.shape[1]
    freq = np.abs(fftfreq(data_len, dt))

    # Filter (low-cut)
    fl = np.sqrt(1.0 - np.exp(-((freq / 0.5) ** 3)))

    # Filter (high-cut)
    gam = freq * 0.1
    fh = (
        1.0
        + 0.694 * gam**2
        + 0.241 * gam**4
        + 0.0557 * gam**6
        + 0.009664 * gam**8
        + 0.00134 * gam**10
        + 0.000155 * gam**12
    ) ** (-0.5)

    ff_arr = np.zeros_like(freq)
    pos = freq > 0.0
    ff_arr[pos] = np.sqrt(1.0 / freq[pos])

    # Set DC component to zero
    ff_arr[0] = 0.0

    # Filter
    f_all = fl * fh * ff_arr

    idx = int(np.ceil(0.3 / dt))

    out = []

    for kk in range(data.shape[0]):
        yk = np.asarray(fft(data[kk, :]), dtype=np.complex128)
        yk_f = yk * f_all.astype(np.complex128, copy=False)
        acc_f = np.asarray(ifft(yk_f), dtype=np.complex128).real

        abs_acc = np.abs(acc_f)
        s_acc = np.sort(abs_acc)[::-1]

        temp = 2 * np.log10(s_acc[idx - 1]) + 0.94
        temp = float(Decimal(str(temp)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        # temp = np.round(temp, 2)
        temp = np.floor(temp * 10) / 10

        out.append(temp)

    return np.array(out)


def extrema_rate(data: np.ndarray, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Number of negative maxima and positive minima per unit time.

    The definitions of these indices follow Rezaeian and Der Kiureghian (2010).

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data.
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    out_1 = []
    out_2 = []

    for ii in range(data.shape[0]):
        negative_maxima = argrelmax(data[ii, :])
        out_1.append(np.count_nonzero(data[ii, :][negative_maxima] < 0))

        positive_minima = argrelmin(data[ii, :])
        out_2.append(np.count_nonzero(data[ii, :][positive_minima] > 0))

    out_1 = np.array(out_1) / ((data.shape[1] - 1) * dt)
    out_2 = np.array(out_2) / ((data.shape[1] - 1) * dt)

    return out_1, out_2


def extrema_rate_range(data: np.ndarray, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute extrema rate using only the D_{5-95} duration window.

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data.
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    all_power = integrate.trapezoid(np.square(data), dx=dt, axis=1)
    cum_sum = integrate.cumulative_trapezoid(np.square(data), dx=dt, axis=1, initial=0) / all_power[:, np.newaxis]

    t5_ind = np.count_nonzero(cum_sum < 0.05, axis=1)
    t95_ind = np.count_nonzero(cum_sum < 0.95, axis=1)

    out_1 = []
    out_2 = []

    for i in range(data.shape[0]):
        acc = data[i, t5_ind[i] : t95_ind[i] + 1]
        temp_1, temp_2 = extrema_rate(acc, dt)
        out_1.append(temp_1[0])
        out_2.append(temp_2[0])

    return np.array(out_1), np.array(out_2)


def pred_freq(
    data: np.ndarray, dt: float = 0.01, band_width: float = 0.5, f_max: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predominant frequency of the ground-motion data.

    The predominant frequency is obtained through the following steps:
        1. Compute the Fourier amplitude spectrum of the input `data` (FFT).
        2. Smooth spectrum.
        3. Identify the frequency at which the smoothed Fourier amplitude attains its maximum.

    The smoothing filter for the Fourier amplitude follows Osaki (1994).

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data.
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    band_width : float
        Band width of the Parzen window used in the smoothing filter (unit: Hz).
    f_max : float
        Frequency extent used to construct the smoothing window; the window is formed over (-f_max, f_max).
    """

    # Check data dimension
    data = check_data_size(data)

    data_len = data.shape[1]
    freq = fftfreq(data_len, dt)[: (data_len + 1) // 2]
    delta_f = float(freq[1] - freq[0])

    center = []
    pred = []

    for i in range(data.shape[0]):
        ck = fft(data[i, :])[: (data_len + 1) // 2] / data_len
        fft_amp = np.abs(ck)
        center.append(np.dot(freq, fft_amp) / np.sum(fft_amp))
        fft_amp_w_num = smoothing_parzen(fft_amp, band_width, delta_f, f_max)
        max_freq_index = np.argmax(fft_amp_w_num)
        pred.append(freq[max_freq_index])

    return np.array(pred), np.array(center)


def mean_period(data: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """
    Mean period.

    Parameters
    ----------
    data : np.ndarray
        Ground-motion data.
        Either a 1D array of length W (single record) or a 2D array of shape (N, W) (N records).
    dt : float
        Sampling interval (unit: s)
    """

    # Check data dimension
    data = check_data_size(data)

    data_len = data.shape[1]
    freq = fftfreq(data_len, dt)[: (data_len + 1) // 2]

    # Compute in the range of (0.25, 20) Hz
    ind = np.where((freq >= 0.25) & (freq <= 20))[0]

    out = []

    for i in range(data.shape[0]):
        ck = fft(data[i, :])[: (data_len + 1) // 2] / data_len
        fft_amp = np.abs(ck)

        temp_1 = np.sum(fft_amp[ind] ** 2 / freq[ind])
        temp_2 = np.sum(fft_amp[ind] ** 2)
        out.append(temp_1 / temp_2)

    return np.array(out)


def check_data_size(data: np.ndarray) -> np.ndarray:
    """
    Ensure the input array is 2-dimensional.

    If the input array is 1D, it is converted to a 2D array by adding a new axis at the beginning.
    If it is already 2D, it is returned unchanged.

    Parameters
    ----------
    data : np.ndarray
        Input data, expected to be 1D or 2D.

    Returns
    -------
    np.ndarray
        A 2D version of the input data.
    """

    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:
        pass
    else:
        raise ValueError("Input array must be 1D or 2D, but got {data.ndim} dimensions.")

    return data


def smoothing_parzen(f_spec: np.ndarray, band_width: float, delta_f: float, f_max: float) -> np.ndarray:
    """
    Smoothing the Fourier amplitude spectrum.

    This function is implemented with reference to Chapter 6 of Osaki (1994), 新・地震動のスペクトル解析入門 (in Japanese).
    Smoothing is performed using a Parzen window

    f_spec : np.ndarray
        One-dimensional array of the Fourier spectrum to be smoothed.
    band_width : float
        Band-width of the Parzen window (unit: Hz)
    delta_f : float
        Frequency increment. Defined as `df = 1 / (N * dt)`
    f_max : float
        Frequency range used to construct the window.
    """

    # Constant
    u = 280 / (band_width * 151)

    # Base kernel
    kernel_range = np.arange(-f_max, f_max + delta_f, delta_f)

    # Window function
    kernel = 0.75 * u * (np.sin(0.5 * np.pi * u * kernel_range) / (0.5 * np.pi * u * kernel_range)) ** 4

    # Apply a correction when f = 0 occurs, as the above equation cannot be evaluated in that case.
    kernel[kernel_range == 0] = 0.75 * u

    smooth_spec = np.convolve(f_spec, kernel, mode="same") * delta_f

    return smooth_spec
