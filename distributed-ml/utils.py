"""
This module contains a set of utility functions used in
MOE design applications.

Function Definitions
--------------------
>>> generate_sine_wave(
        start:int,
        end:int,
        **kwargs
    ) -> Tuple
>>> ml_signal_to_noise(
        gain: float,
        sec: float,
        sep: float,
        spec_mean: floa
    ) -> float
>>> root_mean_squared_err(
        vals: Iterable[float],
        preds: Iterable[float]
    ) -> float
>>> relative_resp_factor(
        trsm: Iterable[float],
        x_cal: Iterable[float]
    ) -> NDArray
>>> theta_range(
        inc_theta:float,
        fov:float,
        n_angles:int=7
    ) -> NDArray
>>> detector_response(
        reg_vec: Iterable[float],
        trsm: Iterable[float],
        x_cal: Iterable[float],
        x_val: Iterable[float]
    ) -> Tuple
>>> regression_vector(
        waves: Iterable[float],
        trsm: Iterable[float],
        refl: Iterable[float],
        opt_comp: int
    ) -> NDArray
>>> roc_curve(
        truths: Iterable[float],
        detects: Iterable[float],
        thresh: Iterable[float]
    ) -> Dict[str, NDArray]
>>> def ssq_prediction_error(
        y_cal_val: NDArray,
        perf: Dict[str, NDArray],
        **kwargs
    ) -> float
"""

# import dependencies
from datetime import datetime as dt
import math
from socket import gethostname, gethostbyname
import sys
from typing import Tuple, Dict, Iterable
import numpy as np
from numpy.typing import NDArray

def get_host_info():
    """
    Return IP address and hostname in ISO format.
    """
    hostname = gethostname()
    ip_addr = gethostbyname(hostname)
    return f"{ip_addr} {hostname}"

def now_iso() -> str:
    """
    Returns the current time in ISO format
     as a string.
    """
    return str(dt.now().isoformat())

def iprint(msg: str) -> int:
    """
    Prints an ISO formatted message to output stream
     and flushes the stdout buffer.

    args
    ----------
    msg: str, message to print

    Returns
    ----------
    int, length of string written to stdout
    """

    # write msg and flush buffer
    out = sys.stdout.write(f"{get_host_info()} [{now_iso()}] \"{msg}\" {len(msg)}\n")
    sys.stdout.flush()
    return out

def generate_sine_wave(start:int, end:int, **kwargs) -> Tuple:
    """
    Generates a sine wave signal over a specified wavelength range.

    args
    ----------
    start: int, starting x value
    end: int, ending x value

    kwargs
    ----------
    steps: int, number of steps
    freq: numerical, sine frequency, default=1
    amp: numerical, wave amplitude, default=1
    theta: numerical, angle, default=0
    shift: numerical, phase shift, default=0

    Returns
    ----------
    Tuple(wavelengths, values)
    """

    # create a sine wave for testing
    wvl = np.arange(start, end, kwargs.get('steps', 1))

    # amp * sin(freq * wvl + angle) + shift
    values = kwargs.get('amp', 0) * np.sin(
        kwargs.get('freq', 0) * wvl + kwargs.get('angle', 0)) + kwargs.get('shift', 0)

    return wvl, values

def ml_signal_to_noise(gain: float, sec: float, sep: float, spec_mean: float) -> float:
    """
    Calculates the model-limited signal-to-noise ratio.

    Parameters
    ----------
    gain: float, MOE throughput
    sec: float, standard error of calibration
    sep: float, standard error of prediction
    spec_mean: float, spectral mean

    Returns
    ----------
    float, model-limited SNR scaled by the MOE throughput
    """
    return np.abs(gain) / (spec_mean * np.mean([sec, sep]))

def root_mean_squared_err(vals: Iterable[float], preds: Iterable[float]) -> float:
    """
    Calculates the root mean squared error.

    Parameters
    ----------
    vals: Iterable[float], 1-D ground truth values
    preds: Iterable[float], 1-D predicted values

    Returns
    ----------
    float, root mean squared error
    """
    return np.sqrt(np.sum((vals - preds)**2) / np.size(vals))

def relative_resp_factor(trsm: Iterable[float], x_cal: Iterable[float]) -> NDArray:
    """
    Calcualates the relative response factor.

    Parameters
    ----------
    trsm: Iterable[float], 1-D transmission values
    x_cal: Iterable[float], 2-D x-calibration data

    Returns
    ----------
    NDArray, relative response factor
    """
    return np.squeeze(np.dot(x_cal, np.transpose(np.real(trsm)))) / np.sum(x_cal, axis=1)

def theta_range(inc_theta:float, fov:float, n_angles:int=7) -> NDArray:
    """
    Calculates a range of angles centered at the incident angle
    out to +/- half the field of view.

    Parameters
    ----------
    inc_angle: float, angle of incidence.
    fov: float, field of view centered at incident angle (degrees).
    n_angles: int, the number of points to generate. (default = 7)

    Returns
    --------
    NDArray, discrete angles from +/- (FOV/2), length n_points

    Notes
    --------
    Variance increases as incident angle goes further
    away from 0 degrees.
    """

    # *******   possibly would need variable sampling base on ******  #
    # *******   beam distribution.. i.e. 'Gaussian' beam here ******  #

    # define start/stop values
    start = inc_theta - (fov / 2)
    stop = inc_theta + (fov / 2)

    return np.linspace(start, stop, num=n_angles) if fov > 0 else np.array([inc_theta])

def detector_response(
        reg_vec: Iterable[float],
        trsm: Iterable[float],
        x_cal: Iterable[float],
        x_val: Iterable[float]
    ) -> Tuple:
    """
    Calculates the detector response.

    Parameters
    ----------
    reg_vec: Iterable[float], 1-D regression vector
    trsm: Iterable[float], 1-D spectral transmission
    x_cal: Iterable[float], 2-D x-calibration spectra
    x_val: Iterable[float], 2-D x-validation spectra

    Returns
    ----------
    Tuple,
        (NDArray) y-calibration detector values,\n
        (NDArray) y-validation detector values,\n
        (NDArray) a-over-b calibration response,\n
        (NDArray) a-over-b validation response

    See Also
    ----------
    >>> moepy.utils.regression_vector()
    """

    # calculate detector response
    y_cal_det = np.dot(x_cal, np.transpose(reg_vec))
    y_val_det = np.dot(x_val, np.transpose(reg_vec))

    # calculate A / B detector response
    a_over_b_cal= (np.dot(x_cal, np.transpose(trsm)) / np.sum(x_cal, axis=1))
    a_over_b_val= (np.dot(x_val, np.transpose(trsm)) / np.sum(x_val, axis=1))

    return y_cal_det, y_val_det, a_over_b_cal, a_over_b_val

def calibrate_response(
        a_over_b_cal,
        a_over_b_val,
        y_cal,
        y_cal_det,
        y_val_det,
        **kwargs
    ) -> Tuple:
    """
    Calibrates the MOE gain, offset, and response.

    args
    ----------
    a_over_b_cal: ArrayLike, Detector ratio response for calibration spectra
    a_over_b_val: ArrayLike, Detector ratio response for validation spectra
    y_cal: ArrayLike, y-calibration values
    y_cal_det: ArrayLike, y-calibration detector values
    y_val_det: ArrayLike, y-validation detector values

    kwargs
    ----------
    fit_order: int, polynomial order for fit function, default=1
    opt_comp: int, optical computation, default=4

    Returns
    ----------
    Tuple(float, float, ArrayLike, ArrayLike)
        gain: float, Gain for the filter design
        offset: float, Offset for the filter design
        yhatcal: array, Predicted concentration values - calibration
        yhatval: array, Predicted concentration values - validation

    """

    fit_order = kwargs.get('fit_order', 1)
    opt_comp = kwargs.get('opt_comp', 4)

    # calibrate the MOE system response and compute gain & offset
    # finalize the MOE prediction responses
    if fit_order == 1:

        if opt_comp == 4:
            gain, offset = np.polyfit(a_over_b_cal, y_cal, fit_order)
            y_hat_cal = gain * a_over_b_cal + offset
            y_hat_val = gain * a_over_b_val + offset

        else:
            gain, offset = np.polyfit(y_cal_det, y_cal, fit_order)
            y_hat_cal = gain * y_cal_det + offset
            y_hat_val = gain * y_val_det + offset

    elif fit_order == 2:

        if opt_comp == 4:
            # Linear fit gain is used for SNR purposes
            _, gain = np.polyfit(a_over_b_cal, y_cal, 1)
            a_fit, b_fit, offset = np.polyfit(a_over_b_cal, y_cal, fit_order)
            y_hat_cal = a_fit * a_over_b_cal**2 + b_fit * a_over_b_cal + offset
            y_hat_val = a_fit * a_over_b_val**2 + b_fit * a_over_b_val + offset

        else:
            _, gain = np.polyfit(y_cal_det, y_cal, 1)
            a_fit, b_fit, offset = np.polyfit(y_cal_det, y_cal, fit_order)
            y_hat_cal = a_fit * y_cal_det**2 + b_fit * y_cal_det + offset
            y_hat_val = a_fit * y_val_det**2 + b_fit * y_val_det + offset

    return gain, offset, y_hat_cal, y_hat_val

def regression_vector(
        waves: Iterable[float],
        trsm: Iterable[float],
        refl: Iterable[float],
        opt_comp: int
    ) -> NDArray:
    """
    Computes a regression vector based on the selected
     optical computation.

    Parameters
    ----------
    waves: Iterable[float], 1-D wavelength array
    trsm: Iterable[float], 1-D transmission spectrum
    refl: Iterable[float], 1-D reflection spectrum
    opt_comp: int, optical computation, range [0, 5]

    Returns
    ----------
    NDArray, regression vector
    """

    # calculation based on 'opt_comp' parameter
    if int(opt_comp) == 0:
        return np.asarray(trsm - refl)

    if int(opt_comp) == 1:
        return np.asarray((trsm - refl) / (trsm + refl))

    if int(opt_comp) == 2:
        return np.asarray(trsm - .5 * refl)

    if int(opt_comp) == 3:
        return 2 * trsm - np.ones(np.size(waves)).astype(complex)

    # case 4, 5
    return np.asarray(trsm)

def roc_curve(
        truths: Iterable[float],
        detects: Iterable[float],
        thresh: Iterable[float]
    ) -> Dict[str, NDArray]:
    """
    Calculate the Receiver Operator Characteristic (ROC) curve.

    Parameters
    -----------
    truths: Iterable[float], 1-D logical index or binary values indicating class membership
            detections - measured results (or prediction scores) from sensor
    detects: Iterable[float], 1-D measured results (or prediction scores) from sensor
    thresh: Iterable[float], 1-D user specified threshold values for computing the ROC curve

    Returns
    ---------
    Dict[str, NDArray],
        'AUROC' : Area Under the Receiver Operator Curve,\n
        'Pd' : Probability of detection (or sensitivity),\n
        'Pfa' : Probability of false alarm (or 1 - specificity),\n
        't_ind' : index of optimal threshold,\n
        't_val' : Optimal threshold based on distance to origin,\n
        'Se' : Optimal sensitivity based upon optimal threshold,\n
        'Sp' : Optimal specificity based upon optimal thresh
    """

    # define a confusion matrix as a dict
    roc_matrix = {
        'true_pos':np.zeros(len(thresh)),
        'true_neg':np.zeros(len(thresh)),
        'false_pos':np.zeros(len(thresh)),
        'false_neg':np.zeros(len(thresh)),
        'prob_det':np.zeros(len(thresh)),
        'prob_fa':np.zeros(len(thresh))
    }

    # Run loop to threshold detections data and calculate TP, TN, FP & FN
    for i, val in enumerate(thresh):

        temp_detects = np.asarray([1 if d >= val else 0 for d in detects])
        tp_temp = np.sum(truths * temp_detects)

        roc_matrix['true_neg'][i] = np.sum((1 - truths) * (1 - temp_detects))
        roc_matrix['false_pos'][i] = np.sum(temp_detects) - tp_temp
        roc_matrix['false_neg'][i] = np.sum(truths) - tp_temp
        roc_matrix['true_pos'][i] = tp_temp

    # Calculate Pd and Pfa
    roc_matrix['prob_det'] = (
        roc_matrix['true_pos'] / (roc_matrix['true_pos'] + roc_matrix['false_neg']))
    roc_matrix['prob_fa'] = (
        1 - (roc_matrix['true_neg'] / (roc_matrix['false_pos'] + roc_matrix['true_neg'])))

    # map points to distance from upper left corner (0, 1)
    euc_dist = list(
        map(lambda x: math.sqrt((0 - x[0])**2 + (1 - x[1])**2),
            zip(roc_matrix['prob_fa'], roc_matrix['prob_det'])))

    # Find the best threshold index
    t_ind = np.argmin(euc_dist)

    # Calculate the AUROC using a simple summation of rectangles
    au_roc = -np.trapz(
        np.append(roc_matrix['prob_det'], 0), np.insert(roc_matrix['prob_fa'], 0, 1))

    return {
        'AUROC': au_roc,                            # area under roc curve
        'Pd':    roc_matrix['prob_det'],            # probability of detection
        'Pfa':   roc_matrix['prob_fa'],             # probability of false alarm
        't_val': thresh[t_ind],                     # roc threshold value
        'Se':    roc_matrix['prob_det'][t_ind],     # sensitivity
        'Sp':    1 - roc_matrix['prob_fa'][t_ind]   # specificity
    }

def ssq_prediction_error(y_cal_val: NDArray, perf: Dict[str, NDArray], **kwargs) -> float:
    """
    Calculates the SSQ prediction error ('msq') of an MOE. First arg must be
        an Iterable in order to use SciPy Optimizer.

    args
    ----------
    y_cal_val: NDArray, combined Y-Calibration and Y-Validation values
    perf: Dict[str, NDArray], results of MOE.performance() method

    kwargs
    ----------
    opt_comp: int, one of range[0, 6]
    fom: int, figure of merit
    ab_thresh: float, A over B per unit
    sec_thresh: float, SEC threshold value
    snr_thresh: float, SNR threshold value
    total_thick: float, total filter thickness if using FOM = 6

    Returns
    --------
    float, ssq prediction error based on figure of merit

    See Also
    ----------
    >>> MOE.performance(
            dat: Dataset,
            angles: Iterable[float],
            **kwargs
        ) -> Dict[str, NDArray]
    >>> scipy.optimize.minimize(fun, x0, **kwargs)
    """

    # extract kwargs used in error calculation
    fom = int(kwargs.get('fom'))
    sec_thresh = float(kwargs.get('sec_thresh'))
    snr_thresh = float(kwargs.get('snr_thresh'))
    ab_thresh = float(kwargs.get('ab_thresh'))
    opt_comp = int(kwargs.get('opt_comp'))


    # Calculate the mean squared error
    # determined by the figure of merit
    if fom == 0:
        sec_msq = (
            (np.mean(perf['sec']) - sec_thresh) /
            (0.33 * np.max(y_cal_val) - np.min(y_cal_val) - sec_thresh)
        )

        return max(sec_msq, 0)

    if fom == 1:
        if opt_comp == 0:
            rv_scale = -100 * np.sum(perf['reg_vec'])

        elif opt_comp == 1:
            rv_scale = -np.sum(perf['reg_vec'])

        elif opt_comp == 2:
            rv_scale = (100 * (np.sum(np.abs(perf['reg_vec']))
                    + np.sum(perf['reg_vec']))
                    / 2 - 50 * np.sum(perf['reg_vec']))

        elif opt_comp == 3:
            rv_scale = (200 * (np.sum(np.abs(perf['reg_vec']))
                    + np.sum(perf['reg_vec']))
                    / 2 - 100 * np.sum(perf['reg_vec']))

        elif opt_comp == 4:
            rv_scale = (200 * (np.sum(np.abs(perf['reg_vec']))
                    + np.sum(perf['reg_vec']))
                    / 2 - 100 * np.sum(perf['reg_vec']))

        return ((np.mean(perf['sec'])
            / sec_thresh)
            * rv_scale / np.abs(perf['reg_vec']
            * np.transpose(perf['reg_vec'])))

    if fom == 2:
        sec_msq = ((np.mean(perf['sec'])
                - sec_thresh)
                / (0.33 * (np.max(y_cal_val)
                - np.min(y_cal_val))
                - sec_thresh))
        snr_msq = ((np.mean(perf['snr'])
                - snr_thresh)
                / (10 * snr_thresh))
        sec_max = max(np.max([sec_msq]), 0)
        snr_max = max(np.max([snr_msq]), 0)
        return sec_max + snr_max

    if fom == 3:

        if opt_comp == 0:
            rv_scale = -100 * sum(perf['reg_vec'])

        if opt_comp == 1:
            rv_scale = -sum(perf['reg_vec'])

        if opt_comp == 2:
            rv_scale = (100 * (sum(abs(perf['reg_vec']))
                    + sum(perf['reg_vec']))
                    / 2 - 50 * sum(perf['reg_vec']))

        if opt_comp == 3:
            rv_scale = (200 * (sum(abs(perf['reg_vec']))
                    + sum(perf['reg_vec']))
                    / 2 - 100 * sum(perf['reg_vec']))

        if opt_comp == 4:
            rv_scale = (200 * (sum(abs(perf['reg_vec']))
                    + sum(perf['reg_vec']))
                    / 2 - 100 * sum(perf['reg_vec']))

        return ((np.mean(perf['sec'])
            / sec_thresh)
            * (rv_scale / abs(perf['reg_vec']
            * np.transpose(perf['reg_vec'])))
            + (np.mean(perf['snr'])
            / snr_thresh))

    if fom == 4:
        sec_msq = np.mean(perf['sec']) - sec_thresh
        ab_msq = ab_thresh - np.mean(perf['delta_a_over_b'])
        sec_max = max(np.max([sec_msq]), 0)
        ab_max = max(np.max([ab_msq]), 0)
        return sec_max + ab_max

    if fom == 5:
        sec_msq = ((np.mean(perf['sec']) - sec_thresh)
                / (0.33 * (np.max(y_cal_val) - np.min(y_cal_val)) - sec_thresh))
        snr_msq = ((np.mean(perf['snr']) - snr_thresh) / (10 * snr_thresh))
        sec_max = max(np.max([sec_msq]), 0)
        snr_max = max(np.max([snr_msq]), 0)
        return sec_max + snr_max + np.mean(perf['delta_a_over_b_ei'])

    if fom == 6:
        sec_msq = ((np.mean(perf['sec']) - sec_thresh)
                / (0.33 * (np.max(y_cal_val) - np.min(y_cal_val)) - sec_thresh))
        sec_max = max(sec_msq, 0)
        thickmsq = np.abs(kwargs.get('total_thick') / 1000 - 2.5)
        return sec_max + thickmsq / 2.5

def epsilon_decay(min_epsilon, max_epsilon, decay, episode):
    """
    Decay function used in Epsilon Greedy algorithm.
    """
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)