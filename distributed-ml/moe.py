"""
This module contains functions used to simulate MOE performance
and error. Depends on metrics.py and utils.py.

Function Definitions
--------------------
>>> performance()
>>> ssq_prediction_error()
>>> opt_callback()
>>> run_optimizer()
"""

# import dependencies
from time import perf_counter
from typing import Dict, Tuple, Iterable, Any
import numpy as np
from numpy.typing import NDArray
from random import uniform
from scipy.optimize import minimize
from tff_lib import ThinFilmFilter, OpticalMedium, ThinFilm, FilmStack
from .dataset import Dataset
from .utils import (regression_vector,
                    detector_response,
                    root_mean_squared_err,
                    ml_signal_to_noise,
                    roc_curve,
                    calibrate_response,
                    iprint)

class MOE(ThinFilmFilter):
    """
    MOE is a sub-class of ThinFilmFilter. MOE inherits all public
    properties, attributes, and methods of ThinFilmFilter, while
    providing additional attributes and methods used in MOE-specific
    thin-film filter modeling applications.

    Methods
    ----------
    >>> def performance(
                self,
                dat: Dataset,
                angles: Iterable[float],
                **kwargs
        ) -> Dict[str, NDArray]

    See Also
    ----------
    >>> class ThinFilmFilter(
            sub: OpticalMedium,
            stack: FilmStack,
            inc: OpticalMedium
        )
    """

    def __init__(
            self,
            sub: OpticalMedium,
            stack: FilmStack,
            inc: OpticalMedium,
    ) -> None:
        """
        Initializes the MOE class by calling parent __init__().
        """

        super().__init__(sub, stack, inc)

    def performance(
            self,
            dat: Dataset,
            angles: Iterable[float],
            **kwargs
    ) -> Dict[str, NDArray]:
        """
        Calculates the performance of the MOE across range of
        angles based on a thin film model.

        args
        ----------
        dat: Dataset, dataset object
        angles: Iterable[float], range of incident radiation angles

        kwargs
        ----------
        analysis_type: str, one of 'classification' or 'concentration'
        opt_comp: int, one of range[0, 5]
        fit_order: int, polynomial fit order
        use_ei: bool, environmental interference flag

        Returns
        --------
        Dict[str, NDArray],
         'sec':                Standard error of calibration for the filter design,\n
         'sep':                Standard error of prediction for the filter design,\n
         'yhatcal':            Predicted concentration values - calibration set,\n
         'yhatval':            Predicted concentration values - validation set,\n
         'gain':               Gain for the filter design,\n
         'offset':             Offset for the filter design,\n
         'a_over_b_cal':       Detector ratio response for calibration spectra,\n
         'a_over_b_val':       Detector ratio response for validation spectra,\n
         'snr':                Model limited Signal-to-noise ratio of instrument from AdivB,\n
         'delta_a_over_b':     signal change in percent for 1 unit (eg. unit = 1% moisture),\n
         'delta_a_over_b_ei':  signal change in percent over the environmental interference,\n
         'reg_vector':         optical computing regression vector

        See Also
        ----------
        >>> class Dataset(path: str | Path, dstype: str, **kwargs)
        """

        ################################################################################

        reg_vec = np.zeros((len(angles), len(dat.data['waves']))).astype(np.complex128)     # optical computing regression vector [num array]
        a_over_b_cal = np.zeros((len(dat.data['ycal']), len(angles))).astype(np.complex128) # Detector ratio response for calibration spectra [num array]
        a_over_b_val = np.zeros((len(dat.data['yval']), len(angles))).astype(np.complex128) # Detector ratio response for validation spectra [num array]
        y_hat_cal = np.zeros((len(dat.data['ycal']), len(angles))).astype(np.complex128)    # Predicted concentration values - calibration set [num array]
        y_hat_val = np.zeros((len(dat.data['yval']), len(angles))).astype(np.complex128)    # Predicted concentration values - validation set [num array]

        # 1-D arrays matching num of angles
        gain = np.zeros(len(angles)).astype(np.complex128)                     # Gain for the filter design [num]
        offset = np.zeros(len(angles)).astype(np.complex128)                   # offset for the filter design
        sec = np.zeros(len(angles)).astype(np.complex128)                      # <----------------------- moved to metrics.py
        sep = np.zeros(len(angles)).astype(np.complex128)                      # <----------------------- moved to metrics.py
        snr = np.zeros(len(angles)).astype(np.complex128)                      # <----------------------- moved to metrics.py
        delta_a_over_b = np.zeros(len(angles)).astype(np.complex128)           # signal change in percent for 1 unit (eg. unit = 1% moisture) [num]
        delta_a_over_b_ei = np.zeros(len(angles)).astype(np.complex128)        # signal change in percent over the environmental interference [num]

        #################################################################################
        c_auroc = np.zeros(len(angles))
        v_auroc = np.zeros(len(angles))

        # combine y-calibration and y-validation
        y_cal_val = np.concatenate((dat.data['ycal'], dat.data['yval']), axis=0)

        # calculate the MOE performance values across all incident angles
        for i, theta in enumerate(angles):

            # compute the filter spectrum @ theta
            spec = self.filter_spectrum(float(theta))

            # optical computation regression vector
            reg_vec[i, :] = regression_vector(dat.data['waves'],
                                              spec['T'],
                                              spec['R'],
                                              kwargs.get('opt_comp'))

            # detector response and A/B detector response
            det_resp = detector_response(reg_vec[i, :],
                                         spec['T'],
                                         dat.data['xcal'],
                                         dat.data['xval'])
            y_cal_det, y_val_det, a_over_b_cal[:, i], a_over_b_val[:, i] = det_resp

            # calibrate the MOE system response and compute gain & offset
            cal_resp = calibrate_response(a_over_b_cal[:, i],
                                          a_over_b_val[:, i],
                                          dat.data['ycal'],
                                          y_cal_det,
                                          y_val_det,
                                          **kwargs)
            gain[i], offset[i], y_hat_cal[:, i], y_hat_val[:, i] = cal_resp

            # Calculate the delta A/B per unit value
            a_div_b = np.concatenate((np.expand_dims(a_over_b_cal[:, i], axis=1),
                                      np.expand_dims(a_over_b_val[:, i], axis=1)), axis=0)
            a_div_b = a_div_b / np.mean(a_div_b, axis=0)
            _, delta_a_over_b[i] = np.polyfit(np.squeeze(y_cal_val), np.squeeze(a_div_b), 1)

            # Calculate the delta A/B using environmental
            # interference, if requested
            if kwargs.get('use_ei', False):

                delta_a_over_b_ei[i] = np.abs(
                    1 - (np.dot(dat.data['resp_ei'], np.transpose(spec['T'])) / np.sum(dat.data['resp_ei']))
                    / (np.dot(dat.data['resp'], np.transpose(spec['T'])) / np.sum(dat.data['resp'])))

            # Calculate the standard err. of cal. (sec) & pred. (sep)
            sec[i] = root_mean_squared_err(dat.data['ycal'], y_hat_cal[:, i])
            sep[i] = root_mean_squared_err(dat.data['yval'], y_hat_val[:, i])

            # Calculate the model limited signal-to-noise ratio
            snr[i] = ml_signal_to_noise(gain[i], sec[i], sep[i], dat.data['spec_mean'])

            # Classification (AUROC) replaces SEC & SEP with (1 - AUROC)
            if kwargs.get('analysis_type') == 'classification':

                # Calculate the ROC performance
                c_auroc[i] = roc_curve(dat.data['ycal'], y_hat_cal[:, i], np.arange(0, 1, 0.1))['AUROC']
                v_auroc[i] = roc_curve(dat.data['yval'], y_hat_val[:, i], np.arange(0, 1, 0.1))['AUROC']

                # Update sec and sep with 1 - AUROC value
                sec[i] = 1 - c_auroc[i]
                sep[i] = 1 - v_auroc[i]

        return {
            'sec': sec,
            'sep': sep,
            'y_hat_cal': y_hat_cal,
            'y_hat_val': y_hat_val,
            'gain': gain,
            'offset': offset,
            'a_over_b_cal': a_over_b_cal,
            'a_over_b_val': a_over_b_val,
            'snr': snr,
            'delta_a_over_b': delta_a_over_b,
            'delta_a_over_b_ei': delta_a_over_b_ei,
            'reg_vec': reg_vec,
            'fil_spec': self.filter_spectrum(float(0.0))
        }

###    def _callback(self, xk):
###        """
###        Private callback function used by minimization function. Checks
###         design limitations of the film stack and sets exit_code
###         before terminating optimization.
###
###        Parameters
###        ------------
###        xk: Iterable[Any], current parameter vector
###
###        See Also
###        -------------
###        Scipy.optimize.minimize, OptimizeResult
###        """
###
###        # update layer thickness values
###        #self.stack.layers = xk
###        iprint(f"[DEBUG] OPT VECTOR -->  {xk}")
###
###        # check total thickness <= max_thick
###        ## if self.stack.total_thick <= self.stack.max_total_thick:
###        ##     self.fault_code = 0
###        ##     return True
###
###        # check first layer min thickness  == exit_code 1
###        ## if self.stack.layers[0] <= self.stack.first_lyr_min_thick:
###        ##     self.fault_code = -1
###        ##     return True
###
###        # set fault codes and _idx for failing layers
###        for idx, lyr in enumerate(xk):
###            iprint(f"Validating layer thicknesses..")
###
###            # verify values are > min_thick
###            if lyr <= self.stack.min_thick:
###                iprint(f"[DEBUG] lyr at index= {idx} <= min_thick]")
###                self.fault_code = -2
###                self.idx = idx
###                return True
###
###            # verify values are < max_thick
###            if lyr >= self.stack.max_thick:
###                self.fault_code = -3
###                self.idx = idx
###                return True
###
###        t_now = perf_counter()
###        if (t_now - self.start) >= self.timeout:
###            self.fault_code = -4
###            return True
###
###        # update stack if no faults
###        self.stack.layers = xk
###
###        return False
###
###    def _optimize(
###            self,
###            dat: Dataset,
###            angles: Iterable[float],
###            sim: Dict[str, Any],
###            err: Dict[str, Any],
###            **kwargs
###        ) -> Dict[str, NDArray]:
###        """
###        Minimizes the SSQ prediction error of the MOE.
###
###        args
###        ----------
###        dat: Dataset, dataset object
###        angles: Iterable[float], range of incident radiation angles
###        sim: Dict[str, Any], simulation **kwargs passed through to performance method
###        err: Dict[str, Any], error function **kwargs passed in by optimizer
###
###        kwargs
###        ----------
###        max_iters: int, maximum iterations for optimizer
###        tolerance: float, optimizer tolerance, default 2.5e-5
###        timeout: float, timeout in minutes
###        ratio: float, ratio to split layers that are too thick
###
###        Returns
###        ----------
###        Dict[str, NDArray], performance info for optimized MOE filter stack
###        """
###
###        # exit condition reached
###        done = False
###
###        # configure timer
###        self.timeout = float(kwargs.get('timeout', 2)) * 60
###        self.start = perf_counter()
###
###        # pop optimizer params from kwargs
###        max_iters = int(kwargs.get('max_iters', 250))
###        tolerance = float(kwargs.get('tolerance', 2.5e-5))
###        ratio = float(kwargs.get('ratio', 0.5))
###
###        while not done:
###
###            # reset exit condition
###            self.fault_code = 1
###
###            # run the optimizer
###            iprint("Entering optimizer")
###            iprint(f"[DEBUG] Input layers --> {self.stack.layers}")
###            result = minimize(self.ssq_prediction_error,
###                            self.stack.layers,
###                            args=(dat, angles, sim, err),
###                            callback=self._callback,
###                            tol=tolerance,
###                            options={'maxiter': max_iters, 'disp': True})
###
###            # update the layers in the stack
###            self.stack.layers = result.x
###            iprint(f"[DEBUG] OUTPUT: {result.x}")
###            iprint(f"[DEBUG] MESSAGE: {result.message}")
###            iprint(f"[DEBUG] N_ITERS: {result.nit}")
###
###            # exit immediately on success, timeout,
###            # max_total_thick exceeded, or first layer
###            # min thick reached
###            if self.fault_code in (-4, -1, 0, 1):
###                iprint(MOE.messages[self.fault_code])
###                done = True
###                break
###
###            # handle minimum thick limit for other layers
###            if self.fault_code == -2:
###                iprint(MOE.messages[self.fault_code])
###
###                # make sure removing layer does not violate min layers
###                # must have at least n + 2 layers remaining in order
###                # to remove one
###                if self.stack.num_layers < self.stack.min_layers + 2:
###                    iprint("Optimization terminated - Too few layers remaining")
###                    done = True
###                    break
###
###                # remove the layer from the stack and restart loop
###                iprint(f"Removing layer at index [{self.idx}].")
###                self.stack.remove_layer(self.idx)
###                self.idx = None
###                continue
###
###            # handle max layer thick limit
###            if self.fault_code == -3:
###                iprint(MOE.messages[self.fault_code])
###
###                # num_layers should not exceed max_layers - 2 in order
###                # for split_layers to be used
###                if self.stack.num_layers > self.stack.max_layers - 2:
###                    iprint("Optimization terminated -- Too many layers remaining")
###                    done = True
###                    break
###
###                # insert a new layer with random thick and split layer that
###                # is too thick using default ratio. New layer must be
###                # opposite ntype of layer being split
###                iprint(f"Splitting layer at index [{self.idx}].")
###                biglyr = self.stack.get_layer(self.idx)
###                lyr = ThinFilm(dat.data['waves'],
###                                dat.data['low'] if biglyr.ntype == 0 else dat.data['high'],
###                                thick=uniform(self.stack.min_thick, self.stack.max_thick),
###                                ntype=1 if biglyr.ntype == 0 else 0)
###                self.stack.insert_split_layer(lyr, self.idx, ratio)
###                continue
###
###        # return the final performance
###        return self.performance(dat, angles, **sim)
