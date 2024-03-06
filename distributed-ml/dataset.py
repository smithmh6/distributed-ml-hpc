"""
This module contains functions used for reading
datasets used in MOE designs.

Function Definitions

"""

# import dependencies
from ast import literal_eval
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
from numpy.typing import NDArray
import pandas as pd
## from scipy import interpolate

class Dataset():
    """
    Dataset object used in MOE design applications.

    Attributes
      path: str | Path, dataset file path
      dstype: str, one of 'json', 'xlsx', 'npz', 'csv'
      cols: Dict, column mapping of input data file
      data: Dict[str, Iterable], dataset used in MOE design

    Methods
    ----------
    >>> interpolate_1d(self, **kwargs) -> None
    >>> spectral_mean(self, thresh: float = -1) -> None
    >>> normalize_spectra(self, axis: int = 1) -> None
    >>> radiometric_response(self, use_ei: bool = False) -> None
    >>> save_dataset(self, out: Path) -> None
    """

    default_column_map = {
        'waves': 0,
        'lamp': 1,
        'det': 2,
        'comp_0': 3,
        'comp_1': 4,
        'comp_2': 5,
        'comp_3': 6,
        'comp_4': 7,
        'comp_5': 8,
        'comp_6': 9,
        'comp_7': 10,
        'comp_8': 11,
        'comp_9': 12,
        'reg_vec': 13,
        'sub': 14,
        'high': 15,
        'low': 16,
        'env_int': 17,
        'cal': 18,
        'val': -1
    }

    def __init__(self, path: str | Path, dstype: str, **kwargs) -> None:
        """
        Initialize the Dataset class.

        args
        ----------
        path: str | Path, dataset file path
        dstype: str, one of 'json', 'xlsx', 'npz', 'csv'

        kwargs
        ----------
        cols: Dict[str, int], column mapping of input data file
        interp1d: bool, determines if interpolation is required
        wv_min: float, min wavelength for interpolation
        wv_max: float, max wavelength for interpolation
        spacing: int, wavelength interval for interpolation, defaults to 1
        kind: str, one of 'linear', 'nearest', 'nearest-up', 'zero',
          'slinear', 'quadratic', 'cubic', 'previous', 'next' (defaults to 'linear')

        Notes
        ----------
        The `data` attribute contains the dataset arrays in a Dict[str, NDArray]
        object. `data` contains the following key-value pairs by default:\n
          'waves' : wavelength data, float64\n
          'lamp' : lamp spectral data, complex128\n
          'det' : detector spectral data, complex128\n
          'comp_0' : component spectral data, complex128\n
          'comp_1' : component spectral data, complex128\n
          'comp_2' : component spectral data, complex128\n
          'comp_3' : component spectral data, complex128\n
          'comp_4' : component spectral data, complex128\n
          'comp_5' : component spectral data, complex128\n
          'comp_6' : component spectral data, complex128\n
          'comp_7' : component spectral data, complex128\n
          'comp_8' : component spectral data, complex128\n
          'comp_9' : component spectral data, complex128\n
          'reg_vec' : regression vector, complex128\n
          'sub' : substrate spectral data, complex128\n
          'high' : high index material spectral data, complex128\n
          'low' : low index material spectral data, complex128\n
          'env_int' : environmental interference, complex128\n
          'xcal' : x-calibration data, complex 128\n
          'xval' : x-validation data, complex128\n
          'ycal' : y-calibration data, complex128\n
          'yval' : y-validation data, complex128
        """

        self.path = path
        self.dstype = dstype
        self.cols = kwargs.pop('cols', Dataset.default_column_map)
        self.data = self._read_dataset()

        if literal_eval(kwargs.get('interp1d', 'False')):

            self._interpolate_1d(**kwargs)

    def spectral_mean(self, thresh: float = -1) -> None:
        """
        Calculates average spectral value of calibration and validation
         datasets and scales data by 1/100 if average exceeds threshold.

        Parameters
        -----------
        thresh: float, threshold value (determines how to scale data)

        Notes
        -----------
        Adds key 'spec_mean' to data attribute.
        """

        # calculate average spectral value of the sample data
        avg_spec = np.mean(
            np.concatenate((self.data['xcal'], self.data['xval']), axis=0))

        # check value to determine scaling
        if 0 < thresh <= avg_spec:

            # divide xc & xv by 100
            scale_xc = self.data['xcal'] / 100.0
            scale_xv = self.data['xval'] / 100.0

            # update xcal and xval in dataset
            self.data['xcal'] = scale_xc
            self.data['xval'] = scale_xv

            # re-calculate spectral average
            avg_spec = np.mean(
                np.concatenate((self.data['xcal'], self.data['xval']), axis=0))

        self.data['spec_mean'] = avg_spec

    def normalize_spectra(self, axis: int) -> None:
        """
        Normalizes x-calibration and x-validation data along axis
        and updates data attribute keys `xcal` and `xval`.

        Parameters
        ----------
        axis: int, axis to normalize, must be greater than 0
        """

        if axis > 0:
            # normalize the data along the specified axis
            norm_xc = np.linalg.norm(self.data['xcal'], axis=int(axis))
            norm_xv = np.linalg.norm(self.data['xval'], axis=int(axis))

            # update xcal and xval
            self.data['xcal'] = norm_xc
            self.data['xval'] = norm_xv

    def radiometric_response(self, use_ei: bool = False) -> None:
        """
        Calculates the radiometric response of the
         optical system and convolves with xcal and
         xval data.

        Notes
        ----------
        Updates data attribute with new keys 'resp' and
         'resp_ei' if use_ei is True. Overwrites xcal and
          xval after convolving with response curve.
        """

        # copy the current x-cal and x-val data
        xcal = self.data['xcal']
        xval = self.data['xval']

        # store the radiometric response without EI
        resp = np.ones_like(self.data['waves']).astype(np.complex128)

        # convolve components without EI
        for key, arr in self.data.items():
            if 'comp_' in key or 'lamp' in key or 'det' in key:
                resp *= arr

        # add new entry to data dict
        self.data['resp'] = resp


        if use_ei:
            # calculate response with environmental
            # interference and add entry to data dict
            resp_ei = resp * self.data['env_int']
            self.data['resp_ei'] = resp_ei

            # convolve response curve with cal/val data
            xcal = xcal * resp_ei
            xval = xval * resp_ei

        else:
            # convolve response curve with cal/val data
            xcal = xcal * resp
            xval = xval * resp

        # update xcal and xval
        self.data['xcal'] = xcal
        self.data['xval'] = xval

    def save_dataset(self, out: Path) -> None:
        """
        Save dataset to file.
        """
        with open(out, 'w', encoding='UTF-8') as file:
            file.write("test")

    def _read_dataset(self) -> Dict[str, Iterable]:
        """
        Factory method which dynamically reads dataset files.

        Returns
        ----------
        Dict[str, Iterable]
        """
        if self.dstype == 'csv':
            return self._read_csv()
        if self.dstype == 'xlsx':
            return self._read_xlsx()
        if self.dstype == 'npz':
            return self._read_npz()
        if self.dstype == 'json':
            return self._read_json()

        return None

    def _read_csv(self) -> Dict[str, Iterable]:
        """
        Read a CSV formatted dataset.
        """

        dataset = {k: [] for k in list(self.cols.keys())[:-2]}

        # parse the data file
        with open(self.path, 'r', encoding='utf-8') as file:

            # 1 header line, last line is y labels
            lines = len(file.readlines())

            # reset position
            file.seek(0)

            header = file.readline().strip('\n').split(',')

            # find beginning validation data if not specified
            if self.cols['val'] == -1:
                val = self.cols['cal']

                for head in header[self.cols['cal']:]:
                    if int(head) == 2:
                        self.cols['val'] = val
                        break
                    val += 1

            # allocate array's for xcal, xval, ycal, yval
            dataset['xcal'] = [
                [0 for x in range(lines - 2)]
                for y in range(self.cols['val'] - self.cols['cal'])]
            dataset['xval'] = [
                [0 for x in range(lines - 2)] for y in range(len(header) - self.cols['val'])]
            dataset['ycal'] = [0 for y in range(self.cols['val'] - self.cols['cal'])]
            dataset['yval'] = [0 for y in range(len(header) - self.cols['val'])]

            # parse each line in data file
            for i in range(lines-2):
                line = file.readline().strip('\n').split(',')

                # extract wavelength values as floating points
                dataset['waves'].append(float(line[self.cols['waves']]))

                # extract the optical component/setup data as complex numbers
                for key, arr in list(self.cols.items())[1:-2]:
                    dataset[key].append(complex(line[arr]))

                # extract x-Calibration data as complex numbers
                for j in range(self.cols['val'] - self.cols['cal']):
                    dataset['xcal'][j][i] = complex(line[j + self.cols['cal']])

                # extract x-Validation data as complex numbers
                for j in range(len(header) - self.cols['val']):
                    dataset['xval'][j][i] = complex(line[j + self.cols['val']])

            # extract the Y label values
            y_labels = file.readline().strip('\n').split(',')[18:]
            y_labels = [float(l) for l in y_labels]

            # extract the y-Calibration values
            dataset['ycal'] = y_labels[ : self.cols['val'] - self.cols['cal']]

            # extract y-Validation values
            dataset['yval'] = y_labels[self.cols['val'] - self.cols['cal'] : ]

            # convert all data to numpy.ndarray's
            data = {}
            for key, arr in dataset.items():
                if key not in ('cal', 'val'):
                    data[key] = np.asarray(arr)

            return data

    def _read_xlsx(self) -> Dict[str, Iterable]:
        """
        Read an xlsx formatted dataset.
        """

        # read the excel sheets into dataframes
        dfr = pd.read_excel(self.path)

        # extract the optical components from the second sheet
        data = {
            'waves': [float(x) for x in dfr.iloc[:, 0].dropna().values.tolist()],
            'lamp': [complex(x) for x in dfr.iloc[:, 1].dropna().values.tolist()],
            'det': [complex(x) for x in dfr.iloc[:, 2].dropna().values.tolist()],
            'env_int': [complex(x) for x in dfr.iloc[:, 17].dropna().values.tolist()]
        }

        # extract the remaining optical setup components (cols 3-12)
        for i in range(3, 13):
            com = f'0{(i-2)}' if (i-2) < 10 else f'{(i-2)}'
            data[f'comp_{com}'] = [complex(x) for x in dfr.iloc[:, i].dropna().values.tolist()]

        # extract the substrate and materials data (cols 14, 15, 16)
        # these require extra steps bc they are complex values
        substrate = dfr.iloc[:, 14].dropna().values.tolist()
        if isinstance(substrate[1], str) and 'i' in substrate[1]:
            substrate = [s.replace(' ', '').replace('i', 'j') for s in substrate]
        substrate = [complex(x) for x in substrate]
        data['sub'] = substrate

        high_material = dfr.iloc[:, 15].dropna().values.tolist()
        if isinstance(high_material[1], str) and 'i' in high_material[1]:
            high_material = [s.replace(' ', '').replace('i', 'j') for s in high_material]
        high_material = [complex(x) for x in high_material]
        data['high'] = high_material

        low_material = dfr.iloc[:, 16].dropna().values.tolist()
        if isinstance(low_material[1], str) and 'i' in low_material[1]:
            low_material = [s.replace(' ', '').replace('i', 'j') for s in low_material]
        low_material = [complex(x) for x in low_material]
        data['low'] = low_material

        # extract the spectral data for calibration --
        #   all columns AFTER environmental interference
        #   contain spectral data -- last value in each column
        #   is the classification or concentration --
        #   spectral data begins at column 18 (zero indexed) --
        #   Cal data is marked with '1' and val data with '2'
        data['xcal'], data['ycal'], data['xval'], data['yval'] = [], [], [], []
        for i in range(18, len(dfr.columns)):

            if float(dfr.columns[i]) < 2.0:
                # slice up to last element
                xcal = dfr.iloc[:, i].dropna().values.tolist()
                data['xcal'].append([complex(x) for x in xcal[:-1]])
                # add last element in list to ycal
                data['ycal'].append(xcal[-1])

            elif float(dfr.columns[i]) >= 2.0:
                # slice up to last element
                xval = dfr.iloc[:, i].dropna().values.tolist()
                data['xval'].append([complex(x) for x in xval[:-1]])
                # add last element in list to ycal
                data['yval'].append(xval[-1])

        # convert all arrays to numpy.ndarray's
        dataset = {}
        for key, arr in data.items():
            dataset[key] = np.asarray(arr)

        # return the complete dataset
        return dataset

    def _read_npz(self) -> Dict[str, Iterable]:
        """
        Read .npz data file.
        """
        raise NotImplementedError

    def _read_json(self) -> Dict[str, Iterable]:
        """
        Read .json data file.
        """
        raise NotImplementedError

    def _interpolate_1d(self, **kwargs) -> None:
        """
        1-D interpolation of dataset.

        kwargs
        ------------
        wv_min: float, min wavelength for interpolation
        wv_max: float, max wavelength for interpolation
        spacing: int, x axis interval, defaults to 1
        kind: str, one of 'linear', 'nearest', 'nearest-up', 'zero',
          'slinear', 'quadratic', 'cubic', 'previous', 'next' (defaults to 'linear')

        Raises
        ----------
        ValueError, if wv_min >= wv_max

        See Also
        ------------
        >>> scipy.interpolate.interp1d()
        """

        wv_min = float(kwargs.get('wv_min'))
        wv_max = float(kwargs.get('wv_max'))

        if wv_min >= wv_max:
            raise ValueError('x_max must be greater than x_min.')

        # get spacing interval from kwargs
        ## spacing = int(kwargs.get('spacing', 1))

        # run interpolation
        ## func = interpolate.interp1d(x_arr, y_arr, kind=kwargs.get('kind', 'linear'))

        # create new x array from min, max, spacing
        ## x_new = np.arange(x_min, x_max, spacing)

        # evaluate interpolated function with new x vals
        # update new x,y values in dataset
        ## return x_new, func(x_new)
