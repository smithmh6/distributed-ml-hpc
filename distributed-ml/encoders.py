"""
This module contains custom encoder classes used
to encode data when converting to JSON.
"""

# import dependencies
import json
import numpy as np
import datetime as dt

class DateTimeEncoder(json.JSONEncoder):
    """
    Sub-class JSONEncoder to convert datetime obects
    from queries on the fly.
    """

    def default(self, obj):
        """
        Over-ride default method to add handling
        for datetime objects.
        """

        # check if object is instance of datetime
        if isinstance(obj, (dt.date, dt.datetime)):

            # return iso formatted date
            return obj.isoformat()

class NumpyEncoder(json.JSONEncoder):
    """
    Sub-Class JSON Encoder to encode Numpy arrays.
    """

    def default(self, obj):
        """
        Override default to add handling for numpy objects.
        """

        # return list if obj is ndarray
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # return object otherwise
        return json.JSONEncoder.default(self, obj)
