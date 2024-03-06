#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the RandomFilmStack
class.

Usage
---------
>>> python -m unittest -v tests.test_randstack
"""

# import external packages
import unittest
from pathlib import Path
import os
import json
from tff_lib import ThinFilm, OpticalMedium

# class under test
from tff_lib import RandomFilmStack

class TestRandomFilmStack(unittest.TestCase):
    """
    Test suite for RandomFilmStack() class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # static navigation to data directory and output directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data')
        cls.data_file = os.path.join(cls.dir, 'test_data.json')

        # read in json file with test input data
        with open(cls.data_file, 'r', encoding='utf=8') as dat:
            cls.test_data = json.load(dat)

        # setup input data from test_expected.json
        cls._waves = cls.test_data['input']['wv']
        cls._high = [complex(x) for x in cls.test_data['input']['high_mat']]
        cls._low = [complex(x) for x in cls.test_data['input']['low_mat']]
        cls._layers = cls.test_data['input']['layers']

        # generate test FilmStack
        cls._stack = [
            ThinFilm(
                cls._waves,
                cls._high if lyr[0] == 1 else cls._low,
                thick=lyr[1],
                ntype=lyr[0]
            )
            for lyr in cls._layers
        ]

        # test kwargs (these are different from defaults)
        cls._kwargs = {
            'max_total_thick': 25_000,
            'max_layers': 20,
            'min_layers': 2,
            'first_lyr_min_thick': 250,
            'min_thick': 50
        }

    def test_rand_stack_init_defaults(self):
        """
        test __init__()
        """

        RandomFilmStack(self._waves, self._high, self._low)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
