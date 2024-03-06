"""
This module contains the RandomFilmStack class used in
MOE design to generate a random thin-film stack. This class
is a sub-class of FilmStack from the tff_lib package.
"""

# import dependencies
import random
from typing import Iterable
import numpy as np
from tff_lib import FilmStack, ThinFilm
from .utils import iprint

class RandomMoeStack(FilmStack):
    """
    A randomized MOE film stack with alternating high/low refractive index
    materials. Inherits public attributes, properties, and methods from
    FilmStack(), in addition to some MOE-specific attributes.

    Attributes
    ----------
    waves: Iterable[float], wavelength values in nanometers
    high: Iterable[complex], high-index material
    low: Iterable[complex], low-index material
    scale_factor: float, scaling factor for random layer thicknesses
     computed from spectral resolution upon instantiation

    See Also
    ----------
    >>> class FilmStack(
                films: Iterable[ThinFilm],
                **kwargs
        )
    """

    def __init__(
            self,
            waves: Iterable[float],
            high: Iterable[complex],
            low: Iterable[complex],
            **kwargs
    ) -> None:
        """
        Initializes a randomized ThinFilmStack object by computing the
        estimated total physical thickness of the film stack based on
        the spectral resolution and the high/low index materials to
        apply the appropriate scaling factor to each random film
        thickness.

        args
        ----------
        waves: Iterable[float], 1-D wavelength array for materials
        high: Iterable[complex], 1-D refractive indices of high index material
        low: Iterable[complex], 1-D refractive indices of low index material

        kwargs
        ----------
        spec_res: float, spectral resolution of the design
        max_total_thick: float, max total thickness in nanometers (default 20_000)
        max_layers: int, maximum number of ThinFilm layers (default 20)
        min_layers: int, minimum number of ThinFilm layers (default 5)
        first_lyr_min_thick: float, min thickness of first layer in nanometers (default 500)
        min_thick: float, min thickness of remaining layers in nanometers (default 10)
        max_thick: float, max thickness of layers in nanometers (default 2500)
        """

        max_layers = int(kwargs.get('max_layers', 20))
        min_layers = int(kwargs.get('min_layers', 5))
        first_lyr_min_thick = float(kwargs.get('first_lyr_min_thick', 500.0))

        if not len(high) == len(waves):
            raise ValueError("high length must match wavelengths")
        if not len(low) == len(waves):
            raise ValueError("low length must match wavelengths")
        if not len(low) == len(high):
            raise ValueError("high length must match low length")

        self.high = high
        self.low = low
        self.waves = waves

        # estimate total thickness used to scale layers
        spec_res = int(kwargs.pop('spec_res'))
        est_total_thick = self._estimated_total_thick(spec_res)
        iprint(f"[DEBUG] Estimated Thickness: {est_total_thick} nm")

        # generate a random number of layers between min_layers - max_layers
        rand_layers = min_layers + round((max_layers - min_layers) * random.uniform(0.0, 1.0))
        self.scale_factor = 2 * est_total_thick / rand_layers
        iprint(f"[DEBUG] Scale Factor: {self.scale_factor}")

        # random film stack
        rand_films = []

        # generate thin film layers
        for i in range(rand_layers):
            rand_films.append(
                ThinFilm(
                    self.waves,
                    self.high if i % 2 == 0 else self.low,
                    thick=self.scale_factor * random.uniform(0.0, 1.0),
                    ntype=1 if i % 2 == 0 else 0
                )
            )

        # pass kwargs to parent __init__()
        super().__init__(rand_films, **kwargs)  # pylint: disable=useless-parent-delegation

    def _estimated_total_thick(self, res: int) -> float:
        """
        Calculates the estimated total physical thickness of
        the film stack based on spectral resolution.

        Parameters
        ------------
        res: int, spectral resolution

        Returns
        ----------
        float, estimated total physical thickness of film stack
         in nanometers based on high and low index material.
        """

        # calculate high/low averages
        h_avg = np.mean(np.real(self.high))
        l_avg = np.mean(np.real(self.low))

        # calculate formula (2 * n1 * n2) / (n1 + n2)
        n_avg = (2 * h_avg * l_avg) / (h_avg + l_avg)

        # calculate thickness based on resolution
        return (10**7) / (4 * n_avg * int(res))