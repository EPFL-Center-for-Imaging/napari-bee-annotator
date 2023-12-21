"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_tracks_layer(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single tracks layer.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """

    # implement your writer logic here ...
    directions = meta["properties"]["Direction"]
    data = np.concatenate((directions[:, None], data), axis=1)
    np.savetxt(path, data, delimiter=",")

    # return path to any file(s) that were successfully written
    return [path]
