try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader
from ._widget import Annotator
from ._writer import write_single_tracks_layer

__all__ = ("Annotator", "write_single_tracks_layer", "napari_get_reader")
