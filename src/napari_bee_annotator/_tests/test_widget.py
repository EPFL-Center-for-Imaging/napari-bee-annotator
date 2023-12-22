from unittest.mock import Mock

import napari
import numpy as np
import pytest

from napari_bee_annotator._widget import Annotator


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
@pytest.mark.parametrize("orientation", ["Vertical", "Horizontal"])
@pytest.mark.parametrize("dt", [5, 10])
@pytest.mark.parametrize("dx", [5, 10])
def test_annotator_integration(make_napari_viewer, orientation, dt, dx):
    viewer = make_napari_viewer()
    _widget = Annotator(viewer)

    # check that a tracks layer was added
    assert len(viewer.layers) == 1
    tracks_layer = viewer.layers["Tracks"]
    assert isinstance(tracks_layer, napari.layers.Tracks)

    # because we saved our widgets as attributes of the container
    # we can set their values without having to "interact" with the viewer
    _widget._orientation_select.value = orientation
    _widget._temporal_spinbox.value = dt
    _widget._spatial_spinbox.value = dx

    # Add a track with a fake click event
    mock_event = Mock()
    mock_event.button = 1
    mock_event.position = (10, 35.4, 50.2)
    mock_event.modifiers = []
    _widget.on_click(tracks_layer, mock_event)
    # Assertions
    _validate_tracks_layer(tracks_layer, expected_n_tracks=2)
    _validate_track(
        tracks_layer,
        track_id=1,
        expected_n_points=2 * dt + 1,
        expected_direction=1,
    )

    # Add another track with a fake click event
    mock_event.position = (15, 33.8, 17.4)
    mock_event.modifiers = ["Shift"]
    _widget.on_click(tracks_layer, mock_event)
    # Assertions
    _validate_tracks_layer(tracks_layer, expected_n_tracks=3)
    _validate_track(
        tracks_layer,
        track_id=2,
        expected_n_points=2 * dt + 1,
        expected_direction=0,
    )

    # Add another track with a fake click event
    mock_event.position = (50, 70.4, 25.3)
    mock_event.modifiers = []
    _widget.on_click(tracks_layer, mock_event)
    # Assertions
    _validate_tracks_layer(tracks_layer, expected_n_tracks=4)
    _validate_track(
        tracks_layer,
        track_id=3,
        expected_n_points=2 * dt + 1,
        expected_direction=1,
    )

    # Remove a track
    mock_event.button = 2
    mock_event.position = (17, 32.8, 20.4)
    mock_event.modifiers = []
    _widget.on_click(tracks_layer, mock_event)
    # Assertions
    _validate_tracks_layer(tracks_layer, expected_n_tracks=3)
    _validate_track(
        tracks_layer,
        track_id=2,
        expected_n_points=0,
        expected_direction=None,
    )


def _validate_tracks_layer(layer, expected_n_tracks):
    # check that there are expected_n_tracks track_ids
    assert len(set(layer.data[:, 0])) == expected_n_tracks


def _validate_track(layer, track_id, expected_n_points, expected_direction):
    mask = layer.data[:, 0] == track_id
    n_points = layer.data[mask].shape[0]
    # Check the number of points
    assert n_points == expected_n_points
    # check direction
    directions = layer.properties["Direction"][mask]
    assert len(directions) == n_points
    if expected_direction is not None:
        np.testing.assert_allclose(directions, expected_direction)
