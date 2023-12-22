import random
from unittest.mock import Mock

import numpy as np
import pandas as pd

from napari_bee_annotator import (
    Annotator,
    napari_get_reader,
    write_single_tracks_layer,
)


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    # write some fake data
    test_file = str(tmp_path / "tracks.csv")

    n_points = 120
    n_tracks = 8
    expected_track_ids = np.arange(n_tracks).repeat(n_points / n_tracks)
    expected_directions = np.zeros(n_points)
    expected_directions[80:] = 1
    expected_positions = np.random.rand(n_points, 3)

    data = np.concatenate(
        (
            expected_directions[:, None],
            expected_track_ids[:, None],
            expected_positions,
        ),
        axis=1,
    )

    np.savetxt(test_file, data, delimiter=",")

    # read it back in
    reader = napari_get_reader(test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's loaded as a tracks layer
    assert layer_data_tuple[2] == "tracks"

    # make sure data is the same as it started
    positions = layer_data_tuple[0][:, 1:]
    np.testing.assert_allclose(expected_positions, positions)

    track_ids = layer_data_tuple[0][:, 0]
    np.testing.assert_allclose(expected_track_ids, track_ids)

    directions = layer_data_tuple[1]["properties"]["Direction"]
    np.testing.assert_allclose(expected_directions, directions)


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None


def test_writer(tmp_path):
    path = str(tmp_path / "test.csv")
    n_points = 30
    data = np.random.rand(n_points, 4)
    directions = np.zeros(n_points)
    meta = {"properties": {"Direction": directions}}
    path_list = write_single_tracks_layer(path, data, meta)

    assert isinstance(path_list, list)
    assert len(path_list) == 1

    saved_array = np.genfromtxt(path, delimiter=",")
    np.testing.assert_allclose(saved_array[:, 0], directions)
    np.testing.assert_allclose(saved_array[:, 1:], data)


def test_io_integration(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()
    _widget = Annotator(viewer)

    original_tracks_layer = viewer.layers["Tracks"]

    n_tracks = 10
    for _ in range(n_tracks):
        mock_event = Mock()
        mock_event.position = np.random.rand(3)
        mock_event.modifiers = random.choice([[], ["Shift"]])
        _widget.insert_new_track(original_tracks_layer, mock_event)

    test_file = str(tmp_path / "test.csv")
    path_list = write_single_tracks_layer(
        test_file,
        original_tracks_layer.data,
        original_tracks_layer._get_state(),
    )

    viewer._add_layers_with_plugins(
        path_list, stack=False, plugin="napari-bee-annotator"
    )
    new_layer = viewer.layers["test"]

    np.testing.assert_allclose(original_tracks_layer.data, new_layer.data)
    original_state = original_tracks_layer._get_state()
    new_state = new_layer._get_state()
    for key, original_value in original_state.items():
        if key == "name":
            continue
        new_value = new_state[key]
        if isinstance(original_value, np.ndarray):
            np.testing.assert_allclose(new_value, original_value)
        elif isinstance(original_value, dict):
            for k, v in original_value.items():
                if isinstance(v, np.ndarray):
                    np.testing.assert_allclose(new_value[k], v)
                else:
                    assert v == new_value[k]
        elif isinstance(original_value, pd.DataFrame):
            pd.testing.assert_frame_equal(original_value, new_value)
        else:
            assert original_value == new_value
