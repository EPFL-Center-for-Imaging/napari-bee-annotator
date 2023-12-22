from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import Container, Select, SpinBox, create_widget

if TYPE_CHECKING:
    import napari


class Annotator(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.offset_up = np.array((0, 10, 30, 0))
        self.offset_down = np.array((0, 10, -30, 0))
        self.offset_right = np.array((0, 10, 0, 30))
        self.offset_left = np.array((0, 10, 0, -30))

        self.currently_connected_layer = None

        self._track_layer_combo = create_widget(
            label="Tracks layer", annotation="napari.layers.Tracks"
        )
        self._track_layer_combo.changed.connect(self._reconnect_mouse_event)

        self._orientation_select = Select(
            name="Video orientation",
            choices=("Vertical", "Horizontal"),
            value="Vertical",
        )

        self._temporal_spinbox = SpinBox(
            min=0, max=1000, value=10, name="Temporal spread"
        )
        self._spatial_spinbox = SpinBox(
            min=0, max=1000, value=10, name="Spatial spread"
        )

        self.extend(
            [
                self._track_layer_combo,
                self._orientation_select,
                self._temporal_spinbox,
                self._spatial_spinbox,
            ]
        )

        # Insert default layer
        self.tracks_layer = self._viewer.add_tracks(
            data=np.zeros((1, 4)),
            properties={"Direction": np.array((0,), dtype=np.uint8)},
        )
        self.tracks_layer.tail_width = 5
        self.tracks_layer.color_by = "Direction"
        self.tracks_layer.colormap = "viridis"

    def _reconnect_mouse_event(self, event):
        if self.currently_connected_layer is not None:
            self.currently_connected_layer.mouse_drag_callbacks.pop()
        self.currently_connected_layer = self._track_layer_combo.value
        self.currently_connected_layer.mouse_drag_callbacks.append(
            self.on_click
        )

    def on_click(self, layer, event):
        if event.button == 1:
            # Left click inserts a new track
            self.insert_new_track(layer, event)
        elif event.button == 2:
            # Right click removes a layer
            self.remove_track(layer, event)

    def insert_new_track(self, layer, event):
        directions = layer.properties["Direction"]
        if len(self._orientation_select.current_choice) != 1:
            raise ValueError("You must select exactly one orientation.")
        temp_spread = self._temporal_spinbox.value
        spatial_spread = self._spatial_spinbox.value
        direction = 0 if "Shift" in event.modifiers else 1
        new_id = max(layer.data[:, 0]) + 1
        origin = np.array((new_id, *event.position))
        points = []
        for dt in range(-temp_spread, temp_spread + 1):
            t = origin[1] + dt
            t = max(t, 0)
            dx = spatial_spread * dt / temp_spread
            if direction == 1:
                dx = -dx
            if self._orientation_select.current_choice[0] == "Vertical":
                point = (new_id, t, origin[2] + dx, origin[3])
            else:
                point = (new_id, t, origin[2], origin[3] + dx)
            points.append(point)

        directions = np.concatenate(
            (directions, np.array((direction,) * len(points), dtype=np.int8))
        )

        new_track = np.stack(points)
        layer.color_by = "track_id"
        layer.data = np.concatenate((layer.data, new_track), axis=0)

        layer.properties = {
            "Direction": directions.astype(np.uint32),
            "track_id": layer.data[:, 0].astype(np.uint32),
        }
        layer.color_by = "Direction"
        layer.refresh()

    def remove_track(self, layer, event):
        position = event.position
        distances = np.linalg.norm(layer.data[:, 1:] - position, axis=1)
        track_id = layer.data[np.argmin(distances), 0]
        mask = layer.data[:, 0] != track_id
        layer.color_by = "track_id"
        directions = layer.properties["Direction"]
        layer.data = layer.data[mask]
        directions = directions[mask]
        layer.properties = {
            "Direction": directions.astype(np.uint32),
            "track_id": layer.data[:, 0].astype(np.uint32),
        }
        layer.color_by = "Direction"
        layer.refresh()
