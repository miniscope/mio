"""
Plot headers from :class:`.SDCard`
"""

from collections import deque
from itertools import count
from time import time
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from mio.models.stream import StreamBufferHeader

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def buffer_count(headers: pd.DataFrame, ax: "plt.Axes") -> "plt.Axes":
    """
    Plot number of buffers by time
    """
    cols = ("write_buffer_count", "dropped_buffer_count", "buffer_count")
    labels = ("Write Buffer", "Dropped Buffer", "Total Buffer")
    for col, label in zip(cols, labels):
        ax.plot(headers[col], label=label)
    ax.legend()
    ax.set_xlabel("Buffer index")
    ax.set_xlabel("Buffer count")
    return ax


def dropped_buffers(headers: pd.DataFrame, ax: "plt.Axes") -> "plt.Axes":
    """
    Plot number of buffers by time
    """
    ax.plot(headers["dropped_buffer_count"], label="Dropped buffers")
    ax.legend()
    ax.set_xlabel("Buffer index")
    return ax


def timestamps(headers: pd.DataFrame, ax: "plt.Axes") -> "plt.Axes":
    """
    Plot frame number against time
    """
    frames = headers["frame_num"].max() - headers["frame_num"].min()
    seconds = headers["timestamp"].max() - headers["timestamp"].min()
    fps = frames / seconds

    ax.plot(headers["timestamp"], headers["frame_num"], label=f"Record: {fps:.2f} FPS")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frame Count")
    return ax


def battery_voltage(headers: pd.DataFrame, ax: "plt.Axes") -> "plt.Axes":
    """
    Plot battery voltage against time
    """
    ax.plot(headers["timestamp"], headers["battery_voltage"], label="Battery voltage")
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Battery Voltage")
    return ax


def plot_headers(
    headers: pd.DataFrame, size: Optional[Tuple[int, int]] = None
) -> ("plt.Figure", "plt.Axes"):
    """
    Plot the headers (generated from :meth:`.Frame.to_df` )

    Mimicking the plot in https://github.com/Aharoni-Lab/Miniscope-v4-Wire-Free/blob/2fd86cc85b810b2ecc6f71c8ee2dffdb838badcf/Miniscope-v4-Wire-Free-Python%20DAQ%20Interface/Load%20raw%20data%20from%20SD%20card%20and%20write%20video%20-%20WireFree%20V4%20Miniscope.ipynb
    For more generic plotting, see :meth:`pandas.DataFrame.plot`

    Arguments:
        headers (:class:`pandas.DataFrame`): headers to plot
        size (tuple[int, int]): Manually override plot ``(width, height)`` . Arbitrary units
    """
    subplots = 4 if "battery_voltage" in headers.columns else 3

    fig, ax = plt.subplots(1, subplots)

    # Successful Buffers
    ax[0] = buffer_count(headers, ax[0])

    # Dropped buffers
    ax[1] = dropped_buffers(headers, ax[1])

    # fps/timestamps
    ax[2] = timestamps(headers, ax[2])

    if "battery_voltage" in headers.columns:
        ax[3] = battery_voltage(headers, ax[3])

    if size is None:
        size = ((subplots * 3) + 1, 3)

    fig.set_figwidth(size[0])
    fig.set_figheight(size[1])

    return fig, ax


class StreamPlotter:
    """
    Plot headers from StreamDaq.

    .. note::

        Eventually this should get generalized into a plotter object that
        can take an arbitrary set of keys and values, but for now is
        somewhat specific to :class:`.StreamDaq` , at least in the type hints.

    """

    def __init__(
        self, header_keys: List[str], history_length: int = 100, update_ms: int = 1000
    ) -> None:
        """
        Constructor of StreamPlotter.

        Parameters:
            header_keys (list[str]): List of header keys to plot or a single header key as a string
            history_length (int): Number of headers to plot
            update_ms (int): milliseconds between each plot update
        """
        global plt
        if plt is None:
            raise ModuleNotFoundError(
                "matplotlib is not a required dependency of mio, to use it, "
                "install it manually or install mio with `pip install mio[plot]`"
            )

        # If a single string is provided, convert it to a list with one element
        if isinstance(header_keys, str):
            header_keys = [header_keys]

        self.header_keys = header_keys
        self.history_length = history_length
        self.update_ms = update_ms
        self.fig, self.axes, self.lines = self._init_plot()
        self.data = {key: deque(maxlen=self.history_length) for key in self.header_keys}
        self.index = deque(maxlen=self.history_length)
        self._last_update = time()
        self._index_counter = count()

    def _init_plot(
        self,
    ) -> tuple["plt.Figure", dict[str, "plt.Axes"], dict[str, "plt.Line2D"]]:
        # initialize matplotlib
        plt.ion()
        fig: plt.Figure
        axes: np.ndarray[Any, np.dtype[plt.Axes]]
        fig, axes = plt.subplots(len(self.header_keys), 1, figsize=(6, len(self.header_keys) * 2))
        axes = np.array(axes).reshape(-1)  # Ensure axes is an array

        # Initialize line objects
        axes_dict = {}
        lines = {}
        for i, header_key in enumerate(self.header_keys):
            ax: plt.Axes = axes[i]
            metadata_trunc = np.zeros((0, 2))
            x_data = metadata_trunc[:, 0]
            y_data = metadata_trunc[:, 1]
            (line,) = ax.plot(x_data, y_data)
            lines[header_key] = line
            axes_dict[header_key] = ax

            if i == len(self.header_keys) - 1:
                ax.set_xlabel("index")

            ax.set_ylabel(header_key)
        return fig, axes_dict, lines

    def update(
        self,
        header: StreamBufferHeader,
    ) -> None:
        """
        Update the plot with the latest data.

        Parameters:
            header: StreamBufferHeader to update with
        """
        this_update = time()

        # update the index
        self.index.append(next(self._index_counter))
        for key in self.header_keys:
            self.data[key].append(getattr(header, key))

        if this_update - self._last_update > (self.update_ms / 1000):
            self._last_update = this_update
            for key in self.header_keys:
                self.lines[key].set_ydata(self.data[key])
                self.lines[key].set_xdata(self.index)
                if len(self.index) > 1:
                    self.axes[key].set_xlim(self.index[0], self.index[-1])
                    self.axes[key].set_ylim(np.min(self.data[key]), np.max(self.data[key]))
            plt.draw()
            plt.pause(0.01)

    def close_plot(self) -> None:
        """
        Close the plot and perform any necessary cleanup.
        """
        plt.ioff()  # Turn off interactive mode
        plt.close(self.fig)
