"""
Plotting functions for video streams and frames.
"""

import numpy as np

from mio import init_logger
from mio.models.frames import NamedFrame, NamedVideo

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.backend_bases import KeyEvent
    from matplotlib.widgets import Button, Slider
except ImportError:
    plt = None
    animation = None
    Button = None
    Slider = None
    KeyEvent = None

logger = init_logger("videoplot")


class VideoPlotter:
    """
    Class to display video streams and static images.

    Parameters
    ----------
    videos : list
        List of NamedFrame or NamedVideo instances.
    start_frame : int
        Start frame index.
    end_frame : int
        End frame index.
    fps : int, optional
        Frames per second, by default 20.
    """

    def __init__(
        self,
        videos: list[NamedFrame | NamedVideo],
        start_frame: int,
        end_frame: int,
        fps: int = 20,
    ):
        self.fps = fps
        self.videos = videos
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._titles = None
        self.video_frames = self._init_video_frames()
        self.num_frames = max(len(stream) for stream in self.video_frames)
        self.playing = False
        self.fig, self.axes = plt.subplots(1, len(self.video_frames), figsize=(20, 5))
        self.frame_displays = self._init_video_displays()
        self.slider, self.button = self._init_controls()

    def _init_video_frames(self) -> list[list]:
        video_frames = [None] * len(self.videos)
        for i, video in enumerate(self.videos):
            if isinstance(video, NamedFrame):
                video_frames[i] = [video.frame]
            elif isinstance(video, NamedVideo):
                video_frames[i] = video.video[self.start_frame : self.end_frame]
        return video_frames

    @property
    def titles(self) -> list[str]:
        """
        Get the titles of the videos from the NamedFrame or NamedVideo instances.

        Returns
        -------
        titles : list
            List of video titles.
        """
        if self._titles is None:
            self._titles = [video.name for video in self.videos]
        return self._titles

    def _init_video_displays(self) -> list:
        """
        Initialize the video displays.
        """
        frame_displays = []
        for idx, ax in enumerate(self.axes):
            initial_frame = self.video_frames[idx][0]
            frame_display = ax.imshow(
                initial_frame, cmap="gray", vmin=0, vmax=np.iinfo(np.uint8).max
            )
            frame_displays.append(frame_display)
            if self.titles:
                ax.set_title(self.videos[idx].name)
            ax.axis("off")
        return frame_displays

    def _init_controls(self) -> tuple:
        """
        Initialize the slider and play/pause button.

        Returns
        -------
        slider : matplotlib.widgets.Slider
            Slider to select the frame index.
        button : matplotlib.widgets.Button
            Button to toggle play/pause.
        """
        ax_slider = plt.axes([0.1, 0.1, 0.65, 0.05], facecolor="lightgoldenrodyellow")
        slider = Slider(
            ax=ax_slider, label="Frame", valmin=0, valmax=self.num_frames - 1, valinit=0, valstep=1
        )
        slider.on_changed(self.on_slider_change)

        ax_button = plt.axes([0.8, 0.1, 0.1, 0.05])
        button = Button(ax_button, "Play/Pause")
        button.on_clicked(self.toggle_play)

        return slider, button

    def toggle_play(self, event: KeyEvent) -> None:  # type: ignore
        """
        Toggle the play/pause state of the video.
        """
        self.playing = not self.playing

    def on_slider_change(self, val: float) -> None:
        """
        Update the frame display based on the slider value.

        Parameters
        ----------
        val : float
            Slider value.
        """
        index = int(self.slider.val)
        self.update_frame(index)

    def update_frame(self, index: int) -> None:
        """
        Update the frame display based on the given index.

        Parameters
        ----------
        index : int
            Index of the frame to display.
        """
        for idx, frame_display in enumerate(self.frame_displays):
            if index < len(self.video_frames[idx]):
                frame = self.video_frames[idx][index]
            else:
                frame = self.video_frames[idx][-1]
            frame_display.set_data(frame)
        self.fig.canvas.draw_idle()

    def animate(self, i: int) -> None:
        """
        Update the frame display for the animation.

        Parameters
        ----------
        i : int
            Frame index.
        """
        if self.playing:
            current_frame = int(self.slider.val)
            next_frame = (current_frame + 1) % self.num_frames
            self.slider.set_val(next_frame)

    def show(self) -> None:
        """
        Display the video stream with controls.
        """
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.num_frames, interval=1000 // self.fps, blit=False
        )
        plt.show()
