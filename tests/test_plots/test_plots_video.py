import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from mio.models.frames import NamedVideo
from mio.plots.video import VideoPlotter


class TestVideoPlotter(unittest.TestCase):

    def setUp(self):
        # Create a list of NamedFrame instances with dummy data
        self.named_frames = [
            NamedVideo(name="Video1", video=[np.random.rand(64, 64) * 255 for _ in range(50)]),
            NamedVideo(name="Video2", video=[np.random.rand(64, 64) * 255 for _ in range(50)]),
        ]

    @patch("mio.plots.video.plt.show")
    @patch("mio.plots.video.plt.subplots")
    @patch("mio.plots.video.Slider")
    @patch("mio.plots.video.Button")
    def test_show_video_with_controls(self, MockButton, MockSlider, mock_subplots, mock_show):
        # Create mock axes and figures
        mock_axes = [MagicMock() for _ in range(len(self.named_frames))]
        mock_figure = MagicMock()
        mock_subplots.return_value = (mock_figure, mock_axes)

        # Instantiate VideoPlotter and call show method
        video_plotter = VideoPlotter(self.named_frames, 0, 10)
        video_plotter.show()

        # Ensure subplots were created
        mock_subplots.assert_called_once()

        # Verify the number of axes matches the number of videos
        self.assertEqual(len(mock_axes), len(self.named_frames))

        # Ensure that sliders and buttons are initialized
        MockSlider.assert_called_once()
        MockButton.assert_called_once()

        # Check that imshow and initial frame setup are done
        for mock_ax in mock_axes:
            mock_ax.imshow.assert_called_once()

        # Verify that plt.show() is called to display the plot
        mock_show.assert_called_once()
