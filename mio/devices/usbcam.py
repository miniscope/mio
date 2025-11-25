"""
USB Camera device implementation.
"""

from typing import Dict

import cv2


class ELPUVCCamera:
    """USB Camera device for listing cameras."""

    @staticmethod
    def list_cameras() -> Dict[int, Dict[str, str]]:
        """
        List available cameras with details.

        Returns:
            Dictionary mapping camera index to camera info
        """
        available_cameras: Dict[int, Dict[str, str]] = {}

        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    resolution = f"{frame.shape[1]}x{frame.shape[0]}"
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    available_cameras[i] = {
                        "resolution": resolution,
                        "fps": str(fps),
                    }
                cap.release()
            else:
                # Stop checking after first failure (no more cameras)
                break

        return available_cameras
