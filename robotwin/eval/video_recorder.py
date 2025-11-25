"""
Multi-camera video recorder for RoboTwin evaluation.

Records frames from multiple cameras and composites them into a single video
with a 2x2 grid layout.
"""

import os
import subprocess
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from datetime import datetime


class MultiCameraRecorder:
    """
    Records multi-camera video during RoboTwin evaluation.

    Creates a 2x2 grid video with:
    - Top-left: Head camera
    - Top-right: Observer (3rd person) camera
    - Bottom-left: Left wrist camera
    - Bottom-right: Right wrist camera

    Also adds text overlay with task name, step count, and success status.
    """

    def __init__(
        self,
        output_dir: str = "eval_videos",
        fps: int = 10,
        camera_size: tuple = (320, 240),
    ):
        """
        Initialize the video recorder.

        Args:
            output_dir: Directory to save videos
            fps: Frames per second for output video
            camera_size: Size of each camera view (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.camera_width, self.camera_height = camera_size

        # Output frame size (2x2 grid)
        self.output_width = self.camera_width * 2
        self.output_height = self.camera_height * 2

        # State
        self.frames = []
        self.episode_id = None
        self.task_name = None
        self.ffmpeg_process = None
        self.video_path = None

    def start_episode(self, episode_id: int, task_name: str):
        """
        Start recording a new episode.

        Args:
            episode_id: Episode number
            task_name: Name of the task being evaluated
        """
        self.episode_id = episode_id
        self.task_name = task_name
        self.frames = []

        # Create video path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = self.output_dir / f"{task_name}_ep{episode_id}_{timestamp}.mp4"

        # Start ffmpeg process
        self.ffmpeg_process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{self.output_width}x{self.output_height}",
                "-framerate", str(self.fps),
                "-i", "-",
                "-pix_fmt", "yuv420p",
                "-vcodec", "libx264",
                "-crf", "23",
                str(self.video_path),
            ],
            stdin=subprocess.PIPE,
        )

    def add_frame(
        self,
        head_rgb: np.ndarray,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        observer_rgb: Optional[np.ndarray] = None,
        step: int = 0,
        success: Optional[bool] = None,
    ):
        """
        Add a frame to the video.

        Args:
            head_rgb: Head camera RGB image (H, W, 3)
            left_rgb: Left wrist camera RGB image (H, W, 3)
            right_rgb: Right wrist camera RGB image (H, W, 3)
            observer_rgb: Observer camera RGB image (optional)
            step: Current step number for overlay
            success: Success status for overlay (None = in progress)
        """
        if self.ffmpeg_process is None:
            return

        # Resize images to camera_size
        head = self._resize(head_rgb)
        left = self._resize(left_rgb)
        right = self._resize(right_rgb)

        if observer_rgb is not None:
            observer = self._resize(observer_rgb)
        else:
            # Create placeholder if no observer camera
            observer = np.zeros(
                (self.camera_height, self.camera_width, 3),
                dtype=np.uint8
            )
            cv2.putText(
                observer, "No Observer",
                (self.camera_width // 4, self.camera_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2
            )

        # Add labels to each quadrant
        head = self._add_label(head, "Head Camera", position="top")
        observer = self._add_label(observer, "Observer", position="top")
        left = self._add_label(left, "Left Wrist", position="top")
        right = self._add_label(right, "Right Wrist", position="top")

        # Compose 2x2 grid
        top_row = np.concatenate([head, observer], axis=1)
        bottom_row = np.concatenate([left, right], axis=1)
        frame = np.concatenate([top_row, bottom_row], axis=0)

        # Add step counter and status overlay
        frame = self._add_overlay(frame, step, success)

        # Write frame to ffmpeg
        self.ffmpeg_process.stdin.write(frame.tobytes())

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to camera_size."""
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        return cv2.resize(
            img,
            (self.camera_width, self.camera_height),
            interpolation=cv2.INTER_LINEAR
        )

    def _add_label(
        self,
        img: np.ndarray,
        label: str,
        position: str = "top"
    ) -> np.ndarray:
        """Add a text label to the image."""
        img = img.copy()

        # Semi-transparent background for text
        overlay = img.copy()
        if position == "top":
            cv2.rectangle(overlay, (0, 0), (self.camera_width, 25), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.putText(
                img, label,
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        return img

    def _add_overlay(
        self,
        frame: np.ndarray,
        step: int,
        success: Optional[bool]
    ) -> np.ndarray:
        """Add step counter and status overlay to frame."""
        frame = frame.copy()

        # Task name at top center
        task_text = f"Task: {self.task_name}"
        text_size = cv2.getTextSize(task_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (self.output_width - text_size[0]) // 2
        cv2.putText(
            frame, task_text,
            (text_x, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Step counter at bottom left
        step_text = f"Step: {step}"
        cv2.putText(
            frame, step_text,
            (10, self.output_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # Success/fail indicator at bottom right
        if success is not None:
            if success:
                status_text = "SUCCESS"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "FAILED"
                status_color = (0, 0, 255)  # Red

            text_size = cv2.getTextSize(
                status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )[0]
            cv2.putText(
                frame, status_text,
                (self.output_width - text_size[0] - 10, self.output_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
            )

        return frame

    def end_episode(self, success: bool):
        """
        End recording for the current episode.

        Args:
            success: Whether the episode was successful
        """
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None

            # Rename file to include success status
            if self.video_path and self.video_path.exists():
                status = "success" if success else "fail"
                new_name = self.video_path.stem + f"_{status}" + self.video_path.suffix
                new_path = self.video_path.parent / new_name
                self.video_path.rename(new_path)
                self.video_path = new_path

                print(f"Video saved: {self.video_path}")

    def close(self):
        """Clean up resources."""
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None


def get_observer_rgb(env) -> Optional[np.ndarray]:
    """
    Get RGB from observer camera if available.

    Args:
        env: RoboTwin environment

    Returns:
        Observer camera RGB or None
    """
    try:
        # Observer camera needs to take a picture first
        if hasattr(env, 'cameras') and hasattr(env.cameras, 'observer_camera'):
            env.cameras.observer_camera.take_picture()
            rgba = env.cameras.observer_camera.get_picture("Color")
            rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
            return rgb
    except Exception as e:
        pass

    return None
