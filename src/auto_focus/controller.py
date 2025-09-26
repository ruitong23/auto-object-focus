"""Controller module for automatically focusing the cursor on detected objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "ultralytics is required to use AutoFocusController. Install the optional dependencies first."
    ) from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "opencv-python is required to use AutoFocusController. Install the optional dependencies first."
    ) from exc

try:  # pragma: no cover - optional runtime dependency
    import mss  # type: ignore
except ImportError:  # pragma: no cover - handled by falling back to pyautogui
    mss = None

try:
    import pyautogui
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "pyautogui is required to use AutoFocusController. Install the optional dependencies first."
    ) from exc


@dataclass
class BoundingBox:
    """Normalized bounding box and confidence information."""

    center_x: float
    center_y: float
    confidence: float


FrameProvider = Callable[[], Optional[np.ndarray]]


class AutoFocusController:
    """Automatically keep the mouse cursor centered on a detected object."""

    def __init__(
        self,
        target_class: Optional[Union[str, int]] = None,
        *,
        confidence_threshold: float = 0.5,
        smoothing_factor: float = 0.2,
        tracking_speed: float = 0.2,
        distance_ratio: float = 2.0,
        model_path: str = "yolov8n.pt",
        cursor_controller: Optional[object] = None,
        frame_provider: Optional[FrameProvider] = None,
    ) -> None:
        """Create a new controller instance.

        Parameters
        ----------
        target_class:
            Name or numeric identifier of the class to track. When ``None`` the first
            detected class with sufficient confidence will be used.
        confidence_threshold:
            Minimum detection confidence for the class to be considered.
        smoothing_factor:
            Exponential smoothing factor between 0 and 1 used to smooth cursor motion.
        model_path:
            Path to the YOLO model weights.
        cursor_controller:
            Optional module-like object implementing the subset of the :mod:`pyautogui`
            API used by the controller. Useful for testing.
        frame_provider:
            Optional callable returning the next video frame as a NumPy array. When not
            supplied a screenshot-based provider is used to track objects displayed on
            the screen.
        """

        if not 0 < smoothing_factor <= 1:
            raise ValueError("smoothing_factor must be between 0 (exclusive) and 1 (inclusive)")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")

        if not 0 < tracking_speed <= 1:
            raise ValueError("tracking_speed must be between 0 (exclusive) and 1 (inclusive)")
        if distance_ratio < 0:
            raise ValueError("distance_ratio must be non-negative")

        self.model = YOLO(model_path)
        self.model_target = target_class
        self.confidence_threshold = confidence_threshold
        self.smoothing_factor = smoothing_factor
        self.tracking_speed = tracking_speed
        self.distance_ratio = distance_ratio
        self.cursor_controller = cursor_controller if cursor_controller is not None else pyautogui

        self.screen_width, self.screen_height = self._normalize_screen_size(self.cursor_controller.size())
        self._smoothed_offset = np.zeros(2, dtype=float)
        self._running = False
        self._closed = False
        self._frame_provider = frame_provider if frame_provider is not None else self._create_frame_provider()
        self._frame_provider_close = getattr(self._frame_provider, "close", None)

    def run(self) -> None:
        """Start capturing screen frames and updating the cursor position."""

        self._running = True
        try:
            while self._running:
                try:
                    frame = self._frame_provider()
                except StopIteration:
                    break

                if frame is None:
                    break

                box = self._select_target_box(frame)
                if box is None:
                    continue

                self._update_cursor(frame, box)
        finally:
            self.close()

    def stop(self) -> None:
        """Signal the controller loop to stop."""

        self._running = False

    def close(self) -> None:
        """Release associated resources."""

        if self._closed:
            return

        self.stop()

        closer = self._frame_provider_close
        if callable(closer):
            closer()
            self._frame_provider_close = None
        if hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()

        self._closed = True

    def update_motion_parameters(
        self,
        *,
        tracking_speed: Optional[float] = None,
        distance_ratio: Optional[float] = None,
    ) -> None:
        """Update motion-related parameters while the controller is running."""

        if tracking_speed is not None:
            if not 0 < tracking_speed <= 1:
                raise ValueError("tracking_speed must be between 0 (exclusive) and 1 (inclusive)")
            self.tracking_speed = float(tracking_speed)

        if distance_ratio is not None:
            if distance_ratio < 0:
                raise ValueError("distance_ratio must be non-negative")
            self.distance_ratio = float(distance_ratio)

    def _select_target_box(self, frame: np.ndarray) -> Optional[BoundingBox]:
        """Run the YOLO model on the frame and select the best matching bounding box."""

        results = self.model(frame)
        if not results:
            return None

        target_id = self._resolve_target_id()
        best_box: Optional[BoundingBox] = None

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            class_ids = self._to_numpy(getattr(boxes, "cls", []))
            confidences = self._to_numpy(getattr(boxes, "conf", []))
            coords = self._to_numpy(getattr(boxes, "xyxy", []))
            if len(coords) == 0:
                continue

            names = self._get_names_mapping(result)

            for coord, confidence, class_id in zip(coords, confidences, class_ids):
                if confidence < self.confidence_threshold:
                    continue

                if target_id is not None and int(class_id) != target_id:
                    continue

                if target_id is None and self.model_target is not None:
                    class_name = names.get(int(class_id), str(class_id)) if names else str(class_id)
                    if str(self.model_target).lower() != str(class_name).lower():
                        continue

                x1, y1, x2, y2 = coord
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                candidate = BoundingBox(center_x=center_x, center_y=center_y, confidence=float(confidence))

                if best_box is None or candidate.confidence > best_box.confidence:
                    best_box = candidate

        return best_box

    def _update_cursor(self, frame: np.ndarray, box: BoundingBox) -> None:
        """Smooth the detected offset and move the cursor accordingly."""

        frame_height, frame_width = frame.shape[:2]
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0])
        detection_center = np.array([box.center_x, box.center_y])
        offset_pixels = detection_center - frame_center
        normalized_offset = offset_pixels / np.array([frame_width, frame_height])

        self._smoothed_offset = (1 - self.smoothing_factor) * self._smoothed_offset + self.smoothing_factor * normalized_offset

        screen_center = np.array([self.screen_width / 2.0, self.screen_height / 2.0])
        desired_position = screen_center + self._smoothed_offset * np.array([self.screen_width, self.screen_height])
        current_position = self._get_cursor_position()
        movement_vector = desired_position - current_position

        if np.allclose(movement_vector, 0):
            return

        distance_norm = float(np.linalg.norm(self._smoothed_offset))
        effective_speed = min(1.0, self.tracking_speed + self.distance_ratio * distance_norm)
        target_position = current_position + movement_vector * effective_speed
        clamped_position = np.clip(target_position, [0, 0], [self.screen_width - 1, self.screen_height - 1])

        self.cursor_controller.moveTo(int(clamped_position[0]), int(clamped_position[1]))

    def _resolve_target_id(self) -> Optional[int]:
        """Resolve the numeric class identifier for the configured target."""

        if isinstance(self.model_target, int):
            return int(self.model_target)

        if isinstance(self.model_target, str):
            names = self._get_model_names()
            if names:
                for class_id, class_name in names.items():
                    if str(class_name).lower() == self.model_target.lower():
                        return int(class_id)

        return None

    def _get_cursor_position(self) -> np.ndarray:
        """Return the current cursor position as a NumPy array."""

        position = self.cursor_controller.position()
        if hasattr(position, "x") and hasattr(position, "y"):
            x, y = float(position.x), float(position.y)
        else:
            x, y = position

        return np.array([x, y], dtype=float)

    def _get_model_names(self) -> Optional[Dict[int, str]]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return None

    @staticmethod
    def _get_names_mapping(result: object) -> Optional[Dict[int, str]]:
        names = getattr(result, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return None

    @staticmethod
    def _normalize_screen_size(size: Union[Tuple[int, int], object]) -> Tuple[int, int]:
        """Normalize the screen size returned by :mod:`pyautogui`."""

        if hasattr(size, "width") and hasattr(size, "height"):
            width = getattr(size, "width")
            height = getattr(size, "height")
        else:
            width, height = size

        return int(width), int(height)

    @staticmethod
    def _to_numpy(values: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
        """Convert framework-specific tensors to NumPy arrays."""

        if values is None:
            return np.empty((0,), dtype=float)

        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()

        return np.asarray(values, dtype=float)

    def _create_frame_provider(self) -> FrameProvider:
        """Return the default frame provider capturing the primary screen."""

        return _ScreenFrameProvider()


class _ScreenFrameProvider:
    """Capture frames from the primary monitor for object tracking."""

    def __init__(self) -> None:
        self._sct = None
        self._monitor = None
        if mss is not None:  # pragma: no branch - simple runtime selection
            self._sct = mss.mss()
            # ``monitors[0]`` corresponds to the entire virtual screen.
            monitor = self._sct.monitors[0]
            self._monitor = {
                "left": monitor.get("left", 0),
                "top": monitor.get("top", 0),
                "width": monitor.get("width", 0),
                "height": monitor.get("height", 0),
            }

    def __call__(self) -> Optional[np.ndarray]:
        if self._sct is not None and self._monitor is not None:
            shot = self._sct.grab(self._monitor)
            frame = np.array(shot)
            # mss returns BGRA frames.
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Fallback to pyautogui's screenshot functionality when mss is unavailable.
        image = pyautogui.screenshot()
        frame = np.array(image)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
