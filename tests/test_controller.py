from __future__ import annotations

import importlib
import sys
import types
from typing import List

import pytest

np = pytest.importorskip("numpy")


def install_stubs(monkeypatch: pytest.MonkeyPatch):
    """Install stub modules for external dependencies."""

    class DummyBoxes:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = np.array(xyxy, dtype=float)
            self.conf = np.array(conf, dtype=float)
            self.cls = np.array(cls, dtype=float)

    class DummyResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.names = {0: "person", 1: "dog"}

    class DummyYOLO:
        result_queue: List[DummyResult] = []

        def __init__(self, model_path: str):
            self.model_path = model_path
            self.names = {0: "person", 1: "dog"}

        def __call__(self, frame):
            if not DummyYOLO.result_queue:
                return []
            result = DummyYOLO.result_queue.pop(0)
            result.names = self.names
            return [result]

    class DummyCursor:
        def __init__(self):
            self.movements = []
            self._position = np.array([960.0, 540.0])

        def size(self):
            return (1920, 1080)

        def moveTo(self, x, y):
            self.movements.append((x, y))
            self._position = np.array([float(x), float(y)])

        def moveRel(self, x, y):
            self.movements.append(("rel", float(x), float(y)))

        def position(self):
            return tuple(self._position)

    dummy_ultralytics = types.ModuleType("ultralytics")
    dummy_ultralytics.YOLO = DummyYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", dummy_ultralytics)

    dummy_cv2 = types.ModuleType("cv2")
    dummy_cv2.COLOR_BGRA2BGR = 0
    dummy_cv2.COLOR_RGB2BGR = 1

    def _cvt_color(frame, code):
        return frame

    dummy_cv2.cvtColor = _cvt_color
    dummy_cv2.destroyAllWindows = lambda: None
    monkeypatch.setitem(sys.modules, "cv2", dummy_cv2)

    dummy_cursor_module = types.ModuleType("pyautogui")
    cursor = DummyCursor()
    dummy_cursor_module.size = cursor.size
    dummy_cursor_module.moveTo = cursor.moveTo
    dummy_cursor_module.moveRel = cursor.moveRel
    dummy_cursor_module.position = cursor.position
    dummy_cursor_module.FAILSAFE = False
    monkeypatch.setitem(sys.modules, "pyautogui", dummy_cursor_module)

    class DummyDirectInput:
        def __init__(self):
            self.movements = []

        def moveRel(self, x, y):
            self.movements.append((x, y))

    direct = DummyDirectInput()
    dummy_direct_module = types.ModuleType("pydirectinput")
    dummy_direct_module.moveRel = direct.moveRel
    monkeypatch.setitem(sys.modules, "pydirectinput", dummy_direct_module)

    sys.modules.pop("auto_focus.controller", None)

    return DummyYOLO, cursor, direct


class DummyFrameProvider:
    def __init__(self, frames: List[np.ndarray]):
        self._frames = frames
        self._index = 0
        self.closed = False

    def __call__(self):
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def close(self):
        self.closed = True


def test_controller_moves_cursor_with_smoothing(monkeypatch: pytest.MonkeyPatch):
    DummyYOLO, cursor, _direct = install_stubs(monkeypatch)

    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    provider = DummyFrameProvider(frames)

    controller_module = importlib.import_module("auto_focus.controller")
    AutoFocusController = controller_module.AutoFocusController

    controller_module.YOLO.result_queue = [
        types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=np.array([[300, 200, 340, 280]], dtype=float),
                conf=np.array([0.9], dtype=float),
                cls=np.array([0], dtype=float),
            ),
            names={0: "person"},
        ),
        types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=np.array([[100, 200, 140, 280]], dtype=float),
                conf=np.array([0.8], dtype=float),
                cls=np.array([0], dtype=float),
            ),
            names={0: "person"},
        ),
        types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=np.empty((0, 4)),
                conf=np.empty((0,)),
                cls=np.empty((0,)),
            ),
            names={0: "person"},
        ),
    ]

    controller = AutoFocusController(
        target_class="person",
        confidence_threshold=0.5,
        smoothing_factor=0.5,
        frame_provider=provider,
        cursor_controller=cursor,
        mode="2d",
    )

    controller.run()

    assert provider.closed is True
    assert len(cursor.movements) == 1

    # Cursor should move left relative to its current position to recenter the detection.
    kind, dx, dy = cursor.movements[0]
    assert kind == "rel"
    assert pytest.approx(dx, rel=1e-3) == -153.75
    assert pytest.approx(dy, abs=1e-6) == 0.0


def test_update_motion_parameters(monkeypatch: pytest.MonkeyPatch):
    DummyYOLO, cursor, _direct = install_stubs(monkeypatch)

    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    provider = DummyFrameProvider(frames)

    controller_module = importlib.import_module("auto_focus.controller")
    AutoFocusController = controller_module.AutoFocusController

    controller_module.YOLO.result_queue = [
        types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=np.array([[300, 200, 340, 280]], dtype=float),
                conf=np.array([0.9], dtype=float),
                cls=np.array([0], dtype=float),
            ),
            names={0: "person"},
        )
    ]

    controller = AutoFocusController(
        target_class="person",
        confidence_threshold=0.5,
        smoothing_factor=0.5,
        tracking_speed=0.3,
        distance_ratio=1.0,
        frame_provider=provider,
        cursor_controller=cursor,
        mode="2d",
    )

    controller.update_motion_parameters(tracking_speed=0.5, distance_ratio=0.8)
    assert controller.tracking_speed == pytest.approx(0.5)
    assert controller.distance_ratio == pytest.approx(0.8)

    with pytest.raises(ValueError):
        controller.update_motion_parameters(tracking_speed=0)

    with pytest.raises(ValueError):
        controller.update_motion_parameters(distance_ratio=-1)


def test_controller_3d_mode_recenters_pointer(monkeypatch: pytest.MonkeyPatch):
    DummyYOLO, cursor, direct = install_stubs(monkeypatch)

    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    provider = DummyFrameProvider(frames)

    controller_module = importlib.import_module("auto_focus.controller")
    AutoFocusController = controller_module.AutoFocusController

    controller_module.YOLO.result_queue = [
        types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=np.array([[400, 240, 440, 280]], dtype=float),
                conf=np.array([0.95], dtype=float),
                cls=np.array([0], dtype=float),
            ),
            names={0: "person"},
        )
    ]

    controller = AutoFocusController(
        target_class="person",
        confidence_threshold=0.5,
        smoothing_factor=0.5,
        tracking_speed=0.2,
        distance_ratio=1.0,
        frame_provider=provider,
        mode="3d",
    )

    assert isinstance(controller.cursor_controller, controller_module._RelativeInputAdapter)

    controller.run()

    # Expect the relative delta to be sent through the direct-input bridge while the
    # visible cursor is re-centred using the base controller.
    assert len(direct.movements) == 1
    dx, dy = direct.movements[0]
    assert dx != 0 or dy != 0

    assert len(cursor.movements) == 1
    x, y = cursor.movements[0]
    assert pytest.approx(x, abs=1e-6) == 960.0
    assert pytest.approx(y, abs=1e-6) == 540.0


def test_controller_3d_mode_falls_back_without_direct_input(monkeypatch: pytest.MonkeyPatch):
    DummyYOLO, cursor, direct = install_stubs(monkeypatch)

    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    provider = DummyFrameProvider(frames)

    controller_module = importlib.import_module("auto_focus.controller")
    AutoFocusController = controller_module.AutoFocusController

    controller_module.YOLO.result_queue = [
        types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=np.array([[400, 240, 440, 280]], dtype=float),
                conf=np.array([0.95], dtype=float),
                cls=np.array([0], dtype=float),
            ),
            names={0: "person"},
        )
    ]

    # Simulate the optional dependency being unavailable after import.
    monkeypatch.setattr(controller_module, "pydirectinput", None, raising=False)

    controller = AutoFocusController(
        target_class="person",
        confidence_threshold=0.5,
        smoothing_factor=0.5,
        tracking_speed=0.2,
        distance_ratio=1.0,
        frame_provider=provider,
        mode="3d",
    )

    assert controller_module.pydirectinput is None
    assert not isinstance(controller.cursor_controller, controller_module._RelativeInputAdapter)

    controller.run()

    # With the direct-input bridge disabled the fallback controller should receive the
    # relative delta, mirroring the legacy behaviour.
    assert any(move[0] == "rel" for move in cursor.movements)
    assert len(direct.movements) == 0


def test_3d_mode_uses_initial_cursor_position_for_recentering(monkeypatch: pytest.MonkeyPatch):
    DummyYOLO, cursor, direct = install_stubs(monkeypatch)

    cursor._position = np.array([100.0, 200.0])

    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    provider = DummyFrameProvider(frames)

    controller_module = importlib.import_module("auto_focus.controller")
    AutoFocusController = controller_module.AutoFocusController

    controller_module.YOLO.result_queue = [
        types.SimpleNamespace(
            boxes=types.SimpleNamespace(
                xyxy=np.array([[400, 240, 440, 280]], dtype=float),
                conf=np.array([0.95], dtype=float),
                cls=np.array([0], dtype=float),
            ),
            names={0: "person"},
        )
    ]

    controller = AutoFocusController(
        target_class="person",
        confidence_threshold=0.5,
        smoothing_factor=0.5,
        tracking_speed=0.2,
        distance_ratio=1.0,
        frame_provider=provider,
        mode="3d",
    )

    assert np.allclose(controller._screen_center, np.array([100.0, 200.0]))

    controller.run()

    # Relative delta should still be emitted via the direct-input bridge.
    assert len(direct.movements) == 1

    # Pointer recentering should respect the cursor's initial position.
    assert len(cursor.movements) == 1
    x, y = cursor.movements[0]
    assert pytest.approx(x, abs=1e-6) == 100.0
    assert pytest.approx(y, abs=1e-6) == 200.0
