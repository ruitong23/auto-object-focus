"""Gradio-powered interface for configuring and running the auto focus controller."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Union

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - handled in runtime usage
    raise ImportError(
        "gradio is required to launch the web UI. Install the optional dependencies first."
    ) from exc

try:
    from pynput import keyboard
except ImportError as exc:  # pragma: no cover - handled in runtime usage
    raise ImportError(
        "pynput is required to use the hotkey functionality. Install the optional dependencies first."
    ) from exc

from .controller import AutoFocusController


@dataclass
class AppState:
    """Mutable application state shared across callbacks."""

    target: Optional[Union[str, int]] = None
    confidence: float = 0.5
    smoothing: float = 0.2
    tracking_speed: float = 0.2
    distance_ratio: float = 2.0
    model_path: str = "yolov8n.pt"
    mode: str = "2d"
    def to_controller_kwargs(self) -> dict:
        return {
            "target_class": self.target,
            "confidence_threshold": self.confidence,
            "smoothing_factor": self.smoothing,
            "tracking_speed": self.tracking_speed,
            "distance_ratio": self.distance_ratio,
            "model_path": self.model_path,
            "mode": self.mode,
        }


class ControllerRunner:
    """Manage the lifecycle of :class:`AutoFocusController` instances."""

    def __init__(self) -> None:
        self._controller: Optional[AutoFocusController] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self, **kwargs) -> tuple[bool, str]:
        with self._lock:
            if self._controller is not None:
                return False, "Controller already running."

            controller = AutoFocusController(**kwargs)
            thread = threading.Thread(target=controller.run, daemon=True)
            self._controller = controller
            self._thread = thread
            thread.start()

        return True, "Controller started."

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            controller = self._controller
            thread = self._thread
            self._controller = None
            self._thread = None

        if controller is None:
            return False, "Controller is not running."

        controller.stop()
        if thread is not None:
            thread.join(timeout=2.0)
        controller.close()
        return True, "Controller stopped."

    def update_motion(self, tracking_speed: float, distance_ratio: float) -> None:
        with self._lock:
            if self._controller is not None:
                self._controller.update_motion_parameters(
                    tracking_speed=tracking_speed, distance_ratio=distance_ratio
                )


class HotkeyManager:
    """Register and manage global hotkeys using :mod:`pynput`."""

    def __init__(self) -> None:
        self._listener: Optional[keyboard.GlobalHotKeys] = None
        self._lock = threading.Lock()

    def configure(
        self,
        start_combo: Optional[str],
        stop_combo: Optional[str],
        on_start,
        on_stop,
    ) -> None:
        with self._lock:
            self._shutdown_unlocked()

            hotkeys: dict[str, callable] = {}
            if start_combo:
                hotkeys[self._format_combo(start_combo)] = on_start
            if stop_combo:
                hotkeys[self._format_combo(stop_combo)] = on_stop

            if hotkeys:
                listener = keyboard.GlobalHotKeys(hotkeys)
                listener.start()
                self._listener = listener

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_unlocked()

    def _shutdown_unlocked(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _format_combo(combo: str) -> str:
        tokens = [token.strip().lower() for token in combo.split("+") if token.strip()]
        if not 1 <= len(tokens) <= 2:
            raise ValueError("Hotkeys must contain one or two keys separated by '+'.")

        replacements = {
            "ctrl": "<ctrl>",
            "control": "<ctrl>",
            "shift": "<shift>",
            "alt": "<alt>",
            "option": "<alt>",
            "cmd": "<cmd>",
            "command": "<cmd>",
            "win": "<cmd>",
        }

        mapped = [replacements.get(token, token) for token in tokens]
        return "+".join(mapped)


def _parse_target(value: Optional[str]) -> Optional[Union[str, int]]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return stripped


def launch_interface(**launch_kwargs) -> gr.Blocks:
    """Launch the Gradio interface."""

    runner = ControllerRunner()
    hotkeys = HotkeyManager()
    state = AppState()
    state_lock = threading.Lock()

    def _start_from_hotkey():  # pragma: no cover - side-effect callback
        with state_lock:
            params = state.to_controller_kwargs()
        success, message = runner.start(**params)
        if not success:
            print(message)

    def _stop_from_hotkey():  # pragma: no cover - side-effect callback
        success, message = runner.stop()
        if not success:
            print(message)

    def start_controller(
        target_value: str,
        confidence: float,
        smoothing: float,
        tracking_speed: float,
        distance_ratio: float,
        model_path: str,
        mode: str,
    ) -> str:
        with state_lock:
            state.target = _parse_target(target_value)
            state.confidence = confidence
            state.smoothing = smoothing
            state.tracking_speed = tracking_speed
            state.distance_ratio = distance_ratio
            state.model_path = model_path.strip() or "yolov8n.pt"
            state.mode = mode
            params = state.to_controller_kwargs()

        success, message = runner.start(**params)
        return f"Status: {message}"

    def stop_controller() -> str:
        success, message = runner.stop()
        return f"Status: {message}"

    def update_motion(tracking_speed: float, distance_ratio: float) -> str:
        with state_lock:
            state.tracking_speed = tracking_speed
            state.distance_ratio = distance_ratio
        runner.update_motion(tracking_speed, distance_ratio)
        return "Status: Updated tracking behaviour."

    def register_hotkeys(start_combo: str, stop_combo: str) -> str:
        try:
            hotkeys.configure(start_combo, stop_combo, _start_from_hotkey, _stop_from_hotkey)
        except ValueError as exc:
            return f"Status: Hotkey error - {exc}" 
        return "Status: Hotkeys registered."

    def update_thresholds(confidence: float, smoothing: float) -> str:
        with state_lock:
            state.confidence = confidence
            state.smoothing = smoothing
        return "Status: Updated detection thresholds."

    def update_mode(mode: str) -> str:
        with state_lock:
            state.mode = mode
        return f"Status: Tracking mode set to {mode.upper()}."

    def cleanup():  # pragma: no cover - triggered on UI shutdown
        runner.stop()
        hotkeys.shutdown()

    with gr.Blocks(title="Auto Object Focus") as demo:
        gr.Markdown(
            "## Auto Object Focus\n"
            "Configure YOLO powered cursor tracking, adjust motion behaviour, and register hotkeys."
        )

        with gr.Row():
            target_input = gr.Textbox(label="Target Class or ID", value="person", placeholder="person")
            model_input = gr.Textbox(label="YOLO Model", value="yolov8n.pt")

        with gr.Row():
            confidence_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Confidence Threshold",
            )
            smoothing_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.2,
                step=0.05,
                label="Smoothing Factor",
            )

        with gr.Row():
            tracking_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=0.2,
                step=0.05,
                label="Tracking Speed",
            )
            distance_slider = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=2.0,
                step=0.1,
                label="Far/Near Speed Ratio",
            )

        mode_selector = gr.Radio(
            choices=["2d", "3d"],
            value="2d",
            label="Tracking Mode",
            info="2D moves the desktop cursor, 3D emits relative input for locked-pointer apps.",
        )

        with gr.Row():
            start_hotkey = gr.Textbox(
                label="Start Hotkey",
                placeholder="e.g. ctrl+shift+s",
            )
            stop_hotkey = gr.Textbox(
                label="Stop Hotkey",
                placeholder="e.g. ctrl+shift+x",
            )

        status_display = gr.Markdown("Status: Idle")

        with gr.Row():
            start_button = gr.Button("Start", variant="primary")
            stop_button = gr.Button("Stop")
            hotkey_button = gr.Button("Apply Hotkeys")

        start_button.click(
            start_controller,
            inputs=[
                target_input,
                confidence_slider,
                smoothing_slider,
                tracking_slider,
                distance_slider,
                model_input,
                mode_selector,
            ],
            outputs=status_display,
        )

        stop_button.click(stop_controller, outputs=status_display)
        hotkey_button.click(
            register_hotkeys,
            inputs=[start_hotkey, stop_hotkey],
            outputs=status_display,
        )

        tracking_slider.change(update_motion, inputs=[tracking_slider, distance_slider], outputs=status_display)
        distance_slider.change(update_motion, inputs=[tracking_slider, distance_slider], outputs=status_display)
        confidence_slider.change(
            update_thresholds, inputs=[confidence_slider, smoothing_slider], outputs=status_display
        )
        smoothing_slider.change(
            update_thresholds, inputs=[confidence_slider, smoothing_slider], outputs=status_display
        )
        mode_selector.change(update_mode, inputs=mode_selector, outputs=status_display)

    try:
        demo.launch(**launch_kwargs)
    finally:
        cleanup()
    return demo


def main() -> None:
    """Entry point for ``python -m auto_focus.webui``."""

    launch_interface()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
