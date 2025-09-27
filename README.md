# auto-object-focus

Automatic cursor focusing on detected objects displayed on your screen using YOLOv8.
The controller nudges the mouse relative to its current position so the chosen object
closest to the screen centre is guided back toward the crosshair, moving faster when
the target is far away and slowing down as it settles. You can operate the desktop
pointer directly or emit relative 3D-style input for games that lock the cursor to the
middle of the screen.

## Installation

```bash
pip install .
```

This project requires the following main dependencies which will be installed automatically:

- [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [numpy](https://numpy.org/)
- [pyautogui](https://pyautogui.readthedocs.io/)
- [gradio](https://gradio.app/) for the optional web interface
- [pynput](https://pypi.org/project/pynput/) for global hotkey support
- [mss](https://github.com/BoboTiG/python-mss) for efficient screen capture (falls back to `pyautogui.screenshot` when unavailable)

## Usage

Run the controller with the desired class name:

```bash
python -m auto_focus.cli --class person
```

Additional options include:

- `--class-id` to specify the numeric class identifier instead of a name.
- `--confidence` to adjust the minimum detection confidence (default `0.5`).
- `--smoothing` to control the exponential moving average factor (default `0.2`).
- `--tracking-speed` to define the base movement speed (default `0.2`).
- `--distance-ratio` to scale how much faster the cursor moves when the target is far from the center (default `2.0`).
- `--model` to provide a custom YOLO weights file.
- `--monitor` to choose which display should be captured. `0` (default) observes the entire virtual desktop while higher
  indices select a specific screen from the layout reported by the operating system.
- `--debug-visualization` to pop up a window showing the YOLO detections used for steering. This helps debug tracking issues
  without affecting the live cursor.

The controller always emits raw relative mouse input so games that capture the pointer continue to receive motion. Ensure
[PyDirectInput](https://github.com/learncodebygaming/pydirectinput) is installed so the events are delivered even when the
desktop cursor is hidden.

Press `Ctrl+C` to stop the controller.

## Web interface

Launch the interactive Gradio interface to configure the tracker, adjust motion parameters, and register hotkeys:

```bash
python -m auto_focus.webui
```

### Windows quick launch

Windows users can double-click `run.bat` to automatically verify that the required Python packages are installed and launch the Gradio web interface. The script installs any missing dependencies before starting the app so a single click opens the UI.

The interface provides:

- Text inputs for the target class or ID and model weights.
- Sliders controlling the detection confidence, smoothing factor, base tracking speed, and far/near speed ratio.
- A dropdown for selecting which monitor to capture and an optional debug overlay toggle that renders the detected bounding
  boxes.
- Text boxes for assigning start and stop keyboard shortcuts (one or two keys separated by `+`).
- Buttons to start/stop the controller and apply hotkey changes. Global shortcuts trigger the same actions even when the interface is not focused.
