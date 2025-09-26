# auto-object-focus

Automatic cursor focusing on detected objects displayed on your screen using YOLOv8.

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
- Text boxes for assigning start and stop keyboard shortcuts (one or two keys separated by `+`).
- Buttons to start/stop the controller and apply hotkey changes. Global shortcuts trigger the same actions even when the interface is not focused.
