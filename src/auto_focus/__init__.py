"""Auto focus package for cursor control using object detection."""

from .controller import AutoFocusController
from .webui import launch_interface

__all__ = ["AutoFocusController", "launch_interface"]
