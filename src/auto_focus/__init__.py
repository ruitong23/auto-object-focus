"""Auto focus package for cursor control using object detection."""

from .controller import AutoFocusController

__all__ = ["AutoFocusController", "launch_interface"]


def launch_interface(*args, **kwargs):
    """Lazy import the web UI launcher to avoid circular imports."""

    from .webui import launch_interface as _launch_interface

    return _launch_interface(*args, **kwargs)
