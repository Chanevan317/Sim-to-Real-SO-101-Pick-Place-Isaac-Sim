"""Text overlay in the Isaac Sim viewport. Import only after the app is launched.

The overlay is UI only: it is not rendered into the robot's cameras, so it never shows up
in recorded images.
"""

# omni.ui colors are 0xAABBGGRR.
COLORS = {
    "red": 0xFF3333FF,
    "blue": 0xFFFF8833,
    "white": 0xFFFFFFFF,
}
FONT_SIZE = 24


class ViewportLabel:
    """A label in the top-right corner of the viewport. Does nothing when running headless."""

    def __init__(self):
        self._label = None
        try:
            self._build()
        except Exception as e:  # the label is a convenience: never stop a recording session for it
            self._label = None
            print(f"[WARNING]: Could not create the viewport label ({e}); the target is still printed in the terminal.")

    def _build(self):
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport_window

        window = get_active_viewport_window()
        if window is None:  # headless
            return
        self._frame = window.get_frame("sim_to_real_so101.label")
        with self._frame:
            with ui.VStack():
                ui.Spacer(height=50)  # below the viewport's own toolbar
                with ui.HStack(height=0):
                    ui.Spacer()  # pushes the label to the right edge
                    with ui.ZStack(width=0, height=0):
                        ui.Rectangle(style={"background_color": 0xB0000000, "border_radius": 6})
                        with ui.HStack(width=0, height=0):
                            ui.Spacer(width=14)
                            self._label = ui.Label("", height=36, style={"font_size": FONT_SIZE, "color": COLORS["white"]})
                            ui.Spacer(width=14)
                    ui.Spacer(width=20)
                ui.Spacer()

    def set(self, text, color="white"):
        if self._label is None:
            return
        try:
            self._label.text = text
            self._label.set_style({"font_size": FONT_SIZE, "color": COLORS.get(color, COLORS["white"])})
        except Exception as e:
            print(f"[WARNING]: Could not update the viewport label ({e}).")
