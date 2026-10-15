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
    """Labels in the top-right corner of the viewport: the current episode number, then the target cube.
    Does nothing when there is no viewport."""

    def __init__(self):
        self._episode = None
        self._target = None
        self._target_box = None
        try:
            self._build()
        except Exception as e:  # the labels are a convenience: never stop a recording session for them
            self._episode = self._target = None
            print(f"[WARNING]: Could not create the viewport labels ({e}); the terminal still shows everything.")

    def _build(self):
        import omni.ui as ui
        from omni.kit.viewport.utility import get_active_viewport_window

        window = get_active_viewport_window()
        if window is None:  # headless
            return

        def boxed_label():
            box = ui.ZStack(width=0, height=0)
            with box:
                ui.Rectangle(style={"background_color": 0xB0000000, "border_radius": 6})
                with ui.HStack(width=0, height=0):
                    ui.Spacer(width=14)
                    label = ui.Label("", height=36, style={"font_size": FONT_SIZE, "color": COLORS["white"]})
                    ui.Spacer(width=14)
            return box, label

        self._frame = window.get_frame("sim_to_real_so101.label")
        with self._frame:
            with ui.VStack():
                ui.Spacer(height=50)  # below the viewport's own toolbar
                with ui.HStack(height=0):
                    ui.Spacer()  # pushes the labels to the right edge
                    _, self._episode = boxed_label()
                    ui.Spacer(width=10)
                    self._target_box, self._target = boxed_label()
                    ui.Spacer(width=20)
                ui.Spacer()
        self._target_box.visible = False  # shown once a target is set (two-cube task)

    def set_episode(self, index):
        """Number of the episode being recorded, as in its file name (episode_0012.h5 -> 0012)."""
        if self._episode is None:
            return
        try:
            self._episode.text = f"Episode {index:04d}"
        except Exception as e:
            print(f"[WARNING]: Could not update the viewport label ({e}).")

    def set_target(self, text, color="white"):
        if self._target is None:
            return
        try:
            self._target.text = text
            self._target.set_style({"font_size": FONT_SIZE, "color": COLORS.get(color, COLORS["white"])})
            self._target_box.visible = True
        except Exception as e:
            print(f"[WARNING]: Could not update the viewport label ({e}).")
