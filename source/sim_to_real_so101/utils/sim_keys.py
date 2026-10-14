"""Key presses in the Isaac Sim window. Import only after the app is launched."""


class SimKeys:
    """Listens for key presses in the Isaac Sim window (the window must have focus)."""

    def __init__(self):
        import carb
        import omni.appwindow

        self._carb = carb
        self._pressed = set()
        self._input = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(keyboard, self._on_event)

    def _on_event(self, event, *args, **kwargs):
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(event.input.name)
        return True

    def pop(self, key):
        """True once per press of `key`."""
        if key in self._pressed:
            self._pressed.discard(key)
            return True
        return False
