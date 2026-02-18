import unittest
from types import SimpleNamespace
from unittest.mock import patch

from emg.inference import trackpad as trackpad_module


class FakeQuartz:
    kCGMouseButtonLeft = 0
    kCGMouseButtonRight = 1
    kCGHIDEventTap = "hid_tap"
    kCGScrollEventUnitPixel = "pixel"

    kCGEventLeftMouseDown = "left_down"
    kCGEventLeftMouseUp = "left_up"
    kCGEventLeftMouseDragged = "left_dragged"
    kCGEventRightMouseDown = "right_down"
    kCGEventRightMouseUp = "right_up"
    kCGEventRightMouseDragged = "right_dragged"
    kCGEventMouseMoved = "mouse_moved"

    def __init__(self):
        self.posts = []
        self.cursor = (100.0, 100.0)

    def CGMainDisplayID(self):
        return 0

    def CGDisplayPixelsWide(self, _display_id):
        return 1920

    def CGDisplayPixelsHigh(self, _display_id):
        return 1080

    def CGEventCreate(self, _source):
        return object()

    def CGEventGetLocation(self, _event):
        return SimpleNamespace(x=self.cursor[0], y=self.cursor[1])

    def CGEventCreateMouseEvent(self, _source, event_type, position, button):
        return {"event_type": event_type, "position": position, "button": button}

    def CGEventPost(self, _tap, event):
        self.posts.append(event)
        if "position" in event:
            self.cursor = event["position"]

    def CGEventCreateScrollWheelEvent(self, _source, _unit, _wheel_count, delta):
        return {"event_type": "scroll", "delta": delta}


class VirtualTrackpadEventTests(unittest.TestCase):
    def setUp(self):
        self.fake_quartz = FakeQuartz()
        self.quartz_patcher = patch.object(trackpad_module, "Quartz", self.fake_quartz)
        self.quartz_patcher.start()
        self.addCleanup(self.quartz_patcher.stop)
        self.trackpad = trackpad_module.VirtualTrackpad()

    def event_types(self):
        return [event["event_type"] for event in self.fake_quartz.posts]

    def test_right_click_transition_without_move_gate(self):
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.RIGHT,
            is_move_enabled=False,
        )
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.NONE,
            is_move_enabled=False,
        )

        self.assertEqual(
            self.event_types(),
            [self.fake_quartz.kCGEventRightMouseDown, self.fake_quartz.kCGEventRightMouseUp],
        )

    def test_right_drag_emits_dragged_when_moving(self):
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.RIGHT,
            is_move_enabled=False,
        )
        self.trackpad.apply(
            dx=6.0,
            dy=-4.0,
            click_state=trackpad_module.ClickState.RIGHT,
            is_move_enabled=False,
        )
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.NONE,
            is_move_enabled=False,
        )

        self.assertEqual(
            self.event_types(),
            [
                self.fake_quartz.kCGEventRightMouseDown,
                self.fake_quartz.kCGEventRightMouseDragged,
                self.fake_quartz.kCGEventRightMouseUp,
            ],
        )

    def test_stationary_hold_does_not_emit_repeated_drag_event(self):
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.RIGHT,
            is_move_enabled=True,
        )
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.RIGHT,
            is_move_enabled=True,
        )
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.NONE,
            is_move_enabled=True,
        )

        self.assertEqual(
            self.event_types(),
            [self.fake_quartz.kCGEventRightMouseDown, self.fake_quartz.kCGEventRightMouseUp],
        )

    def test_free_movement_still_respects_move_gate(self):
        self.trackpad.apply(
            dx=5.0,
            dy=3.0,
            click_state=trackpad_module.ClickState.NONE,
            is_move_enabled=False,
        )
        self.assertEqual(self.event_types(), [])

        self.trackpad.apply(
            dx=5.0,
            dy=3.0,
            click_state=trackpad_module.ClickState.NONE,
            is_move_enabled=True,
        )
        self.assertEqual(self.event_types(), [self.fake_quartz.kCGEventMouseMoved])

    def test_switching_buttons_releases_then_presses(self):
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.LEFT,
            is_move_enabled=True,
        )
        self.trackpad.apply(
            dx=0.0,
            dy=0.0,
            click_state=trackpad_module.ClickState.RIGHT,
            is_move_enabled=True,
        )

        self.assertEqual(
            self.event_types(),
            [
                self.fake_quartz.kCGEventLeftMouseDown,
                self.fake_quartz.kCGEventLeftMouseUp,
                self.fake_quartz.kCGEventRightMouseDown,
            ],
        )


if __name__ == "__main__":
    unittest.main()
