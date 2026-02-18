import time
from enum import IntEnum

import Quartz


class ClickState(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


class VirtualTrackpad:
    def __init__(self):
        self.is_dragging_left = False
        self.is_dragging_right = False
        self.screen_width = Quartz.CGDisplayPixelsWide(Quartz.CGMainDisplayID())
        self.screen_height = Quartz.CGDisplayPixelsHigh(Quartz.CGMainDisplayID())

    def get_current_position(self):
        ev = Quartz.CGEventCreate(None)
        loc = Quartz.CGEventGetLocation(ev)
        return loc.x, loc.y

    def apply(
        self,
        dx: float,
        dy: float,
        click_state: int,  # 0=none, 1=left, 2=right
        scroll_dx: float = 0.0,
        scroll_dy: float = 0.0,
        is_move_enabled: bool = True,
        is_scroll_enabled: bool = True,
    ):
        current_x, current_y = self.get_current_position()
        new_x = current_x + dx
        new_y = current_y + dy

        # Clip to screen boundaries
        new_x = max(0, min(self.screen_width, new_x))
        new_y = max(0, min(self.screen_height, new_y))

        def post_mouse_event(event_type, mouse_button):
            mouse_event = Quartz.CGEventCreateMouseEvent(
                None, event_type, (new_x, new_y), mouse_button
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, mouse_event)

        is_left_click = click_state == ClickState.LEFT
        is_right_click = click_state == ClickState.RIGHT

        # Process click transitions independently from move gating.
        if is_left_click:
            if self.is_dragging_right:
                post_mouse_event(
                    Quartz.kCGEventRightMouseUp,
                    Quartz.kCGMouseButtonRight,
                )
                self.is_dragging_right = False
            if not self.is_dragging_left:
                post_mouse_event(
                    Quartz.kCGEventLeftMouseDown,
                    Quartz.kCGMouseButtonLeft,
                )
                self.is_dragging_left = True
        elif is_right_click:
            if self.is_dragging_left:
                post_mouse_event(
                    Quartz.kCGEventLeftMouseUp,
                    Quartz.kCGMouseButtonLeft,
                )
                self.is_dragging_left = False
            if not self.is_dragging_right:
                post_mouse_event(
                    Quartz.kCGEventRightMouseDown,
                    Quartz.kCGMouseButtonRight,
                )
                self.is_dragging_right = True
        else:
            if self.is_dragging_left:
                post_mouse_event(
                    Quartz.kCGEventLeftMouseUp,
                    Quartz.kCGMouseButtonLeft,
                )
                self.is_dragging_left = False
            if self.is_dragging_right:
                post_mouse_event(
                    Quartz.kCGEventRightMouseUp,
                    Quartz.kCGMouseButtonRight,
                )
                self.is_dragging_right = False

        has_motion = (dx != 0.0) or (dy != 0.0)

        # Emit movement events after transitions.
        if has_motion:
            if self.is_dragging_left:
                post_mouse_event(
                    Quartz.kCGEventLeftMouseDragged,
                    Quartz.kCGMouseButtonLeft,
                )
            elif self.is_dragging_right:
                post_mouse_event(
                    Quartz.kCGEventRightMouseDragged,
                    Quartz.kCGMouseButtonRight,
                )
            elif is_move_enabled:
                post_mouse_event(
                    Quartz.kCGEventMouseMoved,
                    Quartz.kCGMouseButtonLeft,
                )

        # Handle scroll event (separate event)
        if is_scroll_enabled and scroll_dy != 0:
            # Create ScrollWheelEvent (parameters: source, unit, wheel count, wheel 1 change amount...)
            # For smooth scrolling like MacOS trackpad, it may be needed to use pixel unit instead of line unit
            scroll_event = Quartz.CGEventCreateScrollWheelEvent(
                None,
                Quartz.kCGScrollEventUnitPixel,
                1,
                int(scroll_dy * 10),
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, scroll_event)


def main():
    controller = VirtualTrackpad()
    print("EMG Virtual Trackpad started (Press Ctrl+C to exit)")

    try:
        while True:
            pred_dx = 0.0
            pred_dy = 0.0
            click_state = ClickState.NONE
            controller.apply(
                pred_dx, pred_dy, click_state, scroll_dy=10.0, scroll_dx=0.0
            )
            time.sleep(1 / 60)

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
