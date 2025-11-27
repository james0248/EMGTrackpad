import Quartz
import time


class VirtualTrackpad:
    def __init__(self):
        self.is_dragging = False
        self.screen_width = Quartz.CGDisplayPixelsWide(Quartz.CGMainDisplayID())
        self.screen_height = Quartz.CGDisplayPixelsHigh(Quartz.CGMainDisplayID())

    def get_current_position(self):
        ev = Quartz.CGEventCreate(None)
        loc = Quartz.CGEventGetLocation(ev)
        return loc.x, loc.y

    def apply(
        self,
        dx: int,
        dy: int,
        is_click_active: bool,
        scroll_dx: int = 0,
        scroll_dy: int = 0,
    ):
        current_x, current_y = self.get_current_position()
        new_x = current_x + dx
        new_y = current_y + dy

        # Clip to screen boundaries
        new_x = max(0, min(self.screen_width, new_x))
        new_y = max(0, min(self.screen_height, new_y))

        # Determine event type (core logic of dragging logic)
        # In MacOS, 'click and move' is not MouseMoved but LeftMouseDragged
        event_type = None

        if is_click_active:
            if not self.is_dragging:
                # Just pressed (Press)
                event_type = Quartz.kCGEventLeftMouseDown
                self.is_dragging = True
            else:
                # Moving while pressed (Drag) -> if position changed, Dragged, otherwise keep the same
                event_type = Quartz.kCGEventLeftMouseDragged
        else:
            if self.is_dragging:
                # Just released (Release)
                event_type = Quartz.kCGEventLeftMouseUp
                self.is_dragging = False
            else:
                # Just moved (Move)
                event_type = Quartz.kCGEventMouseMoved

        # Create and send mouse move/click event
        # Send only if position needs to be updated or click state has changed
        mouse_event = Quartz.CGEventCreateMouseEvent(
            None, event_type, (new_x, new_y), Quartz.kCGMouseButtonLeft
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, mouse_event)

        # Handle scroll event (separate event)
        if scroll_dy != 0:
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
            pred_dx = 0
            pred_dy = 0
            pred_click = 0
            controller.apply(pred_dx, pred_dy, pred_click, scroll_dy=10, scroll_dx=0)
            time.sleep(1 / 60)

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
