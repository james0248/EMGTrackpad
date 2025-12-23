import Quartz
import time
import sys


def mouse_event_callback(proxy, event_type, event, refcon):
    ts = f"{time.perf_counter():.6f}"

    # Scroll event handling
    if event_type == Quartz.kCGEventScrollWheel:
        # Precision pixel-based scroll delta
        scroll_dx = Quartz.CGEventGetDoubleValueField(
            event, Quartz.kCGScrollWheelEventPointDeltaAxis2
        )
        scroll_dy = Quartz.CGEventGetDoubleValueField(
            event, Quartz.kCGScrollWheelEventPointDeltaAxis1
        )

        if scroll_dy != 0 or scroll_dx != 0:
            print(f"[{ts}] SCROLL | Pixel: {scroll_dy:10.6f}, {scroll_dx:10.6f}")

    # Cursor movement or drag event handling
    elif event_type in [Quartz.kCGEventMouseMoved, Quartz.kCGEventLeftMouseDragged]:
        dx = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaX)
        dy = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaY)

        if dx != 0 or dy != 0:
            status = "MOVE" if event_type == Quartz.kCGEventMouseMoved else "DRAG"
            print(f"[{ts}] {status} | dx: {dx:10.6f}, dy: {dy:10.6f}")

    return event


def run_event_capture():
    # Define which events to listen to using a bitmask
    event_mask = (
        Quartz.CGEventMaskBit(Quartz.kCGEventMouseMoved)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDragged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventScrollWheel)
    )

    # Create an event tap to intercept system-level events
    event_tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        event_mask,
        mouse_event_callback,
        None,
    )

    if not event_tap:
        print(
            "Error: Unable to create event tap. Please check 'Accessibility' permissions in System Settings."
        )
        sys.exit(1)

    # Create a run loop source and add it to the current run loop
    run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, event_tap, 0)
    loop = Quartz.CFRunLoopGetCurrent()
    Quartz.CFRunLoopAddSource(loop, run_loop_source, Quartz.kCFRunLoopDefaultMode)

    # Enable the event tap
    Quartz.CGEventTapEnable(event_tap, True)

    print("Quartz Precision Collector Running... (Press Ctrl+C to stop)")
    try:
        Quartz.CFRunLoopRun()
    except KeyboardInterrupt:
        print("\nStopping collection. Exit.")


if __name__ == "__main__":
    run_event_capture()
