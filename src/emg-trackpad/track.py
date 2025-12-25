import Quartz
import time
import sys


def mouse_event_callback(proxy, event_type, event, refcon):
    ts = f"{time.perf_counter():.6f}"

    # Extract Delta values (Relative movement)
    dx = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaX)
    dy = Quartz.CGEventGetDoubleValueField(event, Quartz.kCGMouseEventDeltaY)

    # 1. Left Mouse Events
    if event_type == Quartz.kCGEventLeftMouseDown:
        print(f"[{ts}] LEFT DOWN  | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventLeftMouseUp:
        print(f"[{ts}] LEFT UP    | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventLeftMouseDragged:
        if dx != 0 or dy != 0:
            print(f"[{ts}] L-DRAGGING | dx: {dx:10.6f}, dy: {dy:10.6f}")

    # 2. Right Mouse Events
    elif event_type == Quartz.kCGEventRightMouseDown:
        print(f"[{ts}] RIGHT DOWN | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventRightMouseUp:
        print(f"[{ts}] RIGHT UP   | dx: {0.0:10.6f}, dy: {0.0:10.6f}")

    elif event_type == Quartz.kCGEventRightMouseDragged:
        if dx != 0 or dy != 0:
            print(f"[{ts}] R-DRAGGING | dx: {dx:10.6f}, dy: {dy:10.6f}")

    # 3. Scroll event handling
    elif event_type == Quartz.kCGEventScrollWheel:
        scroll_dx = Quartz.CGEventGetDoubleValueField(
            event, Quartz.kCGScrollWheelEventPointDeltaAxis2
        )
        scroll_dy = Quartz.CGEventGetDoubleValueField(
            event, Quartz.kCGScrollWheelEventPointDeltaAxis1
        )
        if scroll_dy != 0 or scroll_dx != 0:
            print(f"[{ts}] SCROLL     | dx: {scroll_dx:10.6f}, dy: {scroll_dy:10.6f}")

    # 4. Pure Move event
    elif event_type == Quartz.kCGEventMouseMoved:
        if dx != 0 or dy != 0:
            print(f"[{ts}] MOVE       | dx: {dx:10.6f}, dy: {dy:10.6f}")

    return event


def run_event_capture():
    event_mask = (
        Quartz.CGEventMaskBit(Quartz.kCGEventMouseMoved)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp)
        | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDragged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
        | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseUp)
        | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDragged)
        | Quartz.CGEventMaskBit(Quartz.kCGEventScrollWheel)
    )

    event_tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        event_mask,
        mouse_event_callback,
        None,
    )

    if not event_tap:
        print("Error: Accessibility permissions required.")
        sys.exit(1)

    run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, event_tap, 0)
    loop = Quartz.CFRunLoopGetCurrent()
    Quartz.CFRunLoopAddSource(loop, run_loop_source, Quartz.kCFRunLoopDefaultMode)
    Quartz.CGEventTapEnable(event_tap, True)

    print("Quartz Simple Button/Delta Tracker Running... (Ctrl+C to stop)")
    try:
        Quartz.CFRunLoopRun()
    except KeyboardInterrupt:
        print("\nExit.")


if __name__ == "__main__":
    run_event_capture()
