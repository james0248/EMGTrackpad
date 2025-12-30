import { useCallback, useEffect, useRef, useState } from "react";
import type { HoldProgress, ScrollMeterSpec } from "../../types.ts";

const HOLD_DURATION_MS = 500;
const SCROLL_IDLE_TIMEOUT_MS = 100; // Time after last scroll to consider "stopped scrolling"

interface ScrollMeterTaskProps {
  spec: ScrollMeterSpec;
  onComplete: () => void;
}

export function ScrollMeterTask({ spec, onComplete }: ScrollMeterTaskProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [value, setValue] = useState(spec.initialValue);
  const [isScrolling, setIsScrolling] = useState(false);
  const [holdProgress, setHoldProgress] = useState<HoldProgress>({
    state: "out_of_range",
    startTime: null,
    progress: 0,
  });

  const holdStartRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const completedRef = useRef(false);
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isHorizontal = spec.kind === "scroll_meter_horizontal";
  const isInTarget = value >= spec.targetMin && value <= spec.targetMax;

  // Hold timer animation loop - only starts after scrolling stops
  useEffect(() => {
    if (completedRef.current) return;

    const canStartHold = isInTarget && !isScrolling;

    if (canStartHold && holdProgress.state === "out_of_range") {
      // Stopped scrolling while in target - start hold timer
      holdStartRef.current = performance.now();
      setHoldProgress({
        state: "in_range_pending",
        startTime: holdStartRef.current,
        progress: 0,
      });
    } else if ((!isInTarget || isScrolling) && holdProgress.state === "in_range_pending") {
      // Left target OR started scrolling again - reset timer
      holdStartRef.current = null;
      setHoldProgress({
        state: "out_of_range",
        startTime: null,
        progress: 0,
      });
    }

    // Animation loop for hold progress
    if (holdProgress.state === "in_range_pending") {
      const animate = () => {
        if (!holdStartRef.current || completedRef.current) return;

        const elapsed = performance.now() - holdStartRef.current;
        const progress = Math.min(1, elapsed / HOLD_DURATION_MS);

        if (progress >= 1) {
          completedRef.current = true;
          setHoldProgress({
            state: "completed",
            startTime: holdStartRef.current,
            progress: 1,
          });
          onComplete();
        } else {
          setHoldProgress((prev) => ({
            ...prev,
            progress,
          }));
          animationFrameRef.current = requestAnimationFrame(animate);
        }
      };

      animationFrameRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isInTarget, isScrolling, holdProgress.state, onComplete]);

  // Cleanup scroll timeout on unmount
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Wheel event handler
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();

      // Mark as scrolling
      setIsScrolling(true);

      // Clear existing timeout
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }

      // Set timeout to mark as not scrolling after idle period
      scrollTimeoutRef.current = setTimeout(() => {
        setIsScrolling(false);
      }, SCROLL_IDLE_TIMEOUT_MS);

      // Use the appropriate delta based on orientation
      const delta = isHorizontal ? e.deltaX || e.deltaY : e.deltaY;
      const change = delta * spec.sensitivity * -0.1;

      setValue((prev) => Math.max(0, Math.min(100, prev + change)));
    },
    [isHorizontal, spec.sensitivity]
  );

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-4xl">
      {/* Instructions */}
      <div className="text-center">
        <h2 className="text-xl font-display font-semibold text-surface-900 mb-2">
          {isHorizontal ? "Horizontal" : "Vertical"} Scroll Task
        </h2>
        <p className="text-surface-600">
          Scroll to move the indicator into the target zone, stop, and hold for 0.5 seconds.
        </p>
        <p className="text-surface-500 text-sm mt-1">
          Current: <span className="font-mono text-surface-700">{Math.round(value)}%</span>
          {" | "}
          Target:{" "}
          <span className="font-mono text-accent-600">
            {Math.round(spec.targetMin)}% - {Math.round(spec.targetMax)}%
          </span>
        </p>
        {holdProgress.state === "in_range_pending" && (
          <p className="text-accent-600 text-sm mt-1 font-mono">
            Hold: {Math.round(holdProgress.progress * 100)}%
          </p>
        )}
      </div>

      {/* Meter container */}
      <div
        ref={containerRef}
        className={`relative bg-surface-200 rounded-xl overflow-hidden cursor-ns-resize ${isHorizontal ? "w-full h-24" : "w-24 h-96"
          }`}
        onWheel={handleWheel}
      >
        {/* Track background */}
        <div
          className={`absolute ${isHorizontal ? "inset-y-4 inset-x-4" : "inset-x-4 inset-y-4"
            } bg-surface-300 rounded-lg`}
        />

        {/* Target zone */}
        <div
          className={`absolute rounded-lg bg-accent-500/30 border-2 border-dashed border-accent-500 ${isHorizontal ? "inset-y-4" : "inset-x-4"
            }`}
          style={
            isHorizontal
              ? {
                left: `calc(${spec.targetMin}% + 16px * ${1 - spec.targetMin / 100})`,
                width: `${spec.targetMax - spec.targetMin}%`,
              }
              : {
                bottom: `calc(${spec.targetMin}% + 16px * ${1 - spec.targetMin / 100})`,
                height: `${spec.targetMax - spec.targetMin}%`,
              }
          }
        />

        {/* Value indicator - no transition for responsiveness */}
        <div
          className={`absolute ${isInTarget ? "bg-accent-500" : "bg-amber-500"} ${isHorizontal ? "w-4 inset-y-2 rounded-full" : "h-4 inset-x-2 rounded-full"
            }`}
          style={
            isHorizontal ? { left: `calc(${value}% - 8px)` } : { bottom: `calc(${value}% - 8px)` }
          }
        >
          {/* Hold progress ring */}
          {holdProgress.state === "in_range_pending" && (
            <div
              className={`absolute hold-indicator ${isHorizontal
                ? "inset-x-0 bottom-0 rounded-b-full bg-white/40"
                : "inset-y-0 left-0 rounded-l-full bg-white/40"
                }`}
              style={
                isHorizontal
                  ? { height: `${holdProgress.progress * 100}%` }
                  : { width: `${holdProgress.progress * 100}%` }
              }
            />
          )}
        </div>

        {/* Scale marks */}
        {[0, 25, 50, 75, 100].map((mark) => (
          <div
            key={mark}
            className={`absolute text-surface-500 text-xs font-mono ${isHorizontal ? "top-0 -translate-x-1/2" : "right-0 translate-y-1/2"
              }`}
            style={isHorizontal ? { left: `${mark}%` } : { bottom: `${mark}%` }}
          >
            {mark}
          </div>
        ))}
      </div>

      {/* Scroll hint */}
      <p className="text-surface-500 text-sm">
        {isHorizontal ? "↔ Scroll horizontally" : "↕ Scroll vertically"} to adjust
      </p>
    </div>
  );
}
