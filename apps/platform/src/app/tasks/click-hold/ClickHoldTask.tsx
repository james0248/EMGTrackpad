import { useCallback, useEffect, useRef, useState } from "react";
import type { ClickHoldSpec, HoldProgress } from "../../types.ts";

const CIRCLE_SIZE = 120;

interface ClickHoldTaskProps {
  spec: ClickHoldSpec;
  onComplete: () => void;
}

export function ClickHoldTask({ spec, onComplete }: ClickHoldTaskProps) {
  const [holdProgress, setHoldProgress] = useState<HoldProgress>({
    state: "out_of_range",
    startTime: null,
    progress: 0,
  });

  const holdStartRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const completedRef = useRef(false);

  const isLeft = spec.requiredButton === "left";
  const circleColor = isLeft ? "rgb(34 211 238)" : "rgb(251 113 133)"; // cyan-400 / rose-400
  const instruction = isLeft ? "Hold Left Click" : "Hold Right Click";

  // Animation loop for hold progress
  useEffect(() => {
    if (holdProgress.state !== "in_range_pending" || completedRef.current) {
      return;
    }

    const animate = () => {
      if (!holdStartRef.current || completedRef.current) return;

      const elapsed = performance.now() - holdStartRef.current;
      const progress = Math.min(1, elapsed / spec.holdDurationMs);

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

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [holdProgress.state, onComplete, spec.holdDurationMs]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      if (completedRef.current) return;

      // Check if correct button
      const isLeftClick = e.button === 0;
      const isCorrectButton = isLeft ? isLeftClick : !isLeftClick;

      if (isCorrectButton) {
        holdStartRef.current = performance.now();
        setHoldProgress({
          state: "in_range_pending",
          startTime: holdStartRef.current,
          progress: 0,
        });
      }
    },
    [isLeft]
  );

  const handleMouseUp = useCallback(() => {
    if (completedRef.current) return;

    holdStartRef.current = null;
    setHoldProgress({
      state: "out_of_range",
      startTime: null,
      progress: 0,
    });
  }, []);

  const handleMouseLeave = useCallback(() => {
    if (completedRef.current) return;

    holdStartRef.current = null;
    setHoldProgress({
      state: "out_of_range",
      startTime: null,
      progress: 0,
    });
  }, []);

  const isPending = holdProgress.state === "in_range_pending";
  const progress = holdProgress.progress;

  return (
    <div className="w-full h-full flex flex-col items-center justify-center select-none">
      <div className="text-surface-600 text-lg mb-8">{instruction}</div>

      <div
        className="relative cursor-pointer"
        style={{ width: CIRCLE_SIZE, height: CIRCLE_SIZE }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onContextMenu={(e) => e.preventDefault()}
      >
        {/* Main circle */}
        <div
          className="absolute inset-0 rounded-full transition-transform"
          style={{
            backgroundColor: circleColor,
            transform: isPending ? "scale(1.05)" : "scale(1)",
          }}
        />

        {/* Hold progress indicator - circular ring */}
        {isPending && (
          <svg
            className="absolute inset-0 w-full h-full -rotate-90"
            viewBox="0 0 100 100"
            aria-hidden="true"
          >
            <circle
              cx="50"
              cy="50"
              r="46"
              fill="none"
              stroke="white"
              strokeWidth="4"
              strokeDasharray={`${progress * 289} 289`}
              strokeLinecap="round"
            />
          </svg>
        )}

        {/* Completed checkmark */}
        {holdProgress.state === "completed" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <svg
              className="w-12 h-12 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={3}
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          </div>
        )}
      </div>
    </div>
  );
}
