import { useCallback, useEffect, useRef, useState } from "react";
import type { DragToTargetSpec, HoldProgress } from "../../types.ts";

const HOLD_DURATION_MS = 500;

interface DragToTargetTaskProps {
  spec: DragToTargetSpec;
  onComplete: () => void;
}

export function DragToTargetTask({ spec, onComplete }: DragToTargetTaskProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [holdProgress, setHoldProgress] = useState<HoldProgress>({
    state: "out_of_range",
    startTime: null,
    progress: 0,
  });

  const holdStartRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const completedRef = useRef(false);

  // Initialize position based on container size
  useEffect(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setPosition({
        x: (spec.draggableStart.x / 100) * rect.width,
        y: (spec.draggableStart.y / 100) * rect.height,
      });
    }
  }, [spec.draggableStart]);

  // Check if draggable is in target
  const isInTarget = useCallback(
    (x: number, y: number): boolean => {
      if (!containerRef.current) return false;
      const rect = containerRef.current.getBoundingClientRect();

      const targetX = (spec.targetPosition.x / 100) * rect.width;
      const targetY = (spec.targetPosition.y / 100) * rect.height;
      const targetW = spec.targetSize.width;
      const targetH = spec.targetSize.height;

      const draggableCenter = {
        x: x + spec.draggableSize / 2,
        y: y + spec.draggableSize / 2,
      };

      return (
        draggableCenter.x >= targetX &&
        draggableCenter.x <= targetX + targetW &&
        draggableCenter.y >= targetY &&
        draggableCenter.y <= targetY + targetH
      );
    },
    [spec]
  );

  // Hold timer animation loop - only starts after release (not while dragging)
  useEffect(() => {
    if (completedRef.current) return;

    const inTarget = isInTarget(position.x, position.y);
    const canStartHold = inTarget && !isDragging;

    if (canStartHold && holdProgress.state === "out_of_range") {
      // Just released in target - start hold timer
      holdStartRef.current = performance.now();
      setHoldProgress({
        state: "in_range_pending",
        startTime: holdStartRef.current,
        progress: 0,
      });
    } else if ((!inTarget || isDragging) && holdProgress.state === "in_range_pending") {
      // Left target OR started dragging again - reset timer
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
  }, [position, isDragging, holdProgress.state, isInTarget, onComplete]);

  // Pointer event handlers
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setIsDragging(true);
  }, []);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDragging || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const newX = e.clientX - rect.left - spec.draggableSize / 2;
      const newY = e.clientY - rect.top - spec.draggableSize / 2;

      // Clamp to container bounds
      const clampedX = Math.max(0, Math.min(rect.width - spec.draggableSize, newX));
      const clampedY = Math.max(0, Math.min(rect.height - spec.draggableSize, newY));

      setPosition({ x: clampedX, y: clampedY });
    },
    [isDragging, spec.draggableSize]
  );

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    setIsDragging(false);
  }, []);

  // Compute target position
  const targetStyle = containerRef.current
    ? {
        left: `${spec.targetPosition.x}%`,
        top: `${spec.targetPosition.y}%`,
        width: spec.targetSize.width,
        height: spec.targetSize.height,
      }
    : {};

  const inTarget = isInTarget(position.x, position.y);

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-4xl">
      {/* Instructions */}
      <div className="text-center">
        <h2 className="text-xl font-display font-semibold text-surface-900 mb-2">Drag Task</h2>
        <p className="text-surface-600">
          Drag the circle to the target area, release, and hold position for 0.5 seconds.
        </p>
        {holdProgress.state === "in_range_pending" && (
          <p className="text-accent-600 text-sm mt-1 font-mono">
            Hold: {Math.round(holdProgress.progress * 100)}%
          </p>
        )}
      </div>

      {/* Playground */}
      <div
        ref={containerRef}
        className="relative w-full h-96 bg-surface-200 rounded-xl overflow-hidden"
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        {/* Target area */}
        <div
          className={`absolute rounded-xl border-2 border-dashed transition-colors ${
            inTarget ? "border-accent-500 bg-accent-500/20" : "border-surface-400 bg-surface-300/50"
          }`}
          style={targetStyle}
        >
          {/* Hold progress indicator */}
          {holdProgress.state === "in_range_pending" && (
            <div
              className="absolute bottom-0 left-0 h-1 bg-accent-500 rounded-b-lg hold-indicator"
              style={{ width: `${holdProgress.progress * 100}%` }}
            />
          )}
        </div>

        {/* Draggable */}
        <div
          className={`absolute rounded-full cursor-grab transition-shadow ${
            isDragging
              ? "cursor-grabbing shadow-lg shadow-amber-500/40"
              : "hover:shadow-md hover:shadow-amber-500/30"
          } ${
            inTarget
              ? "bg-gradient-to-br from-accent-400 to-accent-600"
              : "bg-gradient-to-br from-amber-400 to-amber-600"
          }`}
          style={{
            left: position.x,
            top: position.y,
            width: spec.draggableSize,
            height: spec.draggableSize,
          }}
          onPointerDown={handlePointerDown}
        >
          {/* Inner circle */}
          <div className="absolute inset-2 rounded-full bg-white/30" />
        </div>
      </div>
    </div>
  );
}
