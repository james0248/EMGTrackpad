import { useCallback, useEffect, useRef, useState } from "react";
import type { Draggable, DragToTargetSpec, HoldProgress } from "../../types.ts";

const HOLD_DURATION_MS = 500;

interface DragToTargetTaskProps {
  spec: DragToTargetSpec;
  onComplete: () => void;
}

export function DragToTargetTask({ spec, onComplete }: DragToTargetTaskProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [draggables, setDraggables] = useState<Draggable[]>(spec.draggables);
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [activeDraggable, setActiveDraggable] = useState<string | null>(null);
  const [holdProgress, setHoldProgress] = useState<Record<string, HoldProgress>>({});

  const holdStartRef = useRef<Record<string, number | null>>({});
  const animationFrameRef = useRef<number | null>(null);
  const completedRef = useRef<Set<string>>(new Set());

  // Initialize positions based on container size
  useEffect(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const initialPositions: Record<string, { x: number; y: number }> = {};
      for (const d of spec.draggables) {
        initialPositions[d.id] = {
          x: (d.position.x / 100) * rect.width,
          y: (d.position.y / 100) * rect.height,
        };
      }
      setPositions(initialPositions);
    }
  }, [spec.draggables]);

  // Get target for a draggable
  const getTarget = useCallback(
    (targetId: string) => spec.targets.find((t) => t.id === targetId),
    [spec.targets]
  );

  // Check if draggable is in its target
  const isInTarget = useCallback(
    (draggableId: string, x: number, y: number): boolean => {
      if (!containerRef.current) return false;
      const rect = containerRef.current.getBoundingClientRect();

      const draggable = draggables.find((d) => d.id === draggableId);
      if (!draggable) return false;

      const target = getTarget(draggable.targetId);
      if (!target) return false;

      const targetCenterX = (target.position.x / 100) * rect.width + spec.targetSize / 2;
      const targetCenterY = (target.position.y / 100) * rect.height + spec.targetSize / 2;
      const targetRadius = spec.targetSize / 2;

      const draggableCenterX = x + spec.draggableSize / 2;
      const draggableCenterY = y + spec.draggableSize / 2;

      // Check if draggable center is within target circle
      const dx = draggableCenterX - targetCenterX;
      const dy = draggableCenterY - targetCenterY;
      const distance = Math.sqrt(dx * dx + dy * dy);

      return distance <= targetRadius;
    },
    [draggables, getTarget, spec.draggableSize, spec.targetSize]
  );

  // Hold timer animation loop
  useEffect(() => {
    const checkHoldProgress = () => {
      let needsUpdate = false;
      const newHoldProgress = { ...holdProgress };

      for (const d of draggables) {
        if (d.done || completedRef.current.has(d.id)) continue;

        const pos = positions[d.id];
        if (!pos) continue;

        const inTarget = isInTarget(d.id, pos.x, pos.y);
        const isBeingDragged = activeDraggable === d.id;
        const canStartHold = inTarget && !isBeingDragged;
        const currentState = holdProgress[d.id]?.state ?? "out_of_range";

        if (canStartHold && currentState === "out_of_range") {
          holdStartRef.current[d.id] = performance.now();
          newHoldProgress[d.id] = {
            state: "in_range_pending",
            startTime: holdStartRef.current[d.id],
            progress: 0,
          };
          needsUpdate = true;
        } else if ((!inTarget || isBeingDragged) && currentState === "in_range_pending") {
          holdStartRef.current[d.id] = null;
          newHoldProgress[d.id] = {
            state: "out_of_range",
            startTime: null,
            progress: 0,
          };
          needsUpdate = true;
        }
      }

      if (needsUpdate) {
        setHoldProgress(newHoldProgress);
      }
    };

    checkHoldProgress();
  }, [positions, activeDraggable, draggables, holdProgress, isInTarget]);

  // Animation loop for hold progress
  useEffect(() => {
    const animate = () => {
      let needsUpdate = false;
      const newHoldProgress = { ...holdProgress };
      const newDraggables = [...draggables];
      let allDone = true;

      for (let i = 0; i < newDraggables.length; i++) {
        const d = newDraggables[i]!;
        if (d.done || completedRef.current.has(d.id)) continue;

        allDone = false;
        const startTime = holdStartRef.current[d.id];
        if (!startTime || holdProgress[d.id]?.state !== "in_range_pending") continue;

        const elapsed = performance.now() - startTime;
        const progress = Math.min(1, elapsed / HOLD_DURATION_MS);

        if (progress >= 1) {
          completedRef.current.add(d.id);
          newHoldProgress[d.id] = {
            state: "completed",
            startTime,
            progress: 1,
          };
          newDraggables[i] = { ...d, done: true };
          needsUpdate = true;
        } else if (progress !== holdProgress[d.id]?.progress) {
          newHoldProgress[d.id] = {
            ...holdProgress[d.id]!,
            progress,
          };
          needsUpdate = true;
        }
      }

      if (needsUpdate) {
        setHoldProgress(newHoldProgress);
        setDraggables(newDraggables);
      }

      // Check if all done
      const nowAllDone = newDraggables.every((d) => d.done);
      if (nowAllDone && !allDone) {
        setTimeout(onComplete, 100);
        return;
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [draggables, holdProgress, onComplete]);

  // Pointer event handlers
  const handlePointerDown = useCallback((e: React.PointerEvent, draggableId: string) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setActiveDraggable(draggableId);
  }, []);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!activeDraggable || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const newX = e.clientX - rect.left - spec.draggableSize / 2;
      const newY = e.clientY - rect.top - spec.draggableSize / 2;

      // Clamp to container bounds
      const clampedX = Math.max(0, Math.min(rect.width - spec.draggableSize, newX));
      const clampedY = Math.max(0, Math.min(rect.height - spec.draggableSize, newY));

      setPositions((prev) => ({
        ...prev,
        [activeDraggable]: { x: clampedX, y: clampedY },
      }));
    },
    [activeDraggable, spec.draggableSize]
  );

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    setActiveDraggable(null);
  }, []);

  const doneCount = draggables.filter((d) => d.done).length;

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-4xl">
      {/* Instructions */}
      <div className="text-center">
        <h2 className="text-xl font-display font-semibold text-surface-900 mb-2">Drag Task</h2>
        <p className="text-surface-600">
          Drag each circle to its matching colored target, release, and hold for 0.5 seconds.
        </p>
        <p className="text-surface-500 text-sm mt-1">
          Progress: {doneCount} / {draggables.length}
        </p>
      </div>

      {/* Playground */}
      <div
        ref={containerRef}
        className="relative w-full h-96 bg-surface-200 rounded-xl overflow-hidden"
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        {/* Target areas */}
        {spec.targets.map((target) => {
          const draggable = draggables.find((d) => d.targetId === target.id);
          const pos = draggable ? positions[draggable.id] : null;
          const inTarget = draggable && pos ? isInTarget(draggable.id, pos.x, pos.y) : false;
          const isDone = draggable?.done ?? false;
          const progress = draggable ? holdProgress[draggable.id]?.progress ?? 0 : 0;
          const isPending = draggable ? holdProgress[draggable.id]?.state === "in_range_pending" : false;

          return (
            <div
              key={target.id}
              className={`absolute rounded-full border-2 border-dashed transition-colors ${
                isDone ? "border-success-500" : inTarget ? "border-current" : "border-surface-400"
              }`}
              style={{
                left: `${target.position.x}%`,
                top: `${target.position.y}%`,
                width: spec.targetSize,
                height: spec.targetSize,
                backgroundColor: isDone ? "rgb(187 247 208 / 0.5)" : target.color,
                borderColor: isDone ? undefined : inTarget ? draggable?.color : undefined,
              }}
            >
              {/* Hold progress indicator - circular ring */}
              {isPending && (
                <svg
                  className="absolute inset-0 w-full h-full -rotate-90"
                  viewBox="0 0 100 100"
                >
                  <circle
                    cx="50"
                    cy="50"
                    r="46"
                    fill="none"
                    stroke={draggable?.color}
                    strokeWidth="4"
                    strokeDasharray={`${progress * 289} 289`}
                    strokeLinecap="round"
                  />
                </svg>
              )}
              {isDone && (
                <div className="absolute inset-0 flex items-center justify-center text-success-600 text-xl">
                  ✓
                </div>
              )}
            </div>
          );
        })}

        {/* Draggables */}
        {draggables.map((d) => {
          const pos = positions[d.id];
          if (!pos || d.done) return null;

          const isDragging = activeDraggable === d.id;
          const inTarget = isInTarget(d.id, pos.x, pos.y);

          return (
            <div
              key={d.id}
              className={`absolute rounded-full cursor-grab transition-shadow ${
                isDragging ? "cursor-grabbing shadow-lg" : "hover:shadow-md"
              }`}
              style={{
                left: pos.x,
                top: pos.y,
                width: spec.draggableSize,
                height: spec.draggableSize,
                backgroundColor: d.color,
                boxShadow: isDragging ? `0 8px 16px ${d.color}40` : undefined,
                transform: inTarget ? "scale(1.1)" : undefined,
              }}
              onPointerDown={(e) => handlePointerDown(e, d.id)}
            >
              {/* Inner circle */}
              <div className="absolute inset-2 rounded-full bg-white/30" />
            </div>
          );
        })}
      </div>
    </div>
  );
}
