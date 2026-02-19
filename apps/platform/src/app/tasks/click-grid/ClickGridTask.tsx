import { useCallback, useEffect, useRef, useState } from "react";
import type { ClickButton, ClickGridCell, ClickGridSpec } from "../../types.ts";

interface ClickGridTaskProps {
  spec: ClickGridSpec;
  onComplete: () => void;
}

const PROGRESS_RING_CIRCUMFERENCE = 283;

export function ClickGridTask({ spec, onComplete }: ClickGridTaskProps) {
  const [cells, setCells] = useState<ClickGridCell[]>(spec.cells);
  const [completedCount, setCompletedCount] = useState(0);
  const [shakingCell, setShakingCell] = useState<string | null>(null);
  const [holdingCellId, setHoldingCellId] = useState<string | null>(null);
  const [holdProgress, setHoldProgress] = useState(0);

  const holdStartRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const shakeTimeoutRef = useRef<number | null>(null);
  const completedRef = useRef(false);

  // Current target cell ID based on activeOrder
  const currentTargetId = spec.activeOrder[completedCount] ?? null;

  const clearAnimationFrame = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const resetHold = useCallback(() => {
    holdStartRef.current = null;
    setHoldingCellId(null);
    setHoldProgress(0);
    clearAnimationFrame();
  }, [clearAnimationFrame]);

  const triggerShake = useCallback((cellId: string) => {
    setShakingCell(cellId);
    if (shakeTimeoutRef.current !== null) {
      clearTimeout(shakeTimeoutRef.current);
    }
    shakeTimeoutRef.current = window.setTimeout(() => {
      setShakingCell((prev) => (prev === cellId ? null : prev));
      shakeTimeoutRef.current = null;
    }, 300);
  }, []);

  const completeCell = useCallback(
    (cellId: string) => {
      setCells((prevCells) =>
        prevCells.map((cell) => {
          if (cell.id !== cellId || cell.done) return cell;
          return { ...cell, done: true };
        })
      );

      setCompletedCount((prevCount) => {
        const newCount = prevCount + 1;
        if (newCount >= spec.activeOrder.length && !completedRef.current) {
          completedRef.current = true;
          window.setTimeout(onComplete, 100);
        }
        return newCount;
      });
    },
    [onComplete, spec.activeOrder.length]
  );

  const startHold = useCallback(
    (cellId: string, holdDurationMs: number) => {
      holdStartRef.current = performance.now();
      setHoldingCellId(cellId);
      setHoldProgress(0);
      clearAnimationFrame();

      const animate = () => {
        if (holdStartRef.current === null || completedRef.current) return;

        const elapsed = performance.now() - holdStartRef.current;
        const progress = Math.min(1, elapsed / holdDurationMs);

        if (progress >= 1) {
          resetHold();
          completeCell(cellId);
          return;
        }

        setHoldProgress(progress);
        animationFrameRef.current = requestAnimationFrame(animate);
      };

      animationFrameRef.current = requestAnimationFrame(animate);
    },
    [clearAnimationFrame, completeCell, resetHold]
  );

  const handlePressStart = useCallback(
    (cellId: string, button: number, e: React.MouseEvent) => {
      e.preventDefault();

      if (completedRef.current || cellId !== currentTargetId) return;
      if (button !== 0 && button !== 2) return;

      const targetCell = cells.find((cell) => cell.id === cellId);
      if (
        !targetCell ||
        !targetCell.active ||
        targetCell.done ||
        targetCell.holdDurationMs === null
      ) {
        return;
      }

      // Only allow holding with the required button
      const clickButton: ClickButton = button === 0 ? "left" : "right";
      if (targetCell.requiredButton !== clickButton) {
        triggerShake(cellId);
        return;
      }

      startHold(cellId, targetCell.holdDurationMs);
    },
    [cells, currentTargetId, startHold, triggerShake]
  );

  const handlePressEnd = useCallback(() => {
    if (completedRef.current || holdingCellId === null) return;
    resetHold();
  }, [holdingCellId, resetHold]);

  useEffect(() => {
    return () => {
      clearAnimationFrame();
      if (shakeTimeoutRef.current !== null) {
        clearTimeout(shakeTimeoutRef.current);
      }
    };
  }, [clearAnimationFrame]);

  useEffect(() => {
    if (holdingCellId === null) return;

    const handleWindowMouseUp = () => {
      if (!completedRef.current) {
        resetHold();
      }
    };

    window.addEventListener("mouseup", handleWindowMouseUp);
    return () => window.removeEventListener("mouseup", handleWindowMouseUp);
  }, [holdingCellId, resetHold]);

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-4xl px-4">
      {/* Instructions */}
      <div className="text-center">
        <h2 className="text-xl font-display font-semibold text-surface-900 mb-2">
          Click Grid Task
        </h2>
        <p className="text-surface-600">
          Click and hold the highlighted square with the correct mouse button.
          <span className="ml-2 px-2 py-0.5 bg-cyan-100 text-cyan-700 rounded text-sm font-mono">
            L
          </span>
          {" = Left click, "}
          <span className="px-2 py-0.5 bg-rose-100 text-rose-700 rounded text-sm font-mono">R</span>
          {" = Right click"}
        </p>
        <p className="text-surface-500 text-sm mt-1">
          Progress: {completedCount} / {spec.activeOrder.length}
        </p>
      </div>

      {/* Grid */}
      <div className="w-full mx-auto" style={{ maxWidth: "min(90vw, 70vh)" }}>
        <div
          className="grid w-full aspect-square gap-2 p-4 bg-surface-200 rounded-xl"
          style={{
            gridTemplateColumns: `repeat(${spec.cols}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${spec.rows}, minmax(0, 1fr))`,
          }}
        >
          {cells.map((cell) => (
            <GridCell
              key={cell.id}
              cell={cell}
              isCurrent={cell.id === currentTargetId}
              isShaking={shakingCell === cell.id}
              holdProgress={holdingCellId === cell.id ? holdProgress : 0}
              onPressStart={handlePressStart}
              onPressEnd={handlePressEnd}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

interface GridCellProps {
  cell: ClickGridCell;
  isCurrent: boolean;
  isShaking: boolean;
  holdProgress: number;
  onPressStart: (cellId: string, button: number, e: React.MouseEvent) => void;
  onPressEnd: () => void;
}

function GridCell({
  cell,
  isCurrent,
  isShaking,
  holdProgress,
  onPressStart,
  onPressEnd,
}: GridCellProps) {
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 || e.button === 2) {
      onPressStart(cell.id, e.button, e);
    }
  };
  const isHolding = holdProgress > 0;

  // Determine cell style - only show current target as active
  let bgClass = "bg-surface-100 border-surface-300";
  let textClass = "text-surface-400";
  let cursor = "cursor-default";

  if (isCurrent && !cell.done) {
    bgClass =
      cell.requiredButton === "left"
        ? "bg-cyan-100 border-cyan-500 hover:bg-cyan-200"
        : "bg-rose-100 border-rose-500 hover:bg-rose-200";
    textClass = cell.requiredButton === "left" ? "text-cyan-700" : "text-rose-700";
    cursor = "cursor-pointer";
  } else if (cell.done) {
    bgClass = "bg-success-100 border-success-500";
    textClass = "text-success-600";
  }

  return (
    <div
      role="button"
      tabIndex={0}
      className={`
        relative w-full h-full min-w-0 min-h-0 flex items-center justify-center overflow-hidden
        rounded-lg border-2 transition-all duration-150
        select-none font-mono text-sm sm:text-base md:text-lg font-bold
        ${isHolding ? "scale-105" : "scale-100"}
        ${bgClass} ${cursor}
        ${isShaking ? "shake" : ""}
      `}
      onMouseDown={handleMouseDown}
      onMouseUp={onPressEnd}
      onMouseLeave={onPressEnd}
      onContextMenu={(e) => e.preventDefault()}
    >
      {isCurrent && !cell.done && holdProgress > 0 && (
        <svg
          className="absolute inset-0 w-full h-full -rotate-90 pointer-events-none"
          viewBox="0 0 100 100"
          aria-hidden="true"
        >
          <circle
            cx="50"
            cy="50"
            r="45"
            fill="none"
            stroke="white"
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${holdProgress * PROGRESS_RING_CIRCUMFERENCE} ${PROGRESS_RING_CIRCUMFERENCE}`}
          />
        </svg>
      )}
      {isCurrent && !cell.done && (
        <span className={`relative z-10 ${textClass}`}>
          {cell.requiredButton === "left" ? "L" : "R"}
        </span>
      )}
      {cell.done && <span className={`relative z-10 ${textClass}`}>✓</span>}
    </div>
  );
}
