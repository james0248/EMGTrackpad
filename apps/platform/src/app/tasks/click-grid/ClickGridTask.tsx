import { useCallback, useState } from "react";
import type { ClickButton, ClickGridCell, ClickGridSpec } from "../../types.ts";

interface ClickGridTaskProps {
  spec: ClickGridSpec;
  onComplete: () => void;
}

export function ClickGridTask({ spec, onComplete }: ClickGridTaskProps) {
  const [cells, setCells] = useState<ClickGridCell[]>(spec.cells);
  const [completedCount, setCompletedCount] = useState(0);
  const [shakingCell, setShakingCell] = useState<string | null>(null);

  // Current target cell ID based on activeOrder
  const currentTargetId = spec.activeOrder[completedCount] ?? null;

  const handleCellClick = useCallback(
    (cellId: string, button: number, e: React.MouseEvent) => {
      e.preventDefault();

      // Only allow clicking the current target
      if (cellId !== currentTargetId) return;

      const clickButton: ClickButton = button === 0 ? "left" : "right";

      setCells((prevCells) => {
        const newCells = prevCells.map((cell) => {
          if (cell.id !== cellId) return cell;
          if (!cell.active || cell.done) return cell;

          if (cell.requiredButton === clickButton) {
            return { ...cell, done: true };
          } else {
            // Wrong button - trigger shake
            setShakingCell(cellId);
            setTimeout(() => setShakingCell(null), 300);
            return cell;
          }
        });

        // Check if the clicked cell was completed
        const clickedCell = newCells.find((c) => c.id === cellId);
        if (clickedCell?.done) {
          const newCount = completedCount + 1;
          setCompletedCount(newCount);

          // Check if all done
          if (newCount >= spec.activeOrder.length) {
            setTimeout(onComplete, 100);
          }
        }

        return newCells;
      });
    },
    [currentTargetId, completedCount, spec.activeOrder.length, onComplete]
  );

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Instructions */}
      <div className="text-center">
        <h2 className="text-xl font-display font-semibold text-surface-900 mb-2">
          Click Grid Task
        </h2>
        <p className="text-surface-600">
          Click the highlighted square with the correct mouse button.
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
      <div
        className="grid gap-2 p-4 bg-surface-200 rounded-xl"
        style={{
          gridTemplateColumns: `repeat(${spec.cols}, 1fr)`,
        }}
      >
        {cells.map((cell) => (
          <GridCell
            key={cell.id}
            cell={cell}
            isCurrent={cell.id === currentTargetId}
            isShaking={shakingCell === cell.id}
            onClick={handleCellClick}
          />
        ))}
      </div>
    </div>
  );
}

interface GridCellProps {
  cell: ClickGridCell;
  isCurrent: boolean;
  isShaking: boolean;
  onClick: (cellId: string, button: number, e: React.MouseEvent) => void;
}

function GridCell({ cell, isCurrent, isShaking, onClick }: GridCellProps) {
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 || e.button === 2) {
      onClick(cell.id, e.button, e);
    }
  };

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
        w-16 h-16 flex items-center justify-center
        rounded-lg border-2 transition-all duration-150
        select-none font-mono text-lg font-bold
        ${bgClass} ${cursor}
        ${isShaking ? "shake" : ""}
      `}
      onMouseDown={handleMouseDown}
      onContextMenu={(e) => e.preventDefault()}
    >
      {isCurrent && !cell.done && (
        <span className={textClass}>{cell.requiredButton === "left" ? "L" : "R"}</span>
      )}
      {cell.done && <span className={textClass}>✓</span>}
    </div>
  );
}
