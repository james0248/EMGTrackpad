import { useCallback, useState } from "react";
import type { ClickButton, ClickGridCell, ClickGridSpec } from "../../types.ts";

interface ClickGridTaskProps {
  spec: ClickGridSpec;
  onComplete: () => void;
}

export function ClickGridTask({ spec, onComplete }: ClickGridTaskProps) {
  const [cells, setCells] = useState<ClickGridCell[]>(spec.cells);
  const [shakingCell, setShakingCell] = useState<string | null>(null);

  const handleCellClick = useCallback(
    (cellId: string, button: number, e: React.MouseEvent) => {
      e.preventDefault();

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

        // Check if all active cells are done
        const allDone = newCells.every((cell) => !cell.active || cell.done);
        if (allDone) {
          setTimeout(onComplete, 100);
        }

        return newCells;
      });
    },
    [onComplete]
  );

  const activeCells = cells.filter((c) => c.active);
  const doneCells = activeCells.filter((c) => c.done);

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Instructions */}
      <div className="text-center">
        <h2 className="text-xl font-display font-semibold text-surface-100 mb-2">
          Click Grid Task
        </h2>
        <p className="text-surface-400">
          Click all highlighted squares with the correct mouse button.
          <span className="ml-2 px-2 py-0.5 bg-cyan-900/50 text-cyan-400 rounded text-sm font-mono">
            L
          </span>
          {" = Left click, "}
          <span className="px-2 py-0.5 bg-rose-900/50 text-rose-400 rounded text-sm font-mono">
            R
          </span>
          {" = Right click"}
        </p>
        <p className="text-surface-500 text-sm mt-1">
          Progress: {doneCells.length} / {activeCells.length}
        </p>
      </div>

      {/* Grid */}
      <div
        className="grid gap-2 p-4 bg-surface-800 rounded-xl"
        style={{
          gridTemplateColumns: `repeat(${spec.gridSize}, 1fr)`,
        }}
      >
        {cells.map((cell) => (
          <GridCell
            key={cell.id}
            cell={cell}
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
  isShaking: boolean;
  onClick: (cellId: string, button: number, e: React.MouseEvent) => void;
}

function GridCell({ cell, isShaking, onClick }: GridCellProps) {
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 || e.button === 2) {
      onClick(cell.id, e.button, e);
    }
  };

  // Determine cell style
  let bgClass = "bg-surface-700 border-surface-600";
  let textClass = "text-surface-500";
  let cursor = "cursor-default";

  if (cell.active && !cell.done) {
    bgClass =
      cell.requiredButton === "left"
        ? "bg-cyan-900/60 border-cyan-500 hover:bg-cyan-800/80"
        : "bg-rose-900/60 border-rose-500 hover:bg-rose-800/80";
    textClass = cell.requiredButton === "left" ? "text-cyan-300" : "text-rose-300";
    cursor = "cursor-pointer";
  } else if (cell.done) {
    bgClass = "bg-success-500/30 border-success-500";
    textClass = "text-success-400";
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
      {cell.active && !cell.done && (
        <span className={textClass}>{cell.requiredButton === "left" ? "L" : "R"}</span>
      )}
      {cell.done && <span className={textClass}>✓</span>}
    </div>
  );
}
