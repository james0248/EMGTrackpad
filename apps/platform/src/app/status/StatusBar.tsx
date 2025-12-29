import type { SessionData } from "../types.ts";

interface StatusBarProps {
  session: SessionData;
  onStart: () => void;
  onEnd: () => void;
  onRestart: () => void;
}

export function StatusBar({ session, onStart, onEnd, onRestart }: StatusBarProps) {
  const { state, elapsedMs, completedTasks, taskCounts, seed } = session;

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-surface-800 border-b border-surface-700 shrink-0">
      {/* Left: Session controls */}
      <div className="flex items-center gap-4">
        {state === "idle" && (
          <button
            type="button"
            onClick={onStart}
            className="px-5 py-2 bg-accent-500 hover:bg-accent-400 text-surface-900 font-semibold rounded-lg transition-colors"
          >
            Start Session
          </button>
        )}

        {state === "running" && (
          <button
            type="button"
            onClick={onEnd}
            className="px-5 py-2 bg-danger-500 hover:bg-danger-400 text-white font-semibold rounded-lg transition-colors"
          >
            End Session
          </button>
        )}

        {state === "ended" && (
          <button
            type="button"
            onClick={onRestart}
            className="px-5 py-2 bg-accent-500 hover:bg-accent-400 text-surface-900 font-semibold rounded-lg transition-colors"
          >
            Restart
          </button>
        )}

        {/* Session state indicator */}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              state === "idle"
                ? "bg-surface-500"
                : state === "running"
                  ? "bg-success-400 animate-pulse"
                  : "bg-surface-400"
            }`}
          />
          <span className="text-surface-400 text-sm font-medium capitalize">{state}</span>
        </div>
      </div>

      {/* Center: Timer and counts */}
      <div className="flex items-center gap-8">
        {/* Elapsed time */}
        <div className="flex items-center gap-2">
          <span className="text-surface-500 text-sm">Time:</span>
          <span className="font-mono text-lg text-surface-100 tabular-nums">
            {formatTime(elapsedMs)}
          </span>
        </div>

        {/* Total completed */}
        <div className="flex items-center gap-2">
          <span className="text-surface-500 text-sm">Completed:</span>
          <span className="font-mono text-lg text-surface-100 tabular-nums">{completedTasks}</span>
        </div>

        {/* Task breakdown */}
        <div className="flex items-center gap-3 text-sm">
          <TaskCountBadge label="Click" count={taskCounts.click_grid} color="cyan" />
          <TaskCountBadge label="Drag" count={taskCounts.drag_to_target} color="amber" />
          <TaskCountBadge
            label="Scroll"
            count={taskCounts.scroll_meter_horizontal + taskCounts.scroll_meter_vertical}
            color="emerald"
          />
        </div>
      </div>

      {/* Right: Seed display */}
      <div className="flex items-center gap-2 text-surface-500 text-xs">
        <span>Seed:</span>
        <code className="font-mono bg-surface-700 px-2 py-1 rounded">{seed}</code>
      </div>
    </header>
  );
}

interface TaskCountBadgeProps {
  label: string;
  count: number;
  color: "cyan" | "amber" | "emerald";
}

function TaskCountBadge({ label, count, color }: TaskCountBadgeProps) {
  const colorClasses = {
    cyan: "bg-cyan-900/50 text-cyan-400 border-cyan-700",
    amber: "bg-amber-900/50 text-amber-400 border-amber-700",
    emerald: "bg-emerald-900/50 text-emerald-400 border-emerald-700",
  };

  return (
    <span className={`px-2 py-1 rounded border ${colorClasses[color]} font-mono`}>
      {label}: {count}
    </span>
  );
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}
