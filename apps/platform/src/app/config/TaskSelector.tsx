import type { TaskKind } from "../types.ts";

interface TaskOption {
  kind: TaskKind;
  label: string;
  description: string;
}

const TASK_OPTIONS: TaskOption[] = [
  {
    kind: "click_grid",
    label: "Click Grid",
    description: "Click highlighted cells with left or right mouse button",
  },
  {
    kind: "click_hold",
    label: "Click & Hold",
    description: "Press and hold the correct mouse button on a target",
  },
  {
    kind: "drag_to_target",
    label: "Drag to Target",
    description: "Drag colored circles to matching targets and hold",
  },
  {
    kind: "scroll_meter_horizontal",
    label: "Horizontal Scroll",
    description: "Scroll horizontally to move indicator into target zone",
  },
  {
    kind: "scroll_meter_vertical",
    label: "Vertical Scroll",
    description: "Scroll vertically to move indicator into target zone",
  },
];

interface TaskSelectorProps {
  enabledTasks: TaskKind[];
  onToggle: (kind: TaskKind) => void;
}

export function TaskSelector({ enabledTasks, onToggle }: TaskSelectorProps) {
  return (
    <div className="w-full max-w-md">
      <h2 className="text-lg font-semibold text-surface-700 mb-3">Select Tasks</h2>
      <div className="space-y-2">
        {TASK_OPTIONS.map((option) => {
          const isEnabled = enabledTasks.includes(option.kind);
          return (
            <button
              key={option.kind}
              type="button"
              onClick={() => onToggle(option.kind)}
              className={`w-full text-left p-3 rounded-lg border transition-colors ${
                isEnabled
                  ? "bg-accent-50 border-accent-300 text-surface-900"
                  : "bg-surface-100 border-surface-200 text-surface-500"
              }`}
            >
              <div className="flex items-center gap-3">
                <div
                  className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                    isEnabled ? "bg-accent-500 border-accent-500" : "bg-white border-surface-300"
                  }`}
                >
                  {isEnabled && (
                    <svg
                      className="w-3 h-3 text-white"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                      aria-hidden="true"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  )}
                </div>
                <div className="flex-1">
                  <div className="font-medium">{option.label}</div>
                  <div className="text-sm text-surface-500">{option.description}</div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
