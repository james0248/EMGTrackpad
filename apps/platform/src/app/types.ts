// Task types
export type TaskKind =
  | "click_grid"
  | "drag_to_target"
  | "scroll_meter_horizontal"
  | "scroll_meter_vertical";

// Click grid types
export type ClickButton = "left" | "right";

export interface ClickGridCell {
  id: string;
  row: number;
  col: number;
  active: boolean;
  requiredButton: ClickButton | null;
  done: boolean;
}

export interface ClickGridSpec {
  kind: "click_grid";
  gridSize: number;
  cells: ClickGridCell[];
}

// Drag task types
export interface DragToTargetSpec {
  kind: "drag_to_target";
  draggableStart: { x: number; y: number };
  targetPosition: { x: number; y: number };
  targetSize: { width: number; height: number };
  draggableSize: number;
}

// Scroll task types
export interface ScrollMeterSpec {
  kind: "scroll_meter_horizontal" | "scroll_meter_vertical";
  initialValue: number;
  targetMin: number;
  targetMax: number;
  sensitivity: number;
}

export type TaskSpec = ClickGridSpec | DragToTargetSpec | ScrollMeterSpec;

// Session state
export type SessionState = "idle" | "running" | "ended";

export interface TaskCounts {
  click_grid: number;
  drag_to_target: number;
  scroll_meter_horizontal: number;
  scroll_meter_vertical: number;
}

export interface SessionData {
  state: SessionState;
  startTime: number | null;
  elapsedMs: number;
  completedTasks: number;
  taskCounts: TaskCounts;
  currentTask: TaskSpec | null;
  seed: number;
}

// Hold state for drag/scroll tasks
export type HoldState = "out_of_range" | "in_range_pending" | "completed";

export interface HoldProgress {
  state: HoldState;
  startTime: number | null;
  progress: number; // 0-1
}
