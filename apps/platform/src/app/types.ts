// Task types
export type TaskKind =
  | "click_grid"
  | "click_hold"
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
  rows: number;
  cols: number;
  cells: ClickGridCell[];
  activeOrder: string[]; // Cell IDs in the order they should appear
}

// Drag task types
export interface DragTarget {
  id: string;
  color: string;
  position: { x: number; y: number };
}

export interface Draggable {
  id: string;
  color: string;
  position: { x: number; y: number };
  targetId: string; // which target this draggable should go to
  done: boolean;
}

export interface DragToTargetSpec {
  kind: "drag_to_target";
  draggables: Draggable[];
  targets: DragTarget[];
  draggableSize: number;
  targetSize: number; // Circle diameter
}

// Scroll task types
export interface ScrollMeterSpec {
  kind: "scroll_meter_horizontal" | "scroll_meter_vertical";
  initialValue: number;
  targetMin: number;
  targetMax: number;
  sensitivity: number;
}

// Click hold task types
export interface ClickHoldSpec {
  kind: "click_hold";
  requiredButton: ClickButton;
  holdDurationMs: number;
}

export type TaskSpec = ClickGridSpec | ClickHoldSpec | DragToTargetSpec | ScrollMeterSpec;

// Session state
export type SessionState = "idle" | "running" | "ended";

export interface TaskCounts {
  click_grid: number;
  click_hold: number;
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
  enabledTasks: TaskKind[];
}

// Hold state for drag/scroll tasks
export type HoldState = "out_of_range" | "in_range_pending" | "completed";

export interface HoldProgress {
  state: HoldState;
  startTime: number | null;
  progress: number; // 0-1
}
