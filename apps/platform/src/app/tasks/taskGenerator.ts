import type { PRNG } from "../random/prng.ts";
import type {
  ClickGridCell,
  ClickGridSpec,
  Draggable,
  DragTarget,
  DragToTargetSpec,
  ScrollMeterSpec,
  TaskKind,
  TaskSpec,
} from "../types.ts";

const TASK_KINDS: TaskKind[] = [
  "click_grid",
  "drag_to_target",
  "scroll_meter_horizontal",
  "scroll_meter_vertical",
];

/**
 * Generate a random task, avoiding streaks (no more than 2 of same kind in a row)
 */
export function generateTask(prng: PRNG, lastKinds: string[]): TaskSpec {
  // Filter out kinds that would cause a streak of 3+
  let availableKinds = TASK_KINDS;
  if (lastKinds.length >= 2) {
    const last = lastKinds[lastKinds.length - 1];
    const secondLast = lastKinds[lastKinds.length - 2];
    if (last === secondLast) {
      availableKinds = TASK_KINDS.filter((k) => k !== last);
    }
  }

  const kind = prng.pick(availableKinds);

  switch (kind) {
    case "click_grid":
      return generateClickGrid(prng);
    case "drag_to_target":
      return generateDragToTarget(prng);
    case "scroll_meter_horizontal":
      return generateScrollMeter(prng, "scroll_meter_horizontal");
    case "scroll_meter_vertical":
      return generateScrollMeter(prng, "scroll_meter_vertical");
  }
}

// Fixed click grid constants
const CLICK_GRID_ROWS = 5;
const CLICK_GRID_COLS = 10;
const CLICK_GRID_LEFT_COUNT = 6;
const CLICK_GRID_RIGHT_COUNT = 2;

function generateClickGrid(prng: PRNG): ClickGridSpec {
  // Generate cells for rectangular grid (wider than tall)
  const cells: ClickGridCell[] = [];
  for (let row = 0; row < CLICK_GRID_ROWS; row++) {
    for (let col = 0; col < CLICK_GRID_COLS; col++) {
      cells.push({
        id: `r${row}c${col}`,
        row,
        col,
        active: false,
        requiredButton: null,
        done: false,
      });
    }
  }

  // Randomly select active cells (fixed counts: 6 left, 2 right)
  const shuffled = prng.shuffle([...cells]);
  const activeCells: ClickGridCell[] = [];
  for (let i = 0; i < CLICK_GRID_LEFT_COUNT + CLICK_GRID_RIGHT_COUNT; i++) {
    const cell = shuffled[i]!;
    cell.active = true;
    cell.requiredButton = i < CLICK_GRID_LEFT_COUNT ? "left" : "right";
    activeCells.push(cell);
  }

  // Shuffle the active cells to randomize the order they appear
  const activeOrder = prng.shuffle(activeCells.map((c) => c.id));

  return {
    kind: "click_grid",
    rows: CLICK_GRID_ROWS,
    cols: CLICK_GRID_COLS,
    cells,
    activeOrder,
  };
}

// Fixed drag task constants
const DRAGGABLE_SIZE = 60;
const TARGET_SIZE = 80; // Circle slightly bigger than draggable
const DRAG_PAIR_COUNT = 3;

// Color pairs for draggables and targets
const DRAG_COLORS = [
  { draggable: "#f59e0b", target: "#fef3c7" }, // Amber
  { draggable: "#8b5cf6", target: "#ede9fe" }, // Violet
  { draggable: "#10b981", target: "#d1fae5" }, // Emerald
  { draggable: "#f43f5e", target: "#ffe4e6" }, // Rose
  { draggable: "#0ea5e9", target: "#e0f2fe" }, // Sky
];

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

function rectsOverlap(a: Rect, b: Rect, padding = 5): boolean {
  return !(
    a.x + a.width + padding < b.x ||
    b.x + b.width + padding < a.x ||
    a.y + a.height + padding < b.y ||
    b.y + b.height + padding < a.y
  );
}

function generateNonOverlappingPosition(
  prng: PRNG,
  width: number,
  height: number,
  existingRects: Rect[],
  maxAttempts = 100
): { x: number; y: number } | null {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const x = prng.nextFloat(10, 90 - width / 5); // Account for element size in %
    const y = prng.nextFloat(10, 90 - height / 5);

    const newRect: Rect = { x, y, width: width / 5, height: height / 5 };
    const overlaps = existingRects.some((r) => rectsOverlap(newRect, r));

    if (!overlaps) {
      return { x, y };
    }
  }
  return null;
}

function generateDragToTarget(prng: PRNG): DragToTargetSpec {
  const draggables: Draggable[] = [];
  const targets: DragTarget[] = [];
  const placedRects: Rect[] = [];

  // Shuffle colors for variety
  const shuffledColors = prng.shuffle([...DRAG_COLORS]);

  for (let i = 0; i < DRAG_PAIR_COUNT; i++) {
    const color = shuffledColors[i % shuffledColors.length]!;
    const targetId = `target-${i}`;
    const draggableId = `draggable-${i}`;

    // Place target first
    const targetPos = generateNonOverlappingPosition(
      prng,
      TARGET_SIZE,
      TARGET_SIZE,
      placedRects
    );
    if (targetPos) {
      targets.push({
        id: targetId,
        color: color.target,
        position: targetPos,
      });
      placedRects.push({
        x: targetPos.x,
        y: targetPos.y,
        width: TARGET_SIZE / 5,
        height: TARGET_SIZE / 5,
      });
    }

    // Place draggable
    const draggablePos = generateNonOverlappingPosition(
      prng,
      DRAGGABLE_SIZE,
      DRAGGABLE_SIZE,
      placedRects
    );
    if (draggablePos) {
      draggables.push({
        id: draggableId,
        color: color.draggable,
        position: draggablePos,
        targetId,
        done: false,
      });
      placedRects.push({
        x: draggablePos.x,
        y: draggablePos.y,
        width: DRAGGABLE_SIZE / 5,
        height: DRAGGABLE_SIZE / 5,
      });
    }
  }

  return {
    kind: "drag_to_target",
    draggables,
    targets,
    draggableSize: DRAGGABLE_SIZE,
    targetSize: TARGET_SIZE,
  };
}

// Fixed scroll task constants
const SCROLL_BAND_WIDTH = 8; // Fixed 8% band width

function generateScrollMeter(
  prng: PRNG,
  kind: "scroll_meter_horizontal" | "scroll_meter_vertical"
): ScrollMeterSpec {
  // Target band: random position with fixed width
  const targetMin = prng.nextFloat(15, 85 - SCROLL_BAND_WIDTH);
  const targetMax = targetMin + SCROLL_BAND_WIDTH;

  // Initial value: not in target band
  let initialValue: number;
  if (targetMin > 50) {
    initialValue = prng.nextFloat(5, targetMin - 10);
  } else if (targetMax < 50) {
    initialValue = prng.nextFloat(targetMax + 10, 95);
  } else {
    initialValue = prng.nextBool()
      ? prng.nextFloat(5, targetMin - 10)
      : prng.nextFloat(targetMax + 10, 95);
  }

  return {
    kind,
    initialValue,
    targetMin,
    targetMax,
    sensitivity: 0.5,
  };
}
