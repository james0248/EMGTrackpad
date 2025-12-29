import type { PRNG } from "../random/prng.ts";
import type {
  ClickGridCell,
  ClickGridSpec,
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

function generateClickGrid(prng: PRNG): ClickGridSpec {
  // Grid size between 4x4 and 6x6
  const gridSize = prng.nextInt(4, 6);
  const totalCells = gridSize * gridSize;

  // Active ratio between 20% and 50%
  const activeRatio = prng.nextFloat(0.2, 0.5);
  const activeCount = Math.max(2, Math.floor(totalCells * activeRatio));

  // Generate cells
  const cells: ClickGridCell[] = [];
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
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

  // Randomly select active cells
  const shuffled = prng.shuffle([...cells]);
  for (let i = 0; i < activeCount; i++) {
    const cell = shuffled[i]!;
    cell.active = true;
    // Random button requirement: ~60% left, ~40% right
    cell.requiredButton = prng.nextBool(0.6) ? "left" : "right";
  }

  return {
    kind: "click_grid",
    gridSize,
    cells,
  };
}

function generateDragToTarget(prng: PRNG): DragToTargetSpec {
  // Viewport-relative positions (will be scaled in component)
  const draggableSize = prng.nextInt(50, 80);
  const targetWidth = prng.nextInt(100, 160);
  const targetHeight = prng.nextInt(100, 160);

  // Ensure draggable and target don't overlap initially
  // Positions as percentages of container (10-90% range)
  const draggableX = prng.nextFloat(10, 40);
  const draggableY = prng.nextFloat(10, 90);

  const targetX = prng.nextFloat(55, 85);
  const targetY = prng.nextFloat(10, 90);

  return {
    kind: "drag_to_target",
    draggableStart: { x: draggableX, y: draggableY },
    targetPosition: { x: targetX, y: targetY },
    targetSize: { width: targetWidth, height: targetHeight },
    draggableSize,
  };
}

function generateScrollMeter(
  prng: PRNG,
  kind: "scroll_meter_horizontal" | "scroll_meter_vertical"
): ScrollMeterSpec {
  // Target band: random position with 10-20% width
  const bandWidth = prng.nextFloat(10, 20);
  const targetMin = prng.nextFloat(10, 90 - bandWidth);
  const targetMax = targetMin + bandWidth;

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
    sensitivity: 0.5, // Adjust based on testing
  };
}
