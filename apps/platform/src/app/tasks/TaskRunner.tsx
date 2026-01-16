import type { TaskSpec } from "../types.ts";
import { ClickGridTask } from "./click-grid/ClickGridTask.tsx";
import { ClickHoldTask } from "./click-hold/ClickHoldTask.tsx";
import { DragToTargetTask } from "./drag/DragToTargetTask.tsx";
import { ScrollMeterTask } from "./scroll/ScrollMeterTask.tsx";

interface TaskRunnerProps {
  task: TaskSpec;
  onComplete: (task: TaskSpec) => void;
}

export function TaskRunner({ task, onComplete }: TaskRunnerProps) {
  const handleComplete = () => {
    onComplete(task);
  };

  switch (task.kind) {
    case "click_grid":
      return <ClickGridTask spec={task} onComplete={handleComplete} />;
    case "click_hold":
      return <ClickHoldTask spec={task} onComplete={handleComplete} />;
    case "drag_to_target":
      return <DragToTargetTask spec={task} onComplete={handleComplete} />;
    case "scroll_meter_horizontal":
    case "scroll_meter_vertical":
      return <ScrollMeterTask spec={task} onComplete={handleComplete} />;
  }
}
