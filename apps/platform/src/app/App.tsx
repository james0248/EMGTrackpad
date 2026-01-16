import { useCallback, useEffect, useRef, useState } from "react";
import { TaskSelector } from "./config/TaskSelector.tsx";
import { generateSeed, PRNG } from "./random/prng.ts";
import { StatusBar } from "./status/StatusBar.tsx";
import { TaskRunner } from "./tasks/TaskRunner.tsx";
import { generateTask } from "./tasks/taskGenerator.ts";
import type { SessionData, TaskKind, TaskSpec } from "./types.ts";

const ALL_TASK_KINDS: TaskKind[] = [
  "click_grid",
  "click_hold",
  "drag_to_target",
  "scroll_meter_horizontal",
  "scroll_meter_vertical",
];

const INTER_TASK_DELAY_MS = 250;

function createInitialSession(seed: number, enabledTasks: TaskKind[]): SessionData {
  return {
    state: "idle",
    startTime: null,
    elapsedMs: 0,
    completedTasks: 0,
    taskCounts: {
      click_grid: 0,
      click_hold: 0,
      drag_to_target: 0,
      scroll_meter_horizontal: 0,
      scroll_meter_vertical: 0,
    },
    currentTask: null,
    seed,
    enabledTasks,
  };
}

export function App() {
  const [enabledTasks, setEnabledTasks] = useState<TaskKind[]>([...ALL_TASK_KINDS]);
  const [session, setSession] = useState<SessionData>(() =>
    createInitialSession(generateSeed(), ALL_TASK_KINDS)
  );
  const prngRef = useRef<PRNG | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastTaskKindsRef = useRef<string[]>([]);

  const handleToggleTask = useCallback((kind: TaskKind) => {
    setEnabledTasks((prev) =>
      prev.includes(kind) ? prev.filter((k) => k !== kind) : [...prev, kind]
    );
  }, []);

  // Update elapsed time every 100ms while running
  useEffect(() => {
    if (session.state === "running" && session.startTime !== null) {
      timerRef.current = setInterval(() => {
        setSession((prev) => ({
          ...prev,
          elapsedMs: performance.now() - (prev.startTime ?? 0),
        }));
      }, 100);
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [session.state, session.startTime]);

  const spawnNextTask = useCallback(() => {
    if (!prngRef.current) return;

    setSession((prev) => {
      const newTask = generateTask(prngRef.current!, lastTaskKindsRef.current, prev.enabledTasks);
      lastTaskKindsRef.current = [...lastTaskKindsRef.current.slice(-1), newTask.kind];
      return {
        ...prev,
        currentTask: newTask,
      };
    });
  }, []);

  const handleStart = useCallback(() => {
    if (enabledTasks.length === 0) return;

    const seed = generateSeed();
    prngRef.current = new PRNG(seed);
    lastTaskKindsRef.current = [];

    const startTime = performance.now();
    const firstTask = generateTask(prngRef.current, [], enabledTasks);
    lastTaskKindsRef.current = [firstTask.kind];

    setSession({
      state: "running",
      startTime,
      elapsedMs: 0,
      completedTasks: 0,
      taskCounts: {
        click_grid: 0,
        click_hold: 0,
        drag_to_target: 0,
        scroll_meter_horizontal: 0,
        scroll_meter_vertical: 0,
      },
      currentTask: firstTask,
      seed,
      enabledTasks,
    });
  }, [enabledTasks]);

  const handleEnd = useCallback(() => {
    setSession((prev) => ({
      ...prev,
      state: "ended",
      currentTask: null,
    }));
  }, []);

  const handleRestart = useCallback(() => {
    setEnabledTasks([...ALL_TASK_KINDS]);
    setSession(createInitialSession(generateSeed(), ALL_TASK_KINDS));
  }, []);

  // Prevent context menu globally during session
  useEffect(() => {
    if (session.state === "running") {
      const handler = (e: MouseEvent) => e.preventDefault();
      document.addEventListener("contextmenu", handler);
      document.body.classList.add("session-active");
      return () => {
        document.removeEventListener("contextmenu", handler);
        document.body.classList.remove("session-active");
      };
    }
  }, [session.state]);

  // Keyboard shortcut to end session (Escape)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && session.state === "running") {
        handleEnd();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [session.state, handleEnd]);

  const handleTaskComplete = useCallback(
    (task: TaskSpec) => {
      setSession((prev) => ({
        ...prev,
        completedTasks: prev.completedTasks + 1,
        taskCounts: {
          ...prev.taskCounts,
          [task.kind]: prev.taskCounts[task.kind] + 1,
        },
        currentTask: null,
      }));

      // Spawn next task after a short delay
      setTimeout(() => {
        if (session.state === "running") {
          spawnNextTask();
        }
      }, INTER_TASK_DELAY_MS);
    },
    [session.state, spawnNextTask]
  );

  return (
    <div className="flex flex-col h-full bg-surface-50">
      <StatusBar
        session={session}
        onStart={handleStart}
        onEnd={handleEnd}
        onRestart={handleRestart}
      />

      <main className="flex-1 flex items-center justify-center overflow-hidden">
        {session.state === "idle" && (
          <div className="flex flex-col items-center">
            <h1 className="text-4xl font-display font-bold text-surface-900 mb-2">
              EMG Task Playground
            </h1>
            <p className="text-surface-600 mb-6 max-w-md text-center">
              Configure which tasks to include in your session, then press Start.
            </p>
            <TaskSelector enabledTasks={enabledTasks} onToggle={handleToggleTask} />
            <button
              type="button"
              onClick={handleStart}
              disabled={enabledTasks.length === 0}
              className={`mt-6 px-8 py-3 rounded-lg font-semibold transition-colors ${
                enabledTasks.length > 0
                  ? "bg-accent-500 text-white hover:bg-accent-600"
                  : "bg-surface-200 text-surface-400 cursor-not-allowed"
              }`}
            >
              Start Session
            </button>
            {enabledTasks.length === 0 && (
              <p className="mt-2 text-sm text-danger-500">Select at least one task to start</p>
            )}
          </div>
        )}

        {session.state === "running" && session.currentTask && (
          <TaskRunner task={session.currentTask} onComplete={handleTaskComplete} />
        )}

        {session.state === "running" && !session.currentTask && (
          <div className="text-surface-500 font-mono text-lg">Loading next task...</div>
        )}

        {session.state === "ended" && (
          <div className="text-center">
            <h2 className="text-3xl font-display font-bold text-surface-900 mb-4">
              Session Complete
            </h2>
            <div className="bg-surface-100 border border-surface-200 rounded-xl p-6 mb-6 inline-block text-left">
              <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-surface-600">
                <span>Total Time:</span>
                <span className="font-mono text-surface-900">{formatTime(session.elapsedMs)}</span>
                <span>Tasks Completed:</span>
                <span className="font-mono text-surface-900">{session.completedTasks}</span>
                <span className="col-span-2 border-t border-surface-200 my-2"></span>
                <span>Click Grid:</span>
                <span className="font-mono text-surface-900">{session.taskCounts.click_grid}</span>
                <span>Click & Hold:</span>
                <span className="font-mono text-surface-900">{session.taskCounts.click_hold}</span>
                <span>Drag to Target:</span>
                <span className="font-mono text-surface-900">
                  {session.taskCounts.drag_to_target}
                </span>
                <span>Horizontal Scroll:</span>
                <span className="font-mono text-surface-900">
                  {session.taskCounts.scroll_meter_horizontal}
                </span>
                <span>Vertical Scroll:</span>
                <span className="font-mono text-surface-900">
                  {session.taskCounts.scroll_meter_vertical}
                </span>
              </div>
            </div>
            <p className="text-surface-500 text-sm">
              Press{" "}
              <kbd className="px-2 py-1 bg-surface-200 rounded text-xs font-mono">Restart</kbd> to
              begin a new session.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}
