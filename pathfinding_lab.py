from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import argparse
import heapq
import math
import random
import statistics
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Iterator

Point = tuple[int, int]


class Heuristic(str, Enum):
    ZERO = "zero"
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    OCTILE = "octile"


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    blocked: frozenset[Point]

    @classmethod
    def random(
        cls,
        width: int,
        height: int,
        obstacle_count: int,
        start: Point,
        goal: Point,
        seed: int = 42,
    ) -> "GridMap":
        rng = random.Random(seed)
        blocked: set[Point] = set()
        while len(blocked) < obstacle_count:
            point = (rng.randrange(width), rng.randrange(height))
            if point not in (start, goal):
                blocked.add(point)
        return cls(width, height, frozenset(blocked))

    def in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, point: Point) -> bool:
        return point not in self.blocked

    def neighbors(
        self,
        point: Point,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
    ) -> Iterator[tuple[Point, float]]:
        x, y = point
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if allow_diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in directions:
            nxt = (x + dx, y + dy)
            if not self.in_bounds(nxt) or not self.passable(nxt):
                continue
            if dx and dy and prevent_corner_cutting:
                if not self.passable((x + dx, y)) or not self.passable((x, y + dy)):
                    continue
            yield nxt, math.sqrt(2) if dx and dy else 1.0

    def connected(self, start: Point, goal: Point, allow_diagonal: bool = True) -> bool:
        if not self.in_bounds(start) or not self.in_bounds(goal):
            return False
        if not self.passable(start) or not self.passable(goal):
            return False

        seen = {start}
        queue: deque[Point] = deque([start])
        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            for neighbor, _ in self.neighbors(current, allow_diagonal, prevent_corner_cutting=False):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return False

    def with_toggled_cell(self, point: Point) -> "GridMap":
        if not self.in_bounds(point):
            return self
        blocked = set(self.blocked)
        if point in blocked:
            blocked.remove(point)
        else:
            blocked.add(point)
        return GridMap(self.width, self.height, frozenset(blocked))

    def rows(self) -> Iterable[list[int]]:
        for y in range(self.height):
            yield [1 if (x, y) in self.blocked else 0 for x in range(self.width)]


@dataclass(frozen=True)
class AStarConfig:
    heuristic: Heuristic = Heuristic.OCTILE
    weight: float = 1.0
    tie_breaker_scale: bool = False
    tie_breaker_cross: bool = False
    allow_diagonal: bool = True
    prevent_corner_cutting: bool = True
    skip_stale_queue_entries: bool = True
    connectivity_precheck: bool = True

    @property
    def label(self) -> str:
        parts = [self.heuristic.value]
        if self.weight != 1.0:
            parts.append(f"w={self.weight:g}")
        if self.tie_breaker_scale:
            parts.append("scale-tie")
        if self.tie_breaker_cross:
            parts.append("cross-tie")
        if self.connectivity_precheck:
            parts.append("precheck")
        return " + ".join(parts)


@dataclass(frozen=True)
class SearchStep:
    current: Point
    frontier: frozenset[Point]
    reached: frozenset[Point]
    closed: frozenset[Point]


@dataclass(frozen=True)
class SearchResult:
    found: bool
    path: tuple[Point, ...]
    cost: float
    expanded: int
    pushed: int
    reopened: int
    elapsed_ms: float
    steps: tuple[SearchStep, ...] = field(default_factory=tuple)


class AStarPlanner:
    def __init__(self, grid: GridMap):
        self.grid = grid

    def search(
        self,
        start: Point,
        goal: Point,
        config: AStarConfig | None = None,
        record_steps: bool = False,
    ) -> SearchResult:
        config = config or AStarConfig()
        start_time = time.perf_counter()

        if config.connectivity_precheck and not self.grid.connected(start, goal, config.allow_diagonal):
            return SearchResult(False, (), math.inf, 0, 0, 0, self._elapsed(start_time))

        heuristic = self._heuristic(config.heuristic)
        open_heap: list[tuple[float, float, int, Point]] = []
        came_from: dict[Point, Point] = {}
        g_score = {start: 0.0}
        best_f = {start: config.weight * heuristic(start, goal)}
        closed: set[Point] = set()
        steps: list[SearchStep] = []
        counter = 0
        expanded = 0
        pushed = 1
        reopened = 0

        heapq.heappush(open_heap, (best_f[start], 0.0, counter, start))

        while open_heap:
            f_score, _, _, current = heapq.heappop(open_heap)
            if config.skip_stale_queue_entries and f_score != best_f.get(current):
                continue
            if current in closed:
                continue

            closed.add(current)
            expanded += 1

            if record_steps:
                steps.append(
                    SearchStep(
                        current=current,
                        frontier=frozenset(item[-1] for item in open_heap),
                        reached=frozenset(g_score),
                        closed=frozenset(closed),
                    )
                )

            if current == goal:
                path = tuple(self._reconstruct_path(came_from, current))
                return SearchResult(True, path, g_score[current], expanded, pushed, reopened, self._elapsed(start_time), tuple(steps))

            for neighbor, move_cost in self.grid.neighbors(
                current,
                allow_diagonal=config.allow_diagonal,
                prevent_corner_cutting=config.prevent_corner_cutting,
            ):
                tentative_g = g_score[current] + move_cost
                if tentative_g >= g_score.get(neighbor, math.inf):
                    continue

                if neighbor in closed:
                    closed.remove(neighbor)
                    reopened += 1

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                priority = tentative_g + config.weight * self._heuristic_cost(neighbor, goal, heuristic, config, start, current)
                best_f[neighbor] = priority
                counter += 1
                pushed += 1
                heapq.heappush(open_heap, (priority, -tentative_g, counter, neighbor))

        return SearchResult(False, (), math.inf, expanded, pushed, reopened, self._elapsed(start_time), tuple(steps))

    def _heuristic_cost(
        self,
        node: Point,
        goal: Point,
        heuristic: Callable[[Point, Point], float],
        config: AStarConfig,
        start: Point,
        current: Point,
    ) -> float:
        value = heuristic(node, goal)
        if config.tie_breaker_scale:
            value *= 1.0 + 1.0 / max(1, self.grid.width * self.grid.height)
        if config.tie_breaker_cross:
            dx1 = current[0] - goal[0]
            dy1 = current[1] - goal[1]
            dx2 = start[0] - goal[0]
            dy2 = start[1] - goal[1]
            value += abs(dx1 * dy2 - dx2 * dy1) * 0.001
        return value

    def _heuristic(self, heuristic: Heuristic) -> Callable[[Point, Point], float]:
        if heuristic == Heuristic.ZERO:
            return lambda _a, _b: 0.0
        if heuristic == Heuristic.MANHATTAN:
            return lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        if heuristic == Heuristic.EUCLIDEAN:
            return lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
        return self._octile

    @staticmethod
    def _octile(a: Point, b: Point) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)

    @staticmethod
    def _reconstruct_path(came_from: dict[Point, Point], current: Point) -> list[Point]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _elapsed(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000.0


@dataclass(frozen=True)
class Scenario:
    name: str
    config: AStarConfig


DEFAULT_SCENARIOS = (
    Scenario(
        "Plain A* baseline",
        AStarConfig(
            heuristic=Heuristic.EUCLIDEAN,
            connectivity_precheck=False,
            skip_stale_queue_entries=False,
        ),
    ),
    Scenario("Octile heuristic", AStarConfig(heuristic=Heuristic.OCTILE)),
    Scenario("Weighted A* w=1.5", AStarConfig(heuristic=Heuristic.OCTILE, weight=1.5)),
    Scenario("Weighted A* w=2.0", AStarConfig(heuristic=Heuristic.OCTILE, weight=2.0)),
    Scenario("Octile + scale tie", AStarConfig(heuristic=Heuristic.OCTILE, tie_breaker_scale=True)),
    Scenario("Octile + cross tie", AStarConfig(heuristic=Heuristic.OCTILE, tie_breaker_cross=True)),
    Scenario(
        "Full optimized",
        AStarConfig(heuristic=Heuristic.OCTILE, weight=1.2, tie_breaker_scale=True, tie_breaker_cross=True),
    ),
)


def make_demo_grid(width: int = 50, height: int = 50, seed: int = 42) -> tuple[GridMap, Point, Point]:
    start = (0, 0)
    goal = (width - 1, max(0, height // 5))
    obstacle_count = int(width * height * 0.12)
    return GridMap.random(width, height, obstacle_count, start, goal, seed), start, goal


def run_scenarios(grid: GridMap, start: Point, goal: Point, scenarios=DEFAULT_SCENARIOS) -> list[tuple[Scenario, SearchResult]]:
    planner = AStarPlanner(grid)
    return [(scenario, planner.search(start, goal, scenario.config)) for scenario in scenarios]


def summarize(results: list[tuple[Scenario, SearchResult]]) -> str:
    rows = [("scenario", "found", "cost", "expanded", "pushed", "reopened", "time_ms")]
    for scenario, result in results:
        cost = f"{result.cost:.3f}" if result.found else "inf"
        rows.append((scenario.name, "yes" if result.found else "no", cost, str(result.expanded), str(result.pushed), str(result.reopened), f"{result.elapsed_ms:.2f}"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def compare_to_baseline(results: list[tuple[Scenario, SearchResult]]) -> str:
    baseline = next((result for scenario, result in results if scenario.name == "Plain A* baseline"), None)
    if baseline is None or not baseline.found:
        return "No successful baseline run to compare against."

    lines = ["Efficiency compared with plain A* baseline:"]
    for scenario, result in results:
        if not result.found:
            lines.append(f"- {scenario.name}: no path")
            continue
        speedup = baseline.expanded / max(1, result.expanded)
        saved = baseline.expanded - result.expanded
        cost_penalty = result.cost - baseline.cost
        lines.append(f"- {scenario.name}: {speedup:.2f}x fewer expansions, {saved} nodes saved, cost +{cost_penalty:.3f}")
    return "\n".join(lines)


def aggregate_random_trials(trials: int = 20, width: int = 50, height: int = 50) -> str:
    buckets: dict[str, list[SearchResult]] = {scenario.name: [] for scenario in DEFAULT_SCENARIOS}
    for seed in range(trials):
        grid, start, goal = make_demo_grid(width, height, seed)
        for scenario, result in run_scenarios(grid, start, goal):
            buckets[scenario.name].append(result)

    lines = ["scenario,success_rate,median_expanded,median_cost,median_time_ms"]
    for name, results in buckets.items():
        found = [r for r in results if r.found]
        success_rate = len(found) / max(1, len(results))
        median_expanded = statistics.median(r.expanded for r in found) if found else 0
        median_cost = statistics.median(r.cost for r in found) if found else float("inf")
        median_time = statistics.median(r.elapsed_ms for r in found) if found else 0.0
        lines.append(f"{name},{success_rate:.2f},{median_expanded},{median_cost:.3f},{median_time:.2f}")
    return "\n".join(lines)


COLORS = {
    "empty": "#f8fafc",
    "wall": "#334155",
    "start": "#16a34a",
    "goal": "#dc2626",
    "frontier": "#93c5fd",
    "reached": "#bfdbfe",
    "closed": "#60a5fa",
    "path": "#f59e0b",
    "current": "#7c3aed",
    "grid": "#cbd5e1",
}


class PathfindingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("A* Pathfinding Lab")
        self.geometry("1180x780")

        self.grid_map, self.start, self.goal = make_demo_grid()
        self.cell_size = 13
        self.result: SearchResult | None = None
        self.step_index = 0
        self.after_id: str | None = None

        self.heuristic = tk.StringVar(value=Heuristic.OCTILE.value)
        self.weight = tk.DoubleVar(value=1.0)
        self.tie_scale = tk.BooleanVar(value=False)
        self.tie_cross = tk.BooleanVar(value=False)
        self.corner_cutting = tk.BooleanVar(value=False)

        self._build_layout()
        self._draw()

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        toolbar = ttk.Frame(root)
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="Heuristic").pack(side=tk.LEFT)
        ttk.Combobox(toolbar, textvariable=self.heuristic, values=[item.value for item in Heuristic], width=12, state="readonly").pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(toolbar, text="Weight").pack(side=tk.LEFT)
        ttk.Scale(toolbar, variable=self.weight, from_=1.0, to=3.0, orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="Scale tie", variable=self.tie_scale).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="Cross tie", variable=self.tie_cross).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="Allow corner cutting", variable=self.corner_cutting).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(toolbar, text="Run", command=self.run_search).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Play", command=self.play).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Benchmark", command=self.benchmark).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="New map", command=self.new_map).pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.canvas = tk.Canvas(body, width=650, height=650, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        self.canvas.bind("<Button-1>", self.toggle_wall)

        side = ttk.Frame(body, padding=(12, 0, 0, 0))
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(side, text="Metrics").pack(anchor=tk.W)
        self.metrics = tk.Text(side, height=14, wrap=tk.WORD)
        self.metrics.pack(fill=tk.X, pady=(4, 12))
        ttk.Label(side, text="Comparison").pack(anchor=tk.W)
        self.comparison = tk.Text(side, wrap=tk.NONE)
        self.comparison.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def config(self) -> AStarConfig:
        return AStarConfig(
            heuristic=Heuristic(self.heuristic.get()),
            weight=round(self.weight.get(), 2),
            tie_breaker_scale=self.tie_scale.get(),
            tie_breaker_cross=self.tie_cross.get(),
            prevent_corner_cutting=not self.corner_cutting.get(),
        )

    def run_search(self) -> None:
        self.stop_playback()
        self.result = AStarPlanner(self.grid_map).search(self.start, self.goal, self.config(), record_steps=True)
        self.step_index = max(0, len(self.result.steps) - 1)
        self._draw()

    def step_once(self) -> None:
        if self.result is None:
            self.result = AStarPlanner(self.grid_map).search(self.start, self.goal, self.config(), record_steps=True)
            self.step_index = 0
        else:
            self.step_index = min(self.step_index + 1, max(0, len(self.result.steps) - 1))
        self._draw()

    def play(self) -> None:
        if self.result is None:
            self.result = AStarPlanner(self.grid_map).search(self.start, self.goal, self.config(), record_steps=True)
            self.step_index = 0
        self.step_index = min(self.step_index + 1, max(0, len(self.result.steps) - 1))
        self._draw()
        if self.result and self.step_index < len(self.result.steps) - 1:
            self.after_id = self.after(25, self.play)

    def benchmark(self) -> None:
        self.comparison.delete("1.0", tk.END)
        self.comparison.insert(tk.END, summarize(run_scenarios(self.grid_map, self.start, self.goal)))

    def new_map(self) -> None:
        self.stop_playback()
        self.grid_map, self.start, self.goal = make_demo_grid(seed=random.randrange(1_000_000))
        self.result = None
        self.step_index = 0
        self._draw()

    def toggle_wall(self, event: tk.Event) -> None:
        point = (event.x // self.cell_size, event.y // self.cell_size)
        if point not in (self.start, self.goal):
            self.grid_map = self.grid_map.with_toggled_cell(point)
            self.result = None
            self._draw()

    def stop_playback(self) -> None:
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def _draw(self) -> None:
        self.canvas.delete("all")
        step = None
        if self.result and self.result.steps:
            step = self.result.steps[max(0, min(self.step_index, len(self.result.steps) - 1))]
        path = set(self.result.path if self.result and self.step_index >= len(self.result.steps) - 1 else ())

        for y in range(self.grid_map.height):
            for x in range(self.grid_map.width):
                point = (x, y)
                color = COLORS["empty"]
                if point in self.grid_map.blocked:
                    color = COLORS["wall"]
                elif step and point in step.reached:
                    color = COLORS["reached"]
                if step and point in step.frontier:
                    color = COLORS["frontier"]
                if step and point in step.closed:
                    color = COLORS["closed"]
                if point in path:
                    color = COLORS["path"]
                if point == self.start:
                    color = COLORS["start"]
                if point == self.goal:
                    color = COLORS["goal"]
                if step and point == step.current:
                    color = COLORS["current"]
                self._rect(point, color)
        self._update_metrics()

    def _rect(self, point: Point, color: str) -> None:
        x, y = point
        s = self.cell_size
        self.canvas.create_rectangle(x * s, y * s, (x + 1) * s, (y + 1) * s, fill=color, outline=COLORS["grid"])

    def _update_metrics(self) -> None:
        self.metrics.delete("1.0", tk.END)
        if self.result is None:
            self.metrics.insert(tk.END, "Click Run, Step, or Play to start.\nClick cells to toggle walls.")
            return
        result = self.result
        self.metrics.insert(
            tk.END,
            "\n".join(
                [
                    f"Config: {self.config().label}",
                    f"Found: {result.found}",
                    f"Cost: {result.cost:.3f}" if result.found else "Cost: inf",
                    f"Expanded: {result.expanded}",
                    f"Pushed: {result.pushed}",
                    f"Reopened: {result.reopened}",
                    f"Elapsed: {result.elapsed_ms:.2f} ms",
                    f"Step: {self.step_index + 1}/{max(1, len(result.steps))}",
                ]
            ),
        )


def run_benchmark() -> None:
    grid, start, goal = make_demo_grid()
    results = run_scenarios(grid, start, goal)
    print(summarize(results))
    print()
    print(compare_to_baseline(results))
    print()
    print(aggregate_random_trials())


def run_demo() -> None:
    PathfindingApp().mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-file A* pathfinding lab.")
    parser.add_argument("--mode", choices=("demo", "benchmark"), default="demo")
    args = parser.parse_args()
    if args.mode == "benchmark":
        run_benchmark()
    else:
        run_demo()


if __name__ == "__main__":
    main()
