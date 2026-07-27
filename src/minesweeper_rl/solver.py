from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import comb, sqrt

import numpy as np

from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.types import SolverSnapshot


@dataclass(frozen=True)
class Constraint:
    cells: tuple[int, ...]
    required: int


@dataclass(frozen=True)
class ComponentEnumeration:
    cells: tuple[int, ...]
    mine_totals: dict[int, int]
    cell_mine_totals: dict[int, dict[int, int]]
    total_assignments: int


class MinesweeperSolver:
    """Visible-information solver used for forced moves and risk features."""

    def __init__(self, exact_limit: int = 24) -> None:
        self.exact_limit = exact_limit

    def analyze(self, game: MinesweeperGame) -> SolverSnapshot:
        hidden_mask = game.hidden_mask()
        forced_safe: set[int] = set()
        forced_mines: set[int] = set()
        virtual_flags = game.flagged.copy()

        changed = True
        while changed:
            changed = False
            constraints = self._build_constraints(game, virtual_flags)
            safe, mines = self._basic_deductions(constraints)
            safe -= forced_mines
            mines -= forced_safe

            new_mines = {idx for idx in mines if self._is_hidden_unflagged(game, idx, virtual_flags)}
            new_safe = {idx for idx in safe if self._is_hidden_unflagged(game, idx, virtual_flags)}
            if new_safe - forced_safe:
                forced_safe |= new_safe
            if new_mines - forced_mines:
                forced_mines |= new_mines
                for idx in new_mines:
                    row, col = divmod(idx, game.cols)
                    virtual_flags[row, col] = True
                changed = True

            exact_safe, exact_mines, _ = self._exact_component_probabilities(game, constraints, virtual_flags)
            exact_safe -= forced_mines
            exact_mines -= forced_safe
            exact_safe = {idx for idx in exact_safe if self._is_hidden_unflagged(game, idx, virtual_flags)}
            exact_mines = {idx for idx in exact_mines if self._is_hidden_unflagged(game, idx, virtual_flags)}
            if exact_safe - forced_safe:
                forced_safe |= exact_safe
            if exact_mines - forced_mines:
                forced_mines |= exact_mines
                for idx in exact_mines:
                    row, col = divmod(idx, game.cols)
                    virtual_flags[row, col] = True
                changed = True

        constraints = self._build_constraints(game, virtual_flags)
        safe_mask = np.zeros((game.rows, game.cols), dtype=bool)
        mine_mask = np.zeros((game.rows, game.cols), dtype=bool)
        for idx in forced_safe:
            row, col = divmod(idx, game.cols)
            if hidden_mask[row, col]:
                safe_mask[row, col] = True
        for idx in forced_mines:
            row, col = divmod(idx, game.cols)
            if hidden_mask[row, col]:
                mine_mask[row, col] = True

        frontier_mask, frontier_degree = self._frontier_maps(game, constraints)
        risk_map = self._risk_map(game, constraints, frontier_mask, virtual_flags)
        risk_map[safe_mask] = 0.0
        risk_map[mine_mask] = 1.0
        risk_map[game.revealed] = 0.0
        risk_map[game.flagged] = 1.0

        best_guess, best_risk = self._best_guess(game, risk_map, frontier_degree, safe_mask, mine_mask)
        best_guess_index = None if best_guess is None else best_guess[0] * game.cols + best_guess[1]

        return SolverSnapshot(
            hidden_mask=hidden_mask.astype(bool),
            frontier_mask=frontier_mask.astype(bool),
            safe_mask=safe_mask,
            mine_mask=mine_mask,
            risk_map=risk_map.astype(np.float32),
            frontier_degree_map=frontier_degree.astype(np.float32),
            best_guess=best_guess,
            best_guess_index=best_guess_index,
            best_guess_risk=best_risk,
        )

    def _build_constraints(self, game: MinesweeperGame, flags: np.ndarray) -> list[Constraint]:
        constraints: list[Constraint] = []
        for row, col in zip(*np.where(game.revealed)):
            clue = int(game.adjacent[row, col])
            if clue == 0:
                continue
            hidden_neighbors: list[int] = []
            flagged_neighbors = 0
            for nr, nc in game.neighbors(int(row), int(col)):
                if flags[nr, nc]:
                    flagged_neighbors += 1
                elif not game.revealed[nr, nc]:
                    hidden_neighbors.append(nr * game.cols + nc)

            required = clue - flagged_neighbors
            if hidden_neighbors and 0 <= required <= len(hidden_neighbors):
                constraints.append(Constraint(tuple(sorted(hidden_neighbors)), required))
        return self._dedupe_constraints(constraints)

    def _dedupe_constraints(self, constraints: list[Constraint]) -> list[Constraint]:
        seen: dict[tuple[int, ...], int] = {}
        for constraint in constraints:
            if constraint.cells not in seen:
                seen[constraint.cells] = constraint.required
            elif seen[constraint.cells] != constraint.required:
                # Inconsistent visible state. Keep the tighter requirement to avoid crashing.
                seen[constraint.cells] = min(seen[constraint.cells], constraint.required)
        return [Constraint(cells, required) for cells, required in seen.items()]

    def _basic_deductions(self, constraints: list[Constraint]) -> tuple[set[int], set[int]]:
        safe: set[int] = set()
        mines: set[int] = set()

        for constraint in constraints:
            cells = set(constraint.cells)
            if constraint.required == 0:
                safe |= cells
            elif constraint.required == len(cells):
                mines |= cells

        constraint_sets = [(set(c.cells), c.required) for c in constraints]
        for i, (cells_a, req_a) in enumerate(constraint_sets):
            if not cells_a:
                continue
            for cells_b, req_b in constraint_sets[i + 1 :]:
                if not cells_b or cells_a == cells_b:
                    continue
                if cells_a.issubset(cells_b):
                    diff = cells_b - cells_a
                    req_diff = req_b - req_a
                    if req_diff == 0:
                        safe |= diff
                    elif req_diff == len(diff):
                        mines |= diff
                elif cells_b.issubset(cells_a):
                    diff = cells_a - cells_b
                    req_diff = req_a - req_b
                    if req_diff == 0:
                        safe |= diff
                    elif req_diff == len(diff):
                        mines |= diff

        return safe, mines

    def _exact_component_probabilities(
        self,
        game: MinesweeperGame,
        constraints: list[Constraint],
        flags: np.ndarray | None = None,
    ) -> tuple[set[int], set[int], dict[int, float]]:
        if flags is None:
            flags = game.flagged

        probabilities: dict[int, float] = {}
        forced_safe: set[int] = set()
        forced_mines: set[int] = set()

        exact_components = self._exact_component_enumerations(constraints)
        if not exact_components:
            return forced_safe, forced_mines, probabilities

        component_probs, _, exact_cells = self._global_component_probabilities(game, exact_components, flags)
        probabilities.update(component_probs)
        for idx in exact_cells:
            prob = probabilities.get(idx)
            if prob is None:
                continue
            if prob <= 1e-9:
                forced_safe.add(idx)
            elif prob >= 1.0 - 1e-9:
                forced_mines.add(idx)

        return forced_safe, forced_mines, probabilities

    def _risk_map(
        self,
        game: MinesweeperGame,
        constraints: list[Constraint],
        frontier_mask: np.ndarray,
        virtual_flags: np.ndarray,
    ) -> np.ndarray:
        risk = np.zeros((game.rows, game.cols), dtype=np.float32)
        hidden = ~(game.revealed | virtual_flags)
        hidden_count = int(hidden.sum())
        remaining = max(0, game.mine_count - int(virtual_flags.sum()))
        base_prob = remaining / hidden_count if hidden_count else 0.0

        exact_components = self._exact_component_enumerations(constraints)
        exact_probs, outside_prob, _ = self._global_component_probabilities(game, exact_components, virtual_flags)
        if outside_prob is None:
            outside_prob = base_prob

        frontier_cells = set(idx for constraint in constraints for idx in constraint.cells)
        component_probs = dict(exact_probs)
        constraint_lookup = self._constraints_by_cell(constraints)

        for idx in frontier_cells:
            if idx not in component_probs:
                component_probs[idx] = self._heuristic_probability(idx, constraint_lookup, outside_prob)

        outside_mask = hidden & ~frontier_mask
        if outside_mask.any():
            risk[outside_mask] = float(np.clip(outside_prob, 0.0, 1.0))

        for idx, prob in component_probs.items():
            row, col = divmod(idx, game.cols)
            if hidden[row, col]:
                risk[row, col] = float(np.clip(prob, 0.0, 1.0))

        risk[virtual_flags] = 1.0
        risk[game.revealed] = 0.0
        return risk

    def _exact_component_enumerations(self, constraints: list[Constraint]) -> list[ComponentEnumeration]:
        enumerations: list[ComponentEnumeration] = []
        for cells, component_constraints in self._components(constraints):
            enumeration = self._enumerate_component_counts(cells, component_constraints)
            if enumeration is not None:
                enumerations.append(enumeration)
        return enumerations

    def _global_component_probabilities(
        self,
        game: MinesweeperGame,
        component_enums: list[ComponentEnumeration],
        flags: np.ndarray,
    ) -> tuple[dict[int, float], float | None, set[int]]:
        hidden = ~(game.revealed | flags)
        hidden_count = int(hidden.sum())
        remaining = max(0, game.mine_count - int(flags.sum()))
        base_prob = remaining / hidden_count if hidden_count else 0.0

        if not component_enums:
            return {}, float(base_prob), set()

        exact_cells: set[int] = set()
        for enumeration in component_enums:
            exact_cells.update(enumeration.cells)

        free_count = max(0, hidden_count - len(exact_cells))
        prefix = self._convolve_component_distributions(component_enums)
        suffix = self._convolve_component_distributions(list(reversed(component_enums)))
        total_distribution = prefix[-1]
        if not total_distribution:
            return {}, float(base_prob), exact_cells

        comb_cache: dict[tuple[int, int], int] = {}

        def choose(n: int, k: int) -> int:
            if k < 0 or k > n:
                return 0
            key = (n, k)
            if key not in comb_cache:
                comb_cache[key] = comb(n, k)
            return comb_cache[key]

        total_ways = 0
        expected_free_mines = 0
        for exact_mines, ways in total_distribution.items():
            free_mines = remaining - exact_mines
            free_ways = choose(free_count, free_mines)
            if free_ways:
                weight = ways * free_ways
                total_ways += weight
                expected_free_mines += weight * free_mines

        if total_ways == 0:
            return {}, float(base_prob), exact_cells

        probabilities: dict[int, float] = {}
        component_count = len(component_enums)
        for index, enumeration in enumerate(component_enums):
            left_distribution = prefix[index]
            right_distribution = suffix[component_count - index - 1]
            for cell in enumeration.cells:
                cell_distribution = enumeration.cell_mine_totals.get(cell, {})
                numerator = 0
                for left_mines, left_ways in left_distribution.items():
                    for right_mines, right_ways in right_distribution.items():
                        base_exact_mines = left_mines + right_mines
                        for component_mines, cell_mine_ways in cell_distribution.items():
                            exact_mines = base_exact_mines + component_mines
                            free_mines = remaining - exact_mines
                            free_ways = choose(free_count, free_mines)
                            if free_ways:
                                numerator += left_ways * right_ways * cell_mine_ways * free_ways
                probabilities[cell] = numerator / total_ways

        outside_prob = expected_free_mines / (total_ways * free_count) if free_count else 0.0
        return probabilities, float(outside_prob), exact_cells

    def _convolve_component_distributions(
        self,
        component_enums: list[ComponentEnumeration],
    ) -> list[dict[int, int]]:
        distributions: list[dict[int, int]] = [{0: 1}]
        for enumeration in component_enums:
            prev = distributions[-1]
            next_dist: dict[int, int] = defaultdict(int)
            for mine_count_a, ways_a in prev.items():
                for mine_count_b, ways_b in enumeration.mine_totals.items():
                    next_dist[mine_count_a + mine_count_b] += ways_a * ways_b
            distributions.append(dict(next_dist))
        return distributions

    def _frontier_maps(
        self,
        game: MinesweeperGame,
        constraints: list[Constraint],
    ) -> tuple[np.ndarray, np.ndarray]:
        frontier = np.zeros((game.rows, game.cols), dtype=bool)
        degree = np.zeros((game.rows, game.cols), dtype=np.float32)
        for constraint in constraints:
            for idx in constraint.cells:
                row, col = divmod(idx, game.cols)
                frontier[row, col] = True
                degree[row, col] += 1.0
        degree /= 8.0
        return frontier, np.clip(degree, 0.0, 1.0)

    def _components(self, constraints: list[Constraint]) -> list[tuple[list[int], list[Constraint]]]:
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for constraint in constraints:
            cells = list(constraint.cells)
            for cell in cells:
                parent.setdefault(cell, cell)
            for cell in cells[1:]:
                union(cells[0], cell)

        grouped_cells: dict[int, list[int]] = defaultdict(list)
        for cell in parent:
            grouped_cells[find(cell)].append(cell)

        grouped_constraints: dict[int, list[Constraint]] = defaultdict(list)
        for constraint in constraints:
            if constraint.cells:
                grouped_constraints[find(constraint.cells[0])].append(constraint)

        return [(sorted(cells), grouped_constraints[root]) for root, cells in grouped_cells.items()]

    def _enumerate_component_counts(
        self,
        cells: list[int],
        constraints: list[Constraint],
    ) -> ComponentEnumeration | None:
        local_index = {cell: i for i, cell in enumerate(cells)}
        local_constraints: list[tuple[list[int], int]] = []
        for constraint in constraints:
            indices = [local_index[cell] for cell in constraint.cells if cell in local_index]
            if indices:
                local_constraints.append((indices, constraint.required))

        n = len(cells)
        if n == 0:
            return None

        cell_memberships: list[list[int]] = [[] for _ in range(n)]
        for ci, (indices, _) in enumerate(local_constraints):
            for idx in indices:
                cell_memberships[idx].append(ci)

        grouped_cells: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx, memberships in enumerate(cell_memberships):
            grouped_cells[tuple(memberships)].append(idx)

        area_cells = list(grouped_cells.values())
        area_memberships = [list(signature) for signature in grouped_cells.keys()]
        area_sizes = [len(area) for area in area_cells]
        area_count = len(area_cells)
        if area_count > self.exact_limit:
            return None

        order = sorted(range(area_count), key=lambda idx: (-len(area_memberships[idx]), area_sizes[idx]))
        area_assignment = [-1] * area_count
        assigned_sum = [0] * len(local_constraints)
        unassigned = [len(indices) for indices, _ in local_constraints]
        mine_totals: dict[int, int] = defaultdict(int)
        cell_mine_totals: dict[int, dict[int, int]] = {cell: defaultdict(int) for cell in cells}
        total = 0

        def dfs(pos: int, mine_count: int, ways_so_far: int) -> None:
            nonlocal total
            if pos == area_count:
                total += ways_so_far
                mine_totals[mine_count] += ways_so_far
                for area_idx, value in enumerate(area_assignment):
                    if value <= 0:
                        continue
                    per_cell_ways = ways_so_far * value // area_sizes[area_idx]
                    for local_cell in area_cells[area_idx]:
                        cell_mine_totals[cells[local_cell]][mine_count] += per_cell_ways
                return

            area_idx = order[pos]
            size = area_sizes[area_idx]
            affected = area_memberships[area_idx]
            low, high = 0, size
            for ci in affected:
                required = local_constraints[ci][1]
                low = max(low, required - (assigned_sum[ci] + unassigned[ci] - size))
                high = min(high, required - assigned_sum[ci])

            for value in range(max(0, low), min(size, high) + 1):
                area_assignment[area_idx] = value
                valid = True
                for ci in affected:
                    assigned_sum[ci] += value
                    unassigned[ci] -= size
                    required = local_constraints[ci][1]
                    if assigned_sum[ci] > required or assigned_sum[ci] + unassigned[ci] < required:
                        valid = False

                if valid:
                    dfs(pos + 1, mine_count + value, ways_so_far * comb(size, value))

                for ci in affected:
                    assigned_sum[ci] -= value
                    unassigned[ci] += size
                area_assignment[area_idx] = -1

        dfs(0, 0, 1)
        if total == 0:
            return None
        return ComponentEnumeration(
            cells=tuple(cells),
            mine_totals=dict(mine_totals),
            cell_mine_totals={cell: dict(counts) for cell, counts in cell_mine_totals.items()},
            total_assignments=total,
        )

    def _enumerate_component(self, cells: list[int], constraints: list[Constraint]) -> dict[int, float]:
        enumeration = self._enumerate_component_counts(cells, constraints)
        if enumeration is None:
            return {}
        return {
            cell: sum(counts.values()) / enumeration.total_assignments
            for cell, counts in enumeration.cell_mine_totals.items()
        }

    def _constraints_by_cell(self, constraints: list[Constraint]) -> dict[int, list[Constraint]]:
        by_cell: dict[int, list[Constraint]] = defaultdict(list)
        for constraint in constraints:
            for cell in constraint.cells:
                by_cell[cell].append(constraint)
        return by_cell

    def _heuristic_probability(
        self,
        idx: int,
        constraint_lookup: dict[int, list[Constraint]],
        base_prob: float,
    ) -> float:
        constraints = constraint_lookup.get(idx, [])
        if not constraints:
            return base_prob
        weighted_sum = 0.0
        weight_total = 0.0
        for constraint in constraints:
            if not constraint.cells:
                continue
            local_ratio = constraint.required / len(constraint.cells)
            weight = 1.0 / max(1, len(constraint.cells))
            weighted_sum += local_ratio * weight
            weight_total += weight
        local = weighted_sum / weight_total if weight_total else base_prob
        return 0.7 * local + 0.3 * base_prob

    def _best_guess(
        self,
        game: MinesweeperGame,
        risk_map: np.ndarray,
        frontier_degree: np.ndarray,
        safe_mask: np.ndarray,
        mine_mask: np.ndarray,
    ) -> tuple[tuple[int, int] | None, float | None]:
        hidden = game.hidden_mask() & ~safe_mask & ~mine_mask
        cells = list(zip(*np.where(hidden)))
        if not cells:
            return None, None

        center_r = (game.rows - 1) / 2.0
        center_c = (game.cols - 1) / 2.0
        diag = sqrt(center_r**2 + center_c**2) or 1.0

        def key(cell: tuple[int, int]) -> tuple[float, float, float, float]:
            row, col = cell
            center_dist = sqrt((row - center_r) ** 2 + (col - center_c) ** 2) / diag
            edge_bonus = 8 - sum(1 for _ in game.neighbors(int(row), int(col)))
            return (
                float(risk_map[row, col]),
                -float(frontier_degree[row, col]),
                -float(edge_bonus),
                center_dist,
            )

        best = min(cells, key=key)
        return (int(best[0]), int(best[1])), float(risk_map[best])

    def _is_hidden_unflagged(self, game: MinesweeperGame, idx: int, flags: np.ndarray) -> bool:
        row, col = divmod(idx, game.cols)
        return bool(not game.revealed[row, col] and not flags[row, col])
