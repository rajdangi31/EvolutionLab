"""Shared cultural memory for cumulative inheritance experiments."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CulturalMemoryEntry:
    """One structured piece of shared agent knowledge."""

    kind: str
    x: int
    y: int
    dx: float
    dy: float
    signal: float
    value: float
    generation: int
    uses: int = 0


class CulturalMemoryBank:
    """Bounded global repository shared across generations."""

    def __init__(self, max_entries: int, radius: float) -> None:
        self.max_entries = max_entries
        self.radius = radius
        self.entries: deque[CulturalMemoryEntry] = deque(maxlen=max_entries)
        self.read_count = 0
        self.write_count = 0
        self.transfer_events = 0

    def write_food(
        self,
        x: int,
        y: int,
        dx: float,
        dy: float,
        signal: float,
        value: float,
        generation: int,
    ) -> None:
        """Store a successful food-related observation."""

        self.entries.append(
            CulturalMemoryEntry(
                kind="food",
                x=x,
                y=y,
                dx=dx,
                dy=dy,
                signal=signal,
                value=value,
                generation=generation,
            )
        )
        self.write_count += 1

    def read(self, x: int, y: int, width: int, height: int) -> tuple[float, float, float]:
        """Return normalized direction and confidence from nearby cultural knowledge."""

        if not self.entries:
            return 0.0, 0.0, 0.0

        best_entry: CulturalMemoryEntry | None = None
        best_distance = float("inf")
        for entry in self.entries:
            distance = float(np.hypot(entry.x - x, entry.y - y))
            if distance < best_distance:
                best_entry = entry
                best_distance = distance

        if best_entry is None or best_distance > self.radius:
            return 0.0, 0.0, 0.0

        best_entry.uses += 1
        self.read_count += 1
        confidence = max(0.0, 1.0 - best_distance / max(1.0, self.radius))
        return (
            best_entry.dx / max(1, width),
            best_entry.dy / max(1, height),
            confidence,
        )

    def read_many(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized cultural-memory lookup for a whole population."""

        count = len(x_values)
        if not self.entries or count == 0:
            zeros = np.zeros(count, dtype=np.float32)
            return zeros, zeros.copy(), zeros.copy()

        entry_x = np.array([entry.x for entry in self.entries], dtype=np.float32)
        entry_y = np.array([entry.y for entry in self.entries], dtype=np.float32)
        entry_dx = np.array([entry.dx for entry in self.entries], dtype=np.float32)
        entry_dy = np.array([entry.dy for entry in self.entries], dtype=np.float32)

        dx = entry_x[None, :] - x_values[:, None].astype(np.float32)
        dy = entry_y[None, :] - y_values[:, None].astype(np.float32)
        distances = np.sqrt(dx * dx + dy * dy)
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(count), nearest_indices]
        valid = nearest_distances <= self.radius

        cultural_dx = np.zeros(count, dtype=np.float32)
        cultural_dy = np.zeros(count, dtype=np.float32)
        confidence = np.zeros(count, dtype=np.float32)
        cultural_dx[valid] = entry_dx[nearest_indices[valid]] / max(1, width)
        cultural_dy[valid] = entry_dy[nearest_indices[valid]] / max(1, height)
        confidence[valid] = 1.0 - nearest_distances[valid] / max(1.0, self.radius)

        if np.any(valid):
            self.read_count += int(np.sum(valid))
            used_indices, use_counts = np.unique(nearest_indices[valid], return_counts=True)
            entries = list(self.entries)
            for entry_index, use_count in zip(used_indices, use_counts):
                entries[int(entry_index)].uses += int(use_count)

        return cultural_dx, cultural_dy, confidence

    def mark_generation_transfer(self, population_size: int) -> None:
        """Record that a new generation inherited access to the shared bank."""

        if self.entries:
            self.transfer_events += population_size

    def snapshot_stats(self) -> dict[str, float]:
        """Return current memory-bank diagnostics."""

        total_uses = sum(entry.uses for entry in self.entries)
        mean_value = float(np.mean([entry.value for entry in self.entries])) if self.entries else 0.0
        return {
            "cultural_entries": float(len(self.entries)),
            "cultural_reads": float(self.read_count),
            "cultural_writes": float(self.write_count),
            "cultural_transfer_events": float(self.transfer_events),
            "cultural_entry_uses": float(total_uses),
            "cultural_mean_value": mean_value,
        }

    def save_csv(self, path: Path) -> None:
        """Persist the final memory bank in structured CSV form."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["kind", "x", "y", "dx", "dy", "signal", "value", "generation", "uses"])
            for entry in self.entries:
                writer.writerow(
                    [
                        entry.kind,
                        entry.x,
                        entry.y,
                        entry.dx,
                        entry.dy,
                        entry.signal,
                        entry.value,
                        entry.generation,
                        entry.uses,
                    ]
                )
