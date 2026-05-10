from __future__ import annotations

from dataclasses import dataclass

from ltf.config import TaskConfig


@dataclass
class Curriculum:
    start: int
    end: int
    inc: int
    interval: int

    def __post_init__(self) -> None:
        self.n_points = self.start
        self.step_count = 0

    @classmethod
    def from_task(cls, task: TaskConfig) -> "Curriculum":
        return cls(
            start=task.train_length_start,
            end=task.train_length_end,
            inc=task.train_length_inc,
            interval=task.train_length_interval,
        )

    def update(self) -> None:
        self.step_count += 1
        if self.step_count % self.interval == 0:
            self.n_points += self.inc
        self.n_points = min(self.n_points, self.end)

