from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Resolution:
    height: int
    width: int

    @property
    def ratio(self) -> float:
        return self.height / self.width


class ResolutionGroup:
    def __init__(self, base_size: int, step: int | None = None, align: int = 1):
        if base_size is None or not isinstance(base_size, int):
            raise ValueError(f"base_size must be int, got {base_size!r}")
        if base_size % align != 0:
            raise ValueError(f"base_size {base_size} is not divisible by align {align}")

        if step is None:
            step = base_size // 16
        if step > base_size // 2:
            raise ValueError(f"step must be <= base_size // 2, got {step}")

        self.base_size = base_size
        self.step = step
        self.align = align
        self.data = self._build_resolutions()
        self.ratio = np.array([item.ratio for item in self.data])

    def _aligned(self, value: int) -> int:
        return value // self.align * self.align

    def _build_resolutions(self):
        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break
            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(self._aligned(cur_height), self._aligned(cur_width)))

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break
            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(self._aligned(cur_height), self._aligned(cur_width)))

        return sorted(resolutions, key=lambda item: item.ratio)

    def get_target_size(self, width: int, height: int):
        ratio = height / width
        idx = int(np.argmin(np.abs(self.ratio - ratio)))
        target = self.data[idx]
        return target.width, target.height
