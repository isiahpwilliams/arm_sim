"""Task sampling and language instructions for cube stacking."""
from __future__ import annotations

import numpy as np

from arm_env import CUBE_NAMES

COLOUR = {"cube_red": "red", "cube_blue": "blue", "cube_green": "green"}

TEMPLATES = [
    "stack the {src} cube on the {dst} cube",
    "put the {src} cube on top of the {dst} cube",
    "place the {src} cube on the {dst} cube",
    "pick up the {src} cube and put it on the {dst} cube",
    "move the {src} block onto the {dst} block",
]


ONEHOT_NAMES = ([f"src_{COLOUR[c]}" for c in CUBE_NAMES]
                + [f"dst_{COLOUR[c]}" for c in CUBE_NAMES])


def task_onehot(src: str, dst: str) -> np.ndarray:
    """Which (source, destination) pair the episode is, as 6 binary dims.

    ACT takes no language input, so without this the six pairs in the dataset are
    six different behaviours for identical pixels and the best the policy can do
    is average them. Appended to observation.state by --task-cond.
    """
    v = np.zeros(2 * len(CUBE_NAMES), dtype=np.float32)
    v[CUBE_NAMES.index(src)] = 1.0
    v[len(CUBE_NAMES) + CUBE_NAMES.index(dst)] = 1.0
    return v


def sample_task(rng: np.random.Generator) -> tuple[str, str, str]:
    """Return (source cube name, target cube name, instruction text)."""
    src, dst = rng.choice(CUBE_NAMES, size=2, replace=False)
    template = TEMPLATES[rng.integers(len(TEMPLATES))]
    return str(src), str(dst), template.format(src=COLOUR[src], dst=COLOUR[dst])
