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


def sample_task(rng: np.random.Generator) -> tuple[str, str, str]:
    """Return (source cube name, target cube name, instruction text)."""
    src, dst = rng.choice(CUBE_NAMES, size=2, replace=False)
    template = TEMPLATES[rng.integers(len(TEMPLATES))]
    return str(src), str(dst), template.format(src=COLOUR[src], dst=COLOUR[dst])
