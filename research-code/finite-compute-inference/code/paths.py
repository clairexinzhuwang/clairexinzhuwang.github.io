"""Where the package keeps its inputs and outputs.

Defined once here so that every script agrees, and so that the layout can be
read off a single file rather than inferred from a dozen relative paths.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

RESULTS = os.path.join(ROOT, "results")        # grid_*.json, timing_pass.json
FIGURES = os.path.join(ROOT, "figures")        # generated tables and figures
REAL_DATA = os.path.join(ROOT, "real_data")
REAL_RESULTS = os.path.join(REAL_DATA, "results")


def grid(kernel, tag=""):
    """Result file for one kernel.  `tag` names the (n, B) grid: the reported
    grid writes grid_<kernel>.json and any other grid writes its own file, so
    two grids can never accumulate in one place and be tabulated together."""
    suffix = f"_{tag}" if tag else ""
    return os.path.join(RESULTS, f"grid_{kernel}{suffix}.json")


def ensure():
    for d in (RESULTS, FIGURES, REAL_RESULTS):
        os.makedirs(d, exist_ok=True)
