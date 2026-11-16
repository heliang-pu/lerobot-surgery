"""What a dataset holds, and how several datasets differ: the look to take before mixing them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import IMAGE_DTYPES
from .core import Dataset
from .core import column_matrix


def vector_keys(dataset: Dataset) -> list[str]:
    """The float features with more than one dim (states, actions and the like)."""
    return [k for k, v in dataset.features.items()
            if v.get("dtype") not in IMAGE_DTYPES and str(v.get("dtype", "")).startswith("float")
            and (v.get("shape") or [1])[0] > 1]  # fmt: skip


def dim_names(feature: dict, dims: int) -> list[str]:
    names = feature.get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()), None)
    names = list(names) if isinstance(names, (list, tuple)) else []
    return [str(names[i]) if i < len(names) else str(i) for i in range(dims)]


def load(dataset: Dataset, keys: list[str], max_episodes: int | None) -> dict[str, np.ndarray]:
    episodes = [e["episode_index"] for e in dataset.episodes]
    if max_episodes and len(episodes) > max_episodes:  # spread over the dataset, not its first episodes
        episodes = [episodes[i] for i in np.linspace(0, len(episodes) - 1, max_episodes).round().astype(int)]
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for episode in episodes:
        table = dataset.table(episode, [k for k in keys])
        for key in keys:
            parts[key].append(column_matrix(table, key).astype(np.float64))
    return {k: np.concatenate(v) for k, v in parts.items() if v}


SCALE_RATIO = 15  # metres against a 0-1 fraction is 20x, degrees against radians 57x, millimetres against metres 1000x


def inspect(
    roots: list[Path], keys: list[str] | None = None, max_episodes: int | None = None
) -> tuple[list[str], list[str]]:
    """Print a report per dataset and return (warnings, notes). Warnings: differences that break mixing (frame rate,
    features). Notes: dims whose values are much larger in one dataset than in another, which is how another unit
    shows, but also a joint that one task hardly moves; look at them, they prove nothing."""
    warnings: list[str] = []
    notes: list[str] = []
    datasets = [Dataset(root) for root in roots]
    ranges: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for dataset in datasets:
        info = dataset.info
        lengths = np.asarray([e.get("length", 0) for e in dataset.episodes])
        print(f"\n=== {dataset.root}")
        print(f"robot {info.get('robot_type')} | fps {dataset.fps} | episodes {len(dataset.episodes)} | "
              f"frames {int(lengths.sum())} ({lengths.sum() / dataset.fps / 60:.1f} min)")  # fmt: skip
        if len(lengths):
            print(f"episode length: min {lengths.min()}, median {int(np.median(lengths))}, max {lengths.max()} rows")
        per_task: dict[str, list[int]] = {}
        for episode in dataset.episodes:
            for task in episode.get("tasks", ["?"]):
                per_task.setdefault(task, []).append(episode.get("length", 0))
        for task, rows in per_task.items():
            print(f"  task: {len(rows):5d} episodes {sum(rows):8d} frames  {task}")
        for name, feature in dataset.features.items():
            if feature.get("dtype") in IMAGE_DTYPES:
                codec = feature.get("info", {}).get("video.codec", "")
                print(f"  image {name}: {feature.get('shape')} {feature.get('dtype')} {codec}")
        own = [k for k in (keys or vector_keys(dataset)) if k in dataset.features]
        for key in keys or []:
            if key not in dataset.features:
                warnings.append(f"{dataset.root.name}: no feature {key!r}")
        for key, data in load(dataset, own, max_episodes).items():
            names = dim_names(dataset.features[key], data.shape[1])
            print(f"\n{key}: {data.shape[0]} rows x {data.shape[1]} dims")
            print(f"{'dim':>4} {'name':<24} {'min':>10} {'p1':>10} {'p50':>10} {'p99':>10} {'max':>10}")
            for dim in range(data.shape[1]):
                column = data[:, dim]
                p1, p50, p99 = np.percentile(column, [1, 50, 99])
                print(
                    f"{dim:>4} {names[dim]:<24} {column.min():>10.4f} {p1:>10.4f} {p50:>10.4f} {p99:>10.4f} {column.max():>10.4f}"
                )
                ranges.setdefault((key, dim), []).append((dataset.root.name, float(max(abs(p1), abs(p99)))))
    if len(datasets) > 1:
        first = datasets[0]
        for other in datasets[1:]:
            if other.fps != first.fps:
                warnings.append(
                    f"fps: {first.root.name} {first.fps}, {other.root.name} {other.fps}: retime or resample before mixing"
                )
            for key in sorted(set(first.features) | set(other.features)):
                a, b = first.features.get(key), other.features.get(key)
                if a is None or b is None:
                    warnings.append(f"feature {key}: only in {(first if b is None else other).root.name}")
                elif (a.get("dtype"), a.get("shape")) != (b.get("dtype"), b.get("shape")):
                    warnings.append(f"feature {key}: {a.get('dtype')}{a.get('shape')} in {first.root.name}, "
                                    f"{b.get('dtype')}{b.get('shape')} in {other.root.name}")  # fmt: skip
        for (key, dim), scales in ranges.items():
            if len(scales) < 2:
                continue
            low, high = min(scales, key=lambda s: s[1]), max(scales, key=lambda s: s[1])
            if high[1] > 1e-3 and high[1] > SCALE_RATIO * max(low[1], 1e-9):
                notes.append(f"{key}[{dim}]: values reach {high[1]:.4g} in {high[0]} but {low[1]:.4g} in {low[0]}: "
                             "another unit or convention?")  # fmt: skip
    return warnings, notes
