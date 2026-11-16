"""Structural check of a LeRobot v2.1 dataset: do the files, the rows and the metadata agree with each other?"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import Dataset
from .core import SurgeryError

TIMESTAMP_TOLERANCE_S = 1e-4  # lerobot's default tolerance_s


def video_times(path: Path) -> np.ndarray:
    """Each frame's presentation time in seconds, sorted."""
    import av

    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        pts = [p.pts for p in box.demux(stream) if p.pts is not None]
        return np.sort(np.asarray(pts, dtype=np.float64)) * float(stream.time_base)


def check(root: Path | str, videos: bool = False) -> tuple[list[str], list[str]]:
    """(problems, warnings). Problems make the dataset wrong for any reader; warnings matter for some.

    ``videos``: also open every video and compare its frames with the rows (slow; needs PyAV).
    """
    problems: list[str] = []
    warnings: list[str] = []
    try:
        dataset = Dataset(root)
    except SurgeryError as error:
        return [str(error)], []
    info = dataset.info
    episodes = dataset.episodes
    numbers = [e["episode_index"] for e in episodes]
    if numbers != list(range(len(numbers))):
        problems.append("meta/episodes.jsonl: episode_index is not 0..N-1 without gaps")
    if info.get("total_episodes") != len(episodes):
        problems.append(f"info.json total_episodes {info.get('total_episodes')}, episodes.jsonl has {len(episodes)}")
    if info.get("total_tasks") != len(dataset.tasks):
        problems.append(f"info.json total_tasks {info.get('total_tasks')}, tasks.jsonl has {len(dataset.tasks)}")
    if info.get("total_videos") != len(episodes) * len(dataset.video_keys):
        problems.append(
            f"info.json total_videos {info.get('total_videos')}, expected {len(episodes) * len(dataset.video_keys)}"
        )
    without_stats = [n for n in numbers if n not in dataset.stats]
    if without_stats:
        problems.append(
            f"meta/episodes_stats.jsonl: no row for {len(without_stats)} episodes (first: {without_stats[0]})"
        )

    frames = 0
    off_grid: list[int] = []
    expected_files = set()
    wanted = ["episode_index", "index", "frame_index", "timestamp", "task_index"]
    for episode in episodes:
        number = episode["episode_index"]
        path = dataset.data_path(number)
        expected_files.add(path)
        if not path.exists():
            problems.append(f"missing {path.relative_to(dataset.root)}")
            continue
        table = dataset.table(number)
        missing_columns = [c for c in wanted if c not in table.column_names]
        if missing_columns:
            problems.append(f"episode {number}: no column {', '.join(missing_columns)}")
            continue
        missing_features = [k for k, v in dataset.features.items()
                            if v.get("dtype") != "video" and k not in table.column_names]  # fmt: skip
        if missing_features:
            problems.append(f"episode {number}: features without a column: {', '.join(missing_features)}")
        n = table.num_rows
        if n != episode.get("length"):
            problems.append(f"episode {number}: {n} rows, episodes.jsonl says {episode.get('length')}")
        if n == 0:
            continue
        column = {c: table.column(c).to_numpy() for c in wanted}
        if not (column["episode_index"] == number).all():
            problems.append(f"episode {number}: its rows carry another episode_index")
        if not np.array_equal(column["index"], np.arange(frames, frames + n)):
            problems.append(f"episode {number}: index is not {frames}..{frames + n - 1}")
        unknown = set(np.unique(column["task_index"]).tolist()) - set(dataset.tasks)
        if unknown:
            problems.append(f"episode {number}: task_index {sorted(unknown)} not in tasks.jsonl")
        elif episode.get("tasks") and dataset.tasks[int(column["task_index"][0])] not in episode["tasks"]:
            problems.append(f"episode {number}: its rows' task is not the one in episodes.jsonl")
        stamps = column["timestamp"].astype(np.float64)
        if n > 1 and float(np.abs(np.diff(stamps) - 1.0 / dataset.fps).max()) > TIMESTAMP_TOLERANCE_S:
            off_grid.append(number)
        if float(np.abs(stamps - column["frame_index"] / dataset.fps).max()) > TIMESTAMP_TOLERANCE_S:
            problems.append(f"episode {number}: timestamp is not frame_index / fps")
        for key in dataset.video_keys:
            video = dataset.video_path(number, key)
            expected_files.add(video)
            if not video.exists():
                problems.append(f"missing {video.relative_to(dataset.root)}")
            elif videos:
                times = video_times(video)
                if not len(times):
                    problems.append(f"{video.relative_to(dataset.root)}: no frames")
                elif float(stamps.max()) > float(times.max()) + TIMESTAMP_TOLERANCE_S:
                    problems.append(
                        f"{video.relative_to(dataset.root)}: rows go to {stamps.max():.3f} s, the video ends at {times.max():.3f} s"
                    )
                elif float(np.abs(times - np.round(times * dataset.fps) / dataset.fps).max()) > TIMESTAMP_TOLERANCE_S:
                    problems.append(
                        f"{video.relative_to(dataset.root)}: its frames are not on the {dataset.fps} fps grid"
                    )
        frames += n
    if info.get("total_frames") != frames:
        problems.append(f"info.json total_frames {info.get('total_frames')}, the rows add up to {frames}")
    if off_grid:
        warnings.append(
            f"{len(off_grid)} episodes have rows that are not 1/fps apart (first: episode {off_grid[0]}): rows were cut "
            "inside them, so lerobot's own loader refuses the dataset (its timestamp check); readers that take each "
            "row's timestamp as it is are fine"
        )
    stray = [p for pattern in ("data/**/*.parquet", "videos/**/*.mp4") for p in dataset.root.glob(pattern)
             if p not in expected_files]  # fmt: skip
    if stray:
        problems.append(
            f"{len(stray)} parquet/mp4 files belong to no episode (first: {stray[0].relative_to(dataset.root)}); lerobot "
            "counts every such file under the dataset"
        )
    return problems, warnings
