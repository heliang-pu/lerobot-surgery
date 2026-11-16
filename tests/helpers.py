"""A tiny synthetic LeRobot v2.1 dataset for the tests: real parquet files and real (32x24) videos."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_surgery.core import row_stats
from lerobot_surgery.core import write_jsonl

CAMERA = "observation.images.cam"
DIMS = 4


CODEC = None  # None: libx264 when PyAV has it (its dts starts below zero: frame reordering), else mpeg4


def write_video(path: Path, frames: int, fps: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    codec = CODEC or ("libx264" if "libx264" in av.codecs_available else "mpeg4")
    with av.open(str(path), "w") as box:
        stream = box.add_stream(codec, rate=Fraction(fps))
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for i in range(frames):
            picture = np.full((24, 32, 3), (i * 7) % 255, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format="rgb24")):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)


def moving_state(length: int, start: float = 0.0) -> np.ndarray:
    """Every row differs from the one before it by 0.01 in each dim."""
    return (start + 0.01 * np.arange(length)[:, None] + 0.1 * np.arange(DIMS)[None, :]).astype(np.float32)


def make_dataset(root: Path, episodes: list[tuple[str, int | np.ndarray]], fps: int = 10, video: bool = True,
                 stats_json: bool = False) -> Path:  # fmt: skip
    """``episodes``: (prompt, length or the (rows, DIMS) state array). The action is the next row's state."""
    root = Path(root)
    prompts = list(dict.fromkeys(p for p, _ in episodes))
    vector = pa.list_(pa.float32(), DIMS)
    features = {
        "observation.state": {"dtype": "float32", "shape": [DIMS], "names": [f"j{i}" for i in range(DIMS)]},
        "action": {"dtype": "float32", "shape": [DIMS], "names": [f"j{i}" for i in range(DIMS)]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if video:
        features[CAMERA] = {
            "dtype": "video", "shape": [24, 32, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": 24, "video.width": 32, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False},
        }  # fmt: skip
    info = {
        "codebase_version": "v2.1", "robot_type": "test", "total_episodes": len(episodes), "total_frames": 0,
        "total_tasks": len(prompts), "total_videos": len(episodes) * int(video), "total_chunks": 1, "chunks_size": 1000,
        "fps": fps, "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }  # fmt: skip
    index, episode_rows, stats_rows = 0, [], []
    for number, (prompt, content) in enumerate(episodes):
        state = moving_state(content, start=number) if isinstance(content, int) else np.asarray(content, np.float32)
        n = len(state)
        action = np.concatenate([state[1:], state[-1:]])
        table = pa.table({
            "observation.state": pa.array(list(state), type=vector),
            "action": pa.array(list(action), type=vector),
            "timestamp": pa.array(np.arange(n) / fps, type=pa.float32()),
            "frame_index": pa.array(np.arange(n), type=pa.int64()),
            "episode_index": pa.array(np.full(n, number), type=pa.int64()),
            "index": pa.array(np.arange(index, index + n), type=pa.int64()),
            "task_index": pa.array(np.full(n, prompts.index(prompt)), type=pa.int64()),
        })  # fmt: skip
        path = root / info["data_path"].format(episode_chunk=0, episode_index=number)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        stats = row_stats(table, features)
        if video:
            write_video(
                root / info["video_path"].format(episode_chunk=0, video_key=CAMERA, episode_index=number), n, fps
            )
            channel = [[[0.5]], [[0.5]], [[0.5]]]
            stats[CAMERA] = {"min": [[[0.0]]] * 3, "max": [[[1.0]]] * 3, "mean": channel, "std": channel, "count": [n]}
        episode_rows.append({"episode_index": number, "tasks": [prompt], "length": n})
        stats_rows.append({"episode_index": number, "stats": stats})
        index += n
    info["total_frames"] = index
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": p} for i, p in enumerate(prompts)])
    write_jsonl(meta / "episodes.jsonl", episode_rows)
    write_jsonl(meta / "episodes_stats.jsonl", stats_rows)
    if stats_json:
        (meta / "stats.json").write_text("{}\n")
    return root
