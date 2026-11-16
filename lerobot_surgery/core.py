"""Reading and writing LeRobot v2.1 datasets with numpy and pyarrow only (lerobot itself is not needed).

Every operation that changes which episodes or rows a dataset holds goes through ``write_dataset``: a list of
``Piece`` (one output episode each) becomes a new dataset with episodes renumbered from 0, a consecutive global
``index``, hard-linked videos, per-episode stats, and two provenance files:

  meta/surgery.json            what was done, with which arguments, from which sources (their own history kept)
  meta/surgery_episodes.jsonl  each output episode's source dataset and episode
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VERSION = "0.1.0"
STANDARD_META = ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "stats.json")
OWN_META = ("surgery.json", "surgery_episodes.jsonl")
INDEX_COLUMNS = ("episode_index", "index", "task_index")
IMAGE_DTYPES = ("video", "image")


class SurgeryError(Exception):
    """A problem with the input or the request; the CLI prints it without a traceback."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def link(src: Path, dst: Path) -> None:
    """Hard link, or a copy when the destination is on another file system."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_episodes(text: str) -> set[int]:
    """ "0-59,60,77" -> the set of those indices."""
    out: set[int] = set()
    for part in filter(None, (p.strip() for p in text.split(","))):
        first, _, last = part.partition("-")
        out.update(range(int(first), int(last or first) + 1))
    return out


class Dataset:
    """A LeRobot v2.1 dataset on disk (one parquet and one mp4 per camera for each episode)."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        info_path = self.root / "meta/info.json"
        if not info_path.exists():
            raise SurgeryError(f"{root} is not a LeRobot dataset (no meta/info.json)")
        self.info = json.loads(info_path.read_text())
        version = self.info.get("codebase_version")
        if version != "v2.1":
            raise SurgeryError(
                f"{root}: codebase_version {version!r}; this tool reads v2.1 only (v2.0: convert it with lerobot's "
                "convert_dataset_v20_to_v21 first; v3.0 keeps several episodes per file and is not supported)"
            )
        self.chunks = int(self.info["chunks_size"])
        self.fps = self.info["fps"]
        self.features: dict = self.info["features"]
        self.video_keys = [k for k, v in self.features.items() if v.get("dtype") == "video"]
        self.tasks = {t["task_index"]: t["task"] for t in read_jsonl(self.root / "meta/tasks.jsonl")}
        self.episodes = sorted(read_jsonl(self.root / "meta/episodes.jsonl"), key=lambda e: e["episode_index"])
        self.stats = {r["episode_index"]: r["stats"] for r in read_jsonl(self.root / "meta/episodes_stats.jsonl")}

    def data_path(self, episode: int) -> Path:
        return self.root / self.info["data_path"].format(episode_chunk=episode // self.chunks, episode_index=episode)

    def video_path(self, episode: int, key: str) -> Path:
        return self.root / self.info["video_path"].format(
            episode_chunk=episode // self.chunks, video_key=key, episode_index=episode
        )

    def table(self, episode: int, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.data_path(episode), columns=columns)

    def prompt(self, episode: int) -> str:
        """The episode's task, read from its rows (an episode with rows of several tasks is refused)."""
        indices = np.unique(self.table(episode, ["task_index"]).column(0).to_numpy())
        if len(indices) != 1:
            raise SurgeryError(
                f"{self.root} episode {episode}: rows of {len(indices)} tasks; one task per episode only"
            )
        if int(indices[0]) not in self.tasks:
            raise SurgeryError(
                f"{self.root} episode {episode}: task_index {int(indices[0])} is not in meta/tasks.jsonl"
            )
        return self.tasks[int(indices[0])]

    def prompts(self) -> dict[int, str]:
        """episode index -> task, for all episodes."""
        return {e["episode_index"]: self.prompt(e["episode_index"]) for e in self.episodes}


def column_matrix(table: pa.Table, key: str) -> np.ndarray:
    """A column as an (rows, dims) array; a scalar column has one dim."""
    column = table.column(key)
    kind = column.type
    if pa.types.is_list(kind) or pa.types.is_large_list(kind) or pa.types.is_fixed_size_list(kind):
        values = column.to_numpy(zero_copy_only=False)
        return np.stack(values) if len(values) else np.zeros((0, 0))
    return np.asarray(column.to_numpy(zero_copy_only=False)).reshape(-1, 1)


def set_column(table: pa.Table, key: str, values) -> pa.Table:
    """The table with one column replaced, in the column's own type."""
    field = table.schema.field(key)
    array = values if isinstance(values, pa.Array) else pa.array(values)
    return table.set_column(table.schema.get_field_index(key), field, array.cast(field.type))


def stats_of(values: np.ndarray) -> dict:
    """lerobot's per-episode stats of an (rows, dims) array."""
    values = np.asarray(values).reshape(len(values), -1)
    if np.issubdtype(values.dtype, np.integer):
        low, high = [int(x) for x in values.min(0)], [int(x) for x in values.max(0)]
    else:
        values = values.astype(np.float64)
        low, high = [float(x) for x in values.min(0)], [float(x) for x in values.max(0)]
    return {"min": low, "max": high, "mean": [float(x) for x in values.mean(0)],
            "std": [float(x) for x in values.std(0)], "count": [len(values)]}  # fmt: skip


def row_stats(table: pa.Table, features: dict) -> dict:
    """Stats of every numeric, non-image feature that is a column of the table."""
    out = {}
    for key, feature in features.items():
        if feature.get("dtype") in IMAGE_DTYPES or key not in table.column_names:
            continue
        values = column_matrix(table, key)
        if values.dtype.kind in "iufb" and len(values):
            out[key] = stats_of(values.astype(np.float64) if values.dtype.kind == "b" else values)
    return out


def aggregate_stats(per_episode: list[dict]) -> dict:
    """Dataset-wide stats from per-episode ones (as lerobot aggregates them: count-weighted mean and variance)."""
    out = {}
    for key in sorted({k for stats in per_episode for k in stats}):
        rows = [stats[key] for stats in per_episode if key in stats]
        counts = np.asarray([r["count"][0] for r in rows], dtype=np.float64)
        means = np.stack([np.asarray(r["mean"], dtype=np.float64) for r in rows])
        stds = np.stack([np.asarray(r["std"], dtype=np.float64) for r in rows])
        weights = (counts / counts.sum()).reshape(-1, *([1] * (means.ndim - 1)))
        mean = (means * weights).sum(0)
        variance = ((stds**2 + (means - mean) ** 2) * weights).sum(0)
        out[key] = {
            "min": np.stack([np.asarray(r["min"]) for r in rows]).min(0).tolist(),
            "max": np.stack([np.asarray(r["max"]) for r in rows]).max(0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(counts.sum())],
        }
    return out


@dataclasses.dataclass
class Piece:
    """One output episode: an episode of a source, whole or with some of its rows.

    ``table``: the rows to write when they are not the source's (trimmed or edited); frame_index and timestamp must be
    the source's, because the video is linked unchanged. ``prompt``: the task, when it is not the source's.
    ``note``: more fields for the episode's row in meta/surgery_episodes.jsonl.
    """

    source: Dataset
    episode: int
    table: pa.Table | None = None
    prompt: str | None = None
    note: dict = dataclasses.field(default_factory=dict)


def same_layout(a: Dataset, b: Dataset) -> str | None:
    """Why two datasets cannot go into one, or None."""
    if a.fps != b.fps:
        return f"fps {a.fps} and {b.fps}"
    shape = lambda d: {k: (v.get("dtype"), list(v.get("shape") or [])) for k, v in d.features.items()}  # noqa: E731
    one, two = shape(a), shape(b)
    if one != two:
        differing = sorted(k for k in set(one) | set(two) if one.get(k) != two.get(k))
        return f"features differ: {', '.join(differing)}"
    return None


def carry_meta(source: Dataset, meta: Path, mapping: dict[int, list[int]]) -> None:
    """A single source's other metadata files: a .jsonl whose rows carry ``episode_index`` keeps the rows of the kept
    episodes, renumbered (``source_episode_index``: the index in the source; rows without an episode are kept as they
    are); every other file and directory is copied."""
    for path in sorted((source.root / "meta").iterdir()):
        if path.name in STANDARD_META or path.name in OWN_META:
            continue
        if path.is_dir():
            shutil.copytree(path, meta / path.name)
            continue
        rows = read_jsonl(path) if path.suffix == ".jsonl" else None
        if not rows or not any(isinstance(r, dict) and "episode_index" in r for r in rows):
            shutil.copy2(path, meta / path.name)
            continue
        kept = []
        for row in rows:
            old = row.get("episode_index")
            if old is None:
                kept.append(row)
            for new in mapping.get(old, []):
                kept.append({**row, "episode_index": new, "source_episode_index": old})
        if kept:
            write_jsonl(meta / path.name, kept)


def write_dataset(
    out: Path | str,
    pieces: list[Piece],
    operation: dict,
    task_order: list[str] | None = None,
    source_labels: dict[Path, str] | None = None,
    validate: bool = True,
) -> dict:
    """Write the pieces as a new dataset at ``out`` (built in a .building dir next to it, renamed at the end).

    ``operation``: the record for meta/surgery.json ({"op": ..., "args": ...}). ``task_order``: the tasks in
    task_index order (default: in order of first use). ``source_labels``: with several sources, each source's other
    metadata goes to meta/by_source/<label>/; with one source it is carried over (see carry_meta).
    """
    out = Path(out)
    if not pieces:
        raise SurgeryError("nothing to write: no episode is left")
    if out.exists():
        raise SurgeryError(f"{out} exists")
    sources: list[Dataset] = []
    for piece in pieces:
        if all(piece.source is not s for s in sources):
            sources.append(piece.source)
    out_resolved = out.resolve()
    for source in sources:
        if out_resolved.is_relative_to(source.root) or source.root.is_relative_to(out_resolved):
            raise SurgeryError(
                f"{out} and {source.root} are inside one another: lerobot counts every file under a dataset"
            )
        problem = same_layout(sources[0], source)
        if problem:
            raise SurgeryError(f"{source.root} does not fit {sources[0].root}: {problem}")
    base = sources[0]
    tmp = out.with_name(f".{out.name}.building")
    shutil.rmtree(tmp, ignore_errors=True)
    meta = tmp / "meta"
    meta.mkdir(parents=True)

    prompts = [p.prompt if p.prompt is not None else p.source.prompt(p.episode) for p in pieces]
    tasks = list(task_order) if task_order else list(dict.fromkeys(prompts))
    missing = sorted(set(prompts) - set(tasks))
    if missing:
        raise SurgeryError(f"tasks not in the task order: {missing}")
    tasks = [t for t in tasks if t in set(prompts)]

    episodes, stats_rows, origin_rows = [], [], []
    mapping: dict[int, list[int]] = {}
    origins = {
        s.root: {r["episode_index"]: r for r in read_jsonl(s.root / "meta/surgery_episodes.jsonl")} for s in sources
    }
    index = 0
    for new, (piece, prompt) in enumerate(zip(pieces, prompts, strict=True)):
        source, old = piece.source, piece.episode
        table = piece.table if piece.table is not None else source.table(old)
        n = table.num_rows
        if n == 0:
            raise SurgeryError(f"{source.root} episode {old}: no rows left")
        columns = {
            "episode_index": np.full(n, new, dtype=np.int64),
            "index": np.arange(index, index + n, dtype=np.int64),
            "task_index": np.full(n, tasks.index(prompt), dtype=np.int64),
        }
        for key, values in columns.items():
            table = set_column(table, key, values)
        target = tmp / base.info["data_path"].format(episode_chunk=new // base.chunks, episode_index=new)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        for key in base.video_keys:
            video = source.video_path(old, key)
            if not video.exists():
                raise SurgeryError(f"missing video {video}")
            link(video, tmp / base.info["video_path"].format(
                episode_chunk=new // base.chunks, video_key=key, episode_index=new))  # fmt: skip
        episode_stats = dict(source.stats.get(old, {}))  # the images' stats stay the source's
        if piece.table is not None or not episode_stats:
            episode_stats.update(row_stats(table, base.features))
        for key, values in columns.items():
            episode_stats[key] = stats_of(values)
        stats_rows.append({"episode_index": new, "stats": episode_stats})
        source_episode = next(e for e in source.episodes if e["episode_index"] == old)
        episodes.append({**source_episode, "episode_index": new, "tasks": [prompt], "length": n})
        origin = {"episode_index": new, "source": str(source.root), "source_episode": old,
                  "source_length": source_episode.get("length"), "length": n, **piece.note}  # fmt: skip
        if old in origins[source.root]:
            origin["origin"] = origins[source.root][old]
        origin_rows.append(origin)
        if source is base:
            mapping.setdefault(old, []).append(new)
        index += n

    info = dict(base.info)
    info.update(
        total_episodes=len(episodes),
        total_frames=index,
        total_tasks=len(tasks),
        total_chunks=(len(episodes) - 1) // base.chunks + 1,
        total_videos=len(episodes) * len(base.video_keys),
        splits={"train": f"0:{len(episodes)}"},
    )
    (meta / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
    write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(tasks)])
    write_jsonl(meta / "episodes.jsonl", episodes)
    write_jsonl(meta / "episodes_stats.jsonl", stats_rows)
    write_jsonl(meta / "surgery_episodes.jsonl", origin_rows)
    if (base.root / "meta/stats.json").exists():  # a v2.0 leftover that some readers still use: kept in step
        write_json(meta / "stats.json", aggregate_stats([r["stats"] for r in stats_rows]))
    if len(sources) == 1:
        carry_meta(base, meta, mapping)
    else:
        labels = source_labels or {}
        for number, source in enumerate(sources):
            label = labels.get(source.root, f"{number:02d}_{source.root.name}")
            for path in sorted((source.root / "meta").iterdir()):
                if path.name in STANDARD_META or path.name in OWN_META or path.name == "by_source":
                    continue
                target = meta / "by_source" / label / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, target) if path.is_dir() else shutil.copy2(path, target)
    history = {}
    for source in sources:
        earlier = source.root / "meta/surgery.json"
        if earlier.exists():
            history[str(source.root)] = json.loads(earlier.read_text())
    record = {"tool": "lerobot-surgery", "version": VERSION,
              "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), **operation,
              "sources": [str(s.root) for s in sources], "episodes": len(episodes), "frames": index}  # fmt: skip
    if history:
        record["history"] = history
    write_json(meta / "surgery.json", record)

    if validate:
        from .check import check

        problems, _ = check(tmp)
        if problems:
            raise SurgeryError(f"the result failed its check and was left in {tmp}:\n  " + "\n  ".join(problems[:20]))
    tmp.rename(out)
    return {"out": str(out), "episodes": len(episodes), "frames": index, "tasks": tasks}
