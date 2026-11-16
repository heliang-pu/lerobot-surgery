"""The operations: drop, split, merge, trim. Each returns a small summary dict; the CLI prints it."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

import numpy as np

from .core import Dataset
from .core import Piece
from .core import SurgeryError
from .core import column_matrix
from .core import read_jsonl
from .core import set_column
from .core import write_dataset


def drop(src: Path, out: Path, episodes: set[int], reason: str, keep: bool = False) -> dict:
    """A copy of ``src`` without ``episodes`` (``keep``: with only them). The rest is renumbered from 0."""
    dataset = Dataset(src)
    known = {e["episode_index"] for e in dataset.episodes}
    unknown = episodes - known
    if unknown:
        raise SurgeryError(f"not in {src}: episodes {sorted(unknown)}")
    kept = sorted(episodes if keep else known - episodes)
    dropped = sorted(known - set(kept))
    pieces = [Piece(dataset, e) for e in kept]
    operation = {"op": "drop", "args": {"reason": reason, "dropped": dropped}}
    summary = write_dataset(out, pieces, operation, task_order=[dataset.tasks[i] for i in sorted(dataset.tasks)])
    return {**summary, "dropped": len(dropped), "of": len(known)}


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit].rstrip("_") or "task"


def split(src: Path, out_dir: Path, names: dict[str, str] | None = None) -> dict:
    """One dataset per task under ``out_dir``. ``names``: prompt (or a prefix of it) -> directory name; a task without
    a name gets ``<task_index>_<the prompt as a slug>``."""
    dataset = Dataset(src)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SurgeryError(f"{out_dir} exists")
    if out_dir.resolve().is_relative_to(dataset.root):
        raise SurgeryError("the out dir must not be inside the source dataset")
    names = names or {}
    directory = {}
    for task_index, prompt in sorted(dataset.tasks.items()):
        found = [name for prefix, name in names.items() if prompt.startswith(prefix)]
        if len(found) > 1:
            raise SurgeryError(f"several names fit the task {prompt!r}: {found}")
        directory[prompt] = found[0] if found else f"{task_index:02d}_{slug(prompt)}"
    if len(set(directory.values())) != len(directory):
        raise SurgeryError(f"two tasks would get the same directory name: {sorted(directory.values())}")
    by_task: dict[str, list[int]] = {}
    for episode, prompt in dataset.prompts().items():
        by_task.setdefault(prompt, []).append(episode)
    tmp = out_dir.with_name(f".{out_dir.name}.building")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    parts = {}
    for prompt, episodes in by_task.items():
        operation = {"op": "split", "args": {"task": prompt}}
        done = write_dataset(tmp / directory[prompt], [Piece(dataset, e) for e in sorted(episodes)], operation)
        parts[directory[prompt]] = {"episodes": done["episodes"], "frames": done["frames"], "task": prompt}
    tmp.rename(out_dir)
    return {"out": str(out_dir), "parts": parts}


def expand_sources(paths: list[Path]) -> list[Path]:
    """Dataset roots: each path is a dataset, or a directory of datasets (taken in name order, dot dirs left out)."""
    roots = []
    for path in paths:
        path = Path(path)
        if (path / "meta/info.json").exists():
            roots.append(path)
            continue
        inside = sorted(p for p in path.iterdir() if not p.name.startswith(".") and (p / "meta/info.json").exists())
        if not inside:
            raise SurgeryError(f"{path} is no dataset and holds none")
        roots.extend(inside)
    return roots


def match_task(key: str, tasks: list[str], option: str) -> str:
    found = [t for t in tasks if t == key] or [t for t in tasks if t.startswith(key)]
    if len(found) != 1:
        raise SurgeryError(f"{option} {key!r}: fits {len(found)} tasks (tasks: {tasks})")
    return found[0]


def merge(
    sources: list[Path],
    out: Path,
    repeat_task: dict[str, int] | None = None,
    repeat_source: dict[str, int] | None = None,
    tasks_from: Path | None = None,
) -> dict:
    """Several datasets as one. Tasks with the same prompt are one task. Episodes are numbered task by task, inside a
    task source by source. ``repeat_task`` (prompt or a prefix of it -> n) and ``repeat_source`` (source directory
    name -> n) write each episode n times in a row, as exact copies: a sampling weight for readers that have none."""
    datasets = [Dataset(root) for root in expand_sources(sources)]
    labels: dict[Path, str] = {}
    for number, dataset in enumerate(datasets):
        label = dataset.root.name
        labels[dataset.root] = label if label not in labels.values() else f"{label}~{number}"
    prompts = [dataset.prompts() for dataset in datasets]
    seen: list[str] = []  # the tasks in order of first use: source by source, in each by task_index
    for dataset, own in zip(datasets, prompts, strict=True):
        for _, prompt in sorted(dataset.tasks.items()):
            if prompt in own.values() and prompt not in seen:
                seen.append(prompt)
    if tasks_from:
        order = [
            t["task"] for t in sorted(read_jsonl(Path(tasks_from) / "meta/tasks.jsonl"), key=lambda t: t["task_index"])
        ]
        missing = sorted(set(seen) - set(order))
        if missing:
            raise SurgeryError(f"{tasks_from} lacks the tasks {missing}")
        order = [t for t in order if t in seen]
    else:
        order = seen
    task_times = {t: 1 for t in order}
    for key, n in (repeat_task or {}).items():
        task_times[match_task(key, order, "--repeat-task")] = int(n)
    source_times = {label: 1 for label in labels.values()}
    for key, n in (repeat_source or {}).items():
        if key not in source_times:
            raise SurgeryError(f"--repeat-source {key!r}: no such source (sources: {sorted(source_times)})")
        source_times[key] = int(n)
    pieces, count = [], {}
    for task in order:
        for dataset, own in zip(datasets, prompts, strict=True):
            label = labels[dataset.root]
            times = task_times[task] * source_times[label]
            for episode in sorted(e for e, p in own.items() if p == task):
                for copy in range(times):
                    pieces.append(Piece(dataset, episode, note={"source_label": label, "copy": copy}))
                row = count.setdefault((task, label), {"task": task, "source": label, "episodes": 0, "repeat": times})
                row["episodes"] += 1
    operation = {"op": "merge", "args": {"repeat_task": {t: n for t, n in task_times.items() if n != 1},
                                         "repeat_source": {s: n for s, n in source_times.items() if n != 1},
                                         "tasks_from": str(tasks_from or "")}}  # fmt: skip
    summary = write_dataset(out, pieces, operation, task_order=order, source_labels=labels)
    return {**summary, "parts": list(count.values())}


def moving_length(state: np.ndarray, threshold: float) -> int:
    """Rows up to and including the last pose that differs from the one before it by more than threshold."""
    moving = np.flatnonzero(np.abs(np.diff(state, axis=0)).max(axis=1) > threshold)
    return int(moving[-1]) + 2 if len(moving) else 1


def pause_rows(state: np.ndarray, threshold: float, min_steps: int) -> np.ndarray:
    """Row indices left after every still run of min_steps or more steps is cut to its first pose."""
    still = np.abs(np.diff(state, axis=0)).max(axis=1) <= threshold
    keep = np.ones(len(state), bool)
    j = 0
    while j < len(still):
        if not still[j]:
            j += 1
            continue
        k = j
        while k < len(still) and still[k]:
            k += 1
        if k - j >= min_steps and np.ptp(state[j : k + 1], axis=0).max() <= 2.5 * threshold:
            keep[j + 1 : k + 1] = False  # rows j+1..k repeat row j's pose
        j = k
    return np.flatnonzero(keep)


def trim(
    src: Path,
    out: Path,
    state_key: str = "observation.state",
    static_threshold: float = 2e-3,
    tail: bool = True,
    cut_pauses: int = 0,
    start_at: dict | None = None,
    min_length: int = 1,
    next_state_action: bool = False,
    action_key: str = "action",
) -> dict:
    """Cut rows without motion. Only rows go; videos are linked unchanged and the rows keep their timestamps.

    ``tail``: drop the still rows after the last motion (the last moving pose stays). ``cut_pauses`` N > 0: a still run
    of N or more steps inside an episode keeps only its first pose (see ``check``'s warning: lerobot's own loader then
    refuses the dataset). ``start_at``: {episode: first row to keep, or null to leave the episode out}.
    ``next_state_action``: the row before a cut gets the next kept row's state as its action (for data whose action is
    the next state; otherwise it would still point at the cut pose). Episodes shorter than ``min_length`` afterwards
    are left out.
    """
    dataset = Dataset(src)
    if state_key not in dataset.features:
        raise SurgeryError(f"{src}: no feature {state_key!r} (--state-key)")
    start_at = {int(k): v for k, v in (start_at or {}).items()}
    pieces, skipped, cut_total = [], [], {"head": 0, "tail": 0, "pauses": 0}
    for episode in dataset.episodes:
        number = episode["episode_index"]
        table = dataset.table(number)
        length = table.num_rows
        head = 0
        if number in start_at:
            entry = start_at[number]
            if entry is None:
                skipped.append(number)
                continue
            head = int(entry["start"] if isinstance(entry, dict) else entry)
        table = table.slice(head)
        if table.num_rows == 0:
            skipped.append(number)
            continue
        state = column_matrix(table, state_key).astype(np.float64)
        keep = moving_length(state, static_threshold) if tail else len(state)
        table, state = table.slice(0, keep), state[:keep]
        rows = pause_rows(state, static_threshold, cut_pauses) if cut_pauses else np.arange(keep)
        paused = keep - len(rows)
        if paused:
            table = table.take(rows)
            gap = np.flatnonzero(np.diff(rows) > 1)
            if next_state_action and len(gap):
                action = column_matrix(table, action_key).copy()
                action[gap] = state[rows[gap + 1]]
                table = set_column(table, action_key, list(action))
        if table.num_rows < min_length:
            skipped.append(number)
            continue
        tail_cut = length - head - keep
        cut_total["head"] += head
        cut_total["tail"] += tail_cut
        cut_total["pauses"] += paused
        edited = head or tail_cut or paused
        note = {"head_cut": head, "static_tail_cut": tail_cut, "pause_cut": paused}
        pieces.append(Piece(dataset, number, table=table if edited else None, note=note))
    operation = {"op": "trim", "args": {"state_key": state_key, "static_threshold": static_threshold, "tail": tail,
                                        "cut_pauses": cut_pauses, "min_length": min_length,
                                        "next_state_action": next_state_action, "left_out": skipped}}  # fmt: skip
    summary = write_dataset(out, pieces, operation, task_order=[dataset.tasks[i] for i in sorted(dataset.tasks)])
    return {**summary, "rows_cut": cut_total, "left_out": skipped}


def load_json_arg(value: str | None):
    """A CLI argument that is JSON text or a path to a JSON file."""
    if not value:
        return None
    path = Path(value)
    return json.loads(path.read_text() if path.exists() else value)
