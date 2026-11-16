"""Correct a dataset's frame rate label: frames recorded at one rate but stamped at another become a new dataset at
the true rate. Every row and frame stays, so a reader gets the same rows and pictures by index; only the time changes:

  timestamp       round(ts * from_fps) / fps; the timestamp stats follow
  meta/info.json  fps and each video's video.fps
  videos          the same stream with its time stamps scaled (ffmpeg -itsscale -c copy: nothing is re-encoded); every
                  output is read back: same frame count, each frame at k / fps, the container's rate fps

The sources are left alone (their videos may be hard links shared with other datasets), so new video files are written.
``link_map`` (a JSON file kept across runs) remembers source video -> retimed video, so datasets that share videos
share the retimed ones too. Its key is the source's inode, size and mtime, and a hit is linked only if it reads back as
that source retimed (the same packets, every frame on the fps grid); otherwise the video is remuxed again.

Needs PyAV and an ffmpeg binary with the ``setts`` bitstream filter (ffmpeg 4.4 or newer).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .check import TIMESTAMP_TOLERANCE_S
from .check import video_times
from .core import VERSION
from .core import Dataset
from .core import SurgeryError
from .core import read_jsonl
from .core import write_json
from .core import write_jsonl
from .ops import expand_sources


def source_key(video: Path) -> str:
    """A source video's link-map key: the same for all hard links of the file, not for a new file that got a deleted
    one's inode number. No device number (it can change between boots)."""
    st = video.stat()
    return f"{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def stream_digest(path: Path) -> tuple[np.ndarray, str, float]:
    """The frames' times, a hash of the packets' payloads in file order, and the container's frame rate."""
    import av

    digest = hashlib.blake2b(digest_size=16)
    with av.open(str(path)) as box:
        stream = box.streams.video[0]
        pts = []
        for packet in box.demux(stream):
            if packet.pts is not None:
                pts.append(packet.pts)
                digest.update(bytes(packet))
        times = np.sort(np.asarray(pts, dtype=np.float64)) * float(stream.time_base)
        return times, digest.hexdigest(), float(stream.average_rate)


def decoded_frames(path: Path) -> int:
    """How many frames a decoder gets out of the video."""
    import av

    with av.open(str(path)) as box:
        return sum(1 for _ in box.decode(video=0))


def linked_frames(job: tuple[Path, Path, float]) -> int:
    """The frame count of a link-map hit if it is the source retimed to fps, else 0."""
    import av

    src, made, fps = job
    try:
        times, packets, rate = stream_digest(made)
    except (av.error.FFmpegError, OSError, IndexError):  # IndexError: no video stream
        return 0
    before, source_packets, _ = stream_digest(src)
    if packets != source_packets or len(times) != len(before) or not len(times) or abs(rate - fps) > 1e-3:
        return 0
    if before[-1] > 0 and float(np.abs(times - before * (times[-1] / before[-1])).max()) > TIMESTAMP_TOLERANCE_S:
        return 0  # not the source's times scaled (e.g. shifted)
    return len(times) if float(np.abs(times - np.round(times * fps) / fps).max()) <= TIMESTAMP_TOLERANCE_S else 0


def retime_video(job: tuple[Path, Path, float, float]) -> tuple[str, int]:
    import av

    src, dst, from_fps, fps = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    # setts scales pts and dts alike and sets each packet's duration. Not -itsscale: on a stream whose dts starts below
    # zero (H.264 with frame reordering) it moves the pts too, and the first frames no longer decode.
    scale = f"{from_fps / fps!r}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c", "copy", "-an", "-bsf:v",
                    f"setts=pts=round(PTS*{scale}):dts=round(DTS*{scale}):duration=round(1/({fps:g}*TB))", str(dst)],
                   check=True)  # fmt: skip
    times, before = video_times(dst), video_times(src)
    if (
        len(times) == len(before)
        and len(times)
        and float(np.abs(times - before * from_fps / fps).max()) > TIMESTAMP_TOLERANCE_S
    ):
        raise SurgeryError(f"{dst}: its frames are not the source's times scaled by {from_fps / fps:g}")
    if decoded_frames(dst) != len(before):
        raise SurgeryError(f"{dst}: {decoded_frames(dst)} frames decode, the source has {len(before)}")
    with av.open(str(dst)) as box:
        rate = float(box.streams.video[0].average_rate)
    if abs(rate - fps) > 1e-3:
        raise SurgeryError(f"{dst}: the container says {rate} fps")
    if len(times) != len(before):
        raise SurgeryError(f"{dst}: {len(times)} frames, the source has {len(before)}")
    worst = float(np.abs(times - np.round(times * fps) / fps).max())
    if worst > TIMESTAMP_TOLERANCE_S:
        raise SurgeryError(f"{dst}: a frame is {worst:.6f} s off the {fps:g} fps grid")
    return str(src), len(times)


def retime_rows(job: tuple[Path, Path, float, float]) -> dict:
    src, dst, from_fps, fps = job
    table = pq.read_table(src)
    field = table.schema.field("timestamp")
    old = table.column("timestamp").to_numpy().astype(np.float64)
    frames = np.round(old * from_fps)
    if float(np.abs(old * from_fps - frames).max()) > 1e-2:
        raise SurgeryError(f"{src}: timestamps are not on the {from_fps:g} fps grid")
    new = frames / fps
    table = table.set_column(table.schema.get_field_index("timestamp"), field, pa.array(new).cast(field.type))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return {"episode_index": int(src.stem.split("_")[1]), "max_frame": int(frames.max()), "stats": {
        "min": [float(new.min())], "max": [float(new.max())], "mean": [float(new.mean())], "std": [float(new.std())],
        "count": [len(new)]}}  # fmt: skip


def retime_dataset(src: Path, dst: Path, from_fps: float | None, fps: float, pool, links: dict) -> dict:
    dataset = Dataset(src)
    info, src = dataset.info, dataset.root
    if from_fps is None:
        from_fps = float(info["fps"])
    if float(info["fps"]) != from_fps:
        raise SurgeryError(f"{src} is at {info['fps']} fps, not {from_fps:g}")
    if from_fps == fps:
        raise SurgeryError(f"{src} is already at {fps:g} fps")
    tmp = dst.with_name(f".{dst.name}.building")
    if dst.exists():
        raise SurgeryError(f"{dst} exists")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src / "meta", tmp / "meta")

    rows = sorted((src / "data").glob("chunk-*/episode_*.parquet"))
    results = list(pool.map(retime_rows, [(p, tmp / p.relative_to(src), from_fps, fps) for p in rows], chunksize=8))
    if len(results) != info["total_episodes"]:
        raise SurgeryError(f"{src}: {len(results)} parquet files, info.json says {info['total_episodes']} episodes")
    by_episode = {r["episode_index"]: r for r in results}

    hits, jobs = [], []
    for video in sorted((src / "videos").glob("chunk-*/*/episode_*.mp4")):
        entry = links.get(source_key(video))
        made = entry and entry["retimed"]
        if made and Path(made).exists():
            hits.append((video, Path(made), fps))
        else:
            jobs.append((video, tmp / video.relative_to(src), from_fps, fps))
    linked, counts = 0, {}
    for (video, made, _), frames in zip(hits, pool.map(linked_frames, hits, chunksize=8), strict=True):
        out = tmp / video.relative_to(src)
        if not frames:  # another video now (the inode number reused, the retimed set rebuilt) or none: remuxed
            jobs.append((video, out, from_fps, fps))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        os.link(made, out)
        counts[str(video)] = frames
        linked += 1
    counts.update(pool.map(retime_video, jobs, chunksize=8))
    for video, _, _, _ in jobs:
        links[source_key(video)] = {"source": str(video), "retimed": str(dst.resolve() / video.relative_to(src))}
    if len(jobs) + linked != info["total_videos"]:
        raise SurgeryError(f"{src}: {len(jobs) + linked} videos, info.json says {info['total_videos']}")
    for video, frames in counts.items():  # every row has its frame
        episode = int(Path(video).stem.split("_")[1])
        if by_episode[episode]["max_frame"] >= frames:
            raise SurgeryError(f"{video}: a row points past its {frames} frames")

    info["fps"] = int(fps) if float(fps).is_integer() else fps
    for feature in info["features"].values():
        if "video.fps" in feature.get("info", {}):
            feature["info"]["video.fps"] = info["fps"]
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
    stats_path = tmp / "meta/episodes_stats.jsonl"
    stats_rows = read_jsonl(stats_path)
    for row in stats_rows:
        row["stats"]["timestamp"] = by_episode[row["episode_index"]]["stats"]
    write_jsonl(stats_path, stats_rows)
    if (tmp / "meta/stats.json").exists():
        from .core import aggregate_stats

        write_json(tmp / "meta/stats.json", aggregate_stats([r["stats"] for r in stats_rows]))
    record = {"tool": "lerobot-surgery", "version": VERSION, "op": "retime",
              "args": {"from_fps": from_fps, "fps": fps,
                       "note": "rows, frames and pictures unchanged; only the time stamps"},
              "sources": [str(src)], "episodes": len(results)}  # fmt: skip
    earlier = src / "meta/surgery.json"
    if earlier.exists():
        record["history"] = {str(src): json.loads(earlier.read_text())}
    write_json(tmp / "meta/surgery.json", record)
    from .check import check

    problems, _ = check(tmp)
    if problems:
        raise SurgeryError(f"the result failed its check and was left in {tmp}:\n  " + "\n  ".join(problems[:20]))
    tmp.rename(dst)
    return {"out": str(dst), "episodes": len(results), "videos_remuxed": len(jobs), "videos_linked": linked,
            "from_fps": from_fps, "fps": fps}  # fmt: skip


def retime(src: Path, dst: Path, fps: float, from_fps: float | None = None, workers: int = 6,
           link_map: Path | None = None) -> list[dict]:  # fmt: skip
    """``src``: one dataset, or a directory of them (then ``dst`` is a directory too)."""
    if shutil.which("ffmpeg") is None:
        raise SurgeryError("retime needs the ffmpeg binary on PATH")
    src, dst = Path(src), Path(dst)
    links = json.loads(link_map.read_text()) if link_map and link_map.exists() else {}
    if (src / "meta/info.json").exists():
        pairs = [(src, dst)]
    else:
        roots = expand_sources([src])
        dst.mkdir(parents=True, exist_ok=True)
        pairs = [(root, dst / root.name) for root in roots]
    done = []
    with ProcessPoolExecutor(workers) as pool:
        for one, other in pairs:
            done.append(retime_dataset(one, other, from_fps, fps, pool, links))
            if link_map:
                link_map.write_text(json.dumps(links))
    return done
