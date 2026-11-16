from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from helpers import CAMERA
from helpers import DIMS
from helpers import make_dataset
from helpers import moving_state
from lerobot_surgery import ops
from lerobot_surgery.check import check
from lerobot_surgery.check import video_times
from lerobot_surgery.cli import lerobot_load
from lerobot_surgery.cli import main
from lerobot_surgery.core import Dataset
from lerobot_surgery.core import SurgeryError
from lerobot_surgery.core import column_matrix
from lerobot_surgery.core import read_jsonl
from lerobot_surgery.core import write_jsonl

A, B = "Press the button.", "Open the drawer and put the cube inside."


def states(root, episode):
    return column_matrix(Dataset(root).table(episode), "observation.state")


def clean(root, videos=True):
    problems, warnings = check(root, videos=videos)
    assert problems == []
    return warnings


def test_a_fresh_dataset_passes_the_check(tmp_path):
    root = make_dataset(tmp_path / "set", [(A, 12), (B, 9), (A, 7)])
    assert clean(root) == []


def test_check_names_what_is_wrong(tmp_path):
    root = make_dataset(tmp_path / "set", [(A, 12), (B, 9)])
    dataset = Dataset(root)
    dataset.video_path(1, CAMERA).unlink()
    shutil.copy2(dataset.data_path(0), root / "data/chunk-000/episode_000007.parquet")
    info = json.loads((root / "meta/info.json").read_text())
    info["total_frames"] = 5
    (root / "meta/info.json").write_text(json.dumps(info))
    problems, _ = check(root)
    text = "\n".join(problems)
    assert "missing videos/chunk-000" in text
    assert "belong to no episode" in text
    assert "total_frames 5" in text


def test_drop_renumbers_and_keeps_the_trace(tmp_path):
    src = make_dataset(tmp_path / "src", [(A, 12), (B, 9), (A, 7), (B, 5)], stats_json=True)
    write_jsonl(src / "meta/notes.jsonl", [{"episode_index": 1, "note": "gone"}, {"episode_index": 2, "note": "stays"},
                                             {"note": "about no episode"}])  # fmt: skip
    (src / "meta/readme.txt").write_text("hello\n")
    done = ops.drop(src, tmp_path / "out", {1}, "bad grasp")
    assert (done["episodes"], done["frames"], done["dropped"]) == (3, 24, 1)
    out = Dataset(tmp_path / "out")
    assert clean(out.root) == []
    assert [e["length"] for e in out.episodes] == [12, 7, 5]
    np.testing.assert_array_equal(states(out.root, 1), states(src, 2))
    assert out.table(2).column("index").to_pylist() == list(range(19, 24))
    assert out.stats[1]["index"]["min"] == [12] and out.stats[1]["episode_index"]["max"] == [1]
    assert out.stats[1]["observation.state"] == Dataset(src).stats[2]["observation.state"]
    origins = read_jsonl(out.root / "meta/surgery_episodes.jsonl")
    assert [(o["episode_index"], o["source_episode"]) for o in origins] == [(0, 0), (1, 2), (2, 3)]
    notes = read_jsonl(out.root / "meta/notes.jsonl")
    assert notes == [{"episode_index": 1, "note": "stays", "source_episode_index": 2}, {"note": "about no episode"}]
    assert (out.root / "meta/readme.txt").read_text() == "hello\n"
    record = json.loads((out.root / "meta/surgery.json").read_text())
    assert record["op"] == "drop" and record["args"] == {"reason": "bad grasp", "dropped": [1]}
    aggregated = json.loads((out.root / "meta/stats.json").read_text())
    assert aggregated["index"]["count"] == [24] and aggregated["index"]["max"] == [23]
    everything = np.concatenate([states(out.root, e) for e in range(3)]).astype(np.float64)
    np.testing.assert_allclose(aggregated["observation.state"]["mean"], everything.mean(0), rtol=1e-6)
    np.testing.assert_allclose(aggregated["observation.state"]["std"], everything.std(0), rtol=1e-6)

    again = ops.drop(out.root, tmp_path / "out2", {0, 2}, "only these", keep=True)
    assert again["episodes"] == 2
    second = json.loads((tmp_path / "out2/meta/surgery.json").read_text())
    assert second["history"][str(out.root)]["args"]["reason"] == "bad grasp"
    chain = read_jsonl(tmp_path / "out2/meta/surgery_episodes.jsonl")[1]
    assert chain["source_episode"] == 2 and chain["origin"]["source_episode"] == 3


def test_drop_refuses_bad_requests(tmp_path):
    src = make_dataset(tmp_path / "src", [(A, 6), (B, 6)], video=False)
    with pytest.raises(SurgeryError, match="not in"):
        ops.drop(src, tmp_path / "out", {5}, "x")
    with pytest.raises(SurgeryError, match="no episode is left"):
        ops.drop(src, tmp_path / "out", {0, 1}, "x")
    with pytest.raises(SurgeryError, match="inside one another"):
        ops.drop(src, src / "inner", {0}, "x")
    assert not (tmp_path / "out").exists()


def test_split_then_merge_gives_the_rows_back(tmp_path):
    src = make_dataset(tmp_path / "src", [(A, 12), (B, 9), (A, 7), (B, 5)])
    done = ops.split(src, tmp_path / "by_task", {"Press": "press"})
    assert set(done["parts"]) == {"press", "01_open_the_drawer_and_put_the_cube_inside"}
    press = Dataset(tmp_path / "by_task/press")
    assert clean(press.root) == []
    assert [e["length"] for e in press.episodes] == [12, 7] and press.tasks == {0: A}
    assert press.table(1).column("task_index").to_pylist() == [0] * 7

    merged = ops.merge([tmp_path / "by_task"], tmp_path / "merged", tasks_from=src)
    assert (merged["episodes"], merged["frames"], merged["tasks"]) == (4, 33, [A, B])
    out = Dataset(tmp_path / "merged")
    assert clean(out.root) == []
    for new, old in enumerate([0, 2, 1, 3]):  # numbered task by task
        np.testing.assert_array_equal(states(out.root, new), states(src, old))
    assert out.table(2).column("task_index").to_pylist() == [1] * 9
    assert (out.root / "meta/by_source").exists() is False  # the parts had no metadata of their own


def test_merge_repeats_and_refuses_what_does_not_fit(tmp_path):
    one = make_dataset(tmp_path / "one", [(A, 6), (B, 6)], video=False)
    two = make_dataset(tmp_path / "two", [(B, 8)], video=False)
    (two / "meta/collector.json").write_text("{}\n")
    done = ops.merge([one, two], tmp_path / "out", repeat_task={"Press": 3}, repeat_source={"two": 2})
    assert done["episodes"] == 3 + 1 + 2
    out = Dataset(tmp_path / "out")
    assert clean(out.root, videos=False) == []
    assert [e["tasks"][0] for e in out.episodes] == [A, A, A, B, B, B]
    assert [e["length"] for e in out.episodes] == [6, 6, 6, 6, 8, 8]
    copies = [(r["source_label"], r["copy"]) for r in read_jsonl(out.root / "meta/surgery_episodes.jsonl")]
    assert copies == [("one", 0), ("one", 1), ("one", 2), ("one", 0), ("two", 0), ("two", 1)]
    assert (out.root / "meta/by_source/two/collector.json").exists()

    other_fps = make_dataset(tmp_path / "fast", [(A, 6)], fps=25, video=False)
    with pytest.raises(SurgeryError, match="fps 10 and 25"):
        ops.merge([one, other_fps], tmp_path / "bad")
    with pytest.raises(SurgeryError, match="fits 0 tasks"):
        ops.merge([one, two], tmp_path / "bad", repeat_task={"Nothing": 2})


def test_trim_cuts_tail_pause_and_head(tmp_path):
    move = moving_state(30)
    still = np.repeat(move[9:10], 6, axis=0)  # rows 10..15 repeat row 9: a pause of 6 steps
    paused = np.concatenate([move[:10], still, move[10:20], np.repeat(move[19:20], 5, axis=0)])  # + a still tail of 5
    src = make_dataset(tmp_path / "src", [(A, paused), (A, 14), (A, np.repeat(move[:1], 8, axis=0))])
    done = ops.trim(src, tmp_path / "out", cut_pauses=5, start_at={"1": 4}, min_length=3, next_state_action=True)
    assert done["rows_cut"] == {"head": 4, "tail": 5, "pauses": 6}
    assert done["left_out"] == [2]  # it never moves: one row is left, shorter than --min-length
    out = Dataset(tmp_path / "out")
    warnings = clean(out.root)
    assert len(warnings) == 1 and "1 episodes have rows that are not 1/fps apart" in warnings[0]
    assert [e["length"] for e in out.episodes] == [20, 10]
    first = out.table(0)
    assert first.column("frame_index").to_pylist() == [*range(10), *range(16, 26)]  # the video is the source's
    np.testing.assert_array_equal(column_matrix(first, "observation.state"), move[:20])
    np.testing.assert_array_equal(column_matrix(first, "action")[9], move[10])  # not the cut pose any more
    assert out.table(1).column("frame_index").to_pylist() == list(range(4, 14))
    assert out.stats[0]["observation.state"]["count"] == [20]
    np.testing.assert_allclose(out.stats[0]["observation.state"]["max"], move[19], rtol=1e-6)
    note = read_jsonl(out.root / "meta/surgery_episodes.jsonl")[0]
    assert (note["head_cut"], note["static_tail_cut"], note["pause_cut"]) == (0, 5, 6)

    plain = ops.trim(src, tmp_path / "tails_only")
    assert plain["rows_cut"] == {"head": 0, "tail": 5 + 7, "pauses": 0}
    assert clean(tmp_path / "tails_only") == []  # only tails cut: still on the grid


@pytest.mark.parametrize("codec", ["libx264", "mpeg4"])  # with and without frame reordering
def test_retime_changes_only_the_time(tmp_path, monkeypatch, codec):
    av = pytest.importorskip("av")
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg binary")
    if codec not in av.codecs_available:
        pytest.skip(f"PyAV has no {codec}")
    import helpers
    from lerobot_surgery.retime import decoded_frames
    from lerobot_surgery.retime import retime

    monkeypatch.setattr(helpers, "CODEC", codec)

    src = make_dataset(tmp_path / "src", [(A, 12), (B, 9)], fps=25)
    link_map = tmp_path / "links.json"
    done = retime(src, tmp_path / "out", fps=10, workers=2, link_map=link_map)
    assert done[0]["videos_remuxed"] == 2 and done[0]["videos_linked"] == 0
    out = Dataset(tmp_path / "out")
    assert out.fps == 10 and out.features[CAMERA]["info"]["video.fps"] == 10
    assert clean(out.root) == []
    np.testing.assert_allclose(out.table(0).column("timestamp").to_numpy(), np.arange(12) / 10, atol=1e-6)
    np.testing.assert_array_equal(states(out.root, 1), states(src, 1))
    np.testing.assert_allclose(video_times(out.video_path(0, CAMERA)), np.arange(12) / 10, atol=1e-4)
    assert decoded_frames(out.video_path(0, CAMERA)) == 12
    assert out.stats[0]["timestamp"]["max"] == [pytest.approx(1.1)]
    assert json.loads((out.root / "meta/surgery.json").read_text())["args"]["from_fps"] == 25

    again = retime(src, tmp_path / "out_again", fps=10, workers=2, link_map=link_map)  # the same videos: linked
    assert again[0]["videos_linked"] == 2 and again[0]["videos_remuxed"] == 0
    assert (
        out.video_path(0, CAMERA).stat().st_ino
        == (tmp_path / "out_again" / out.video_path(0, CAMERA).relative_to(out.root)).stat().st_ino
    )
    with pytest.raises(SurgeryError, match="already at 10 fps"):
        retime(out.root, tmp_path / "no", fps=10)


def test_command_line(tmp_path, capsys):
    src = make_dataset(tmp_path / "src", [(A, 12), (B, 9), (A, 7)], video=False)
    assert main(["drop", str(src), str(tmp_path / "out"), "--drop", "1-2", "--reason", "test"]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] == 1
    assert main(["check", str(tmp_path / "out")]) == 0
    assert "ok" in capsys.readouterr().out
    assert main(["drop", str(src), str(tmp_path / "out"), "--drop", "0", "--reason", "test"]) == 2
    assert "exists" in capsys.readouterr().err
    assert main(["inspect", str(src), str(tmp_path / "out")]) == 0
    report = capsys.readouterr().out
    assert "observation.state: 28 rows x 4 dims" in report and "episodes 3" in report and "warning" not in report
    scaled = make_dataset(tmp_path / "mm", [(A, 1000 * moving_state(10))], fps=25, video=False)
    assert main(["inspect", str(src), str(scaled)]) == 1
    report = capsys.readouterr().out
    assert "warning: fps" in report and "note: observation.state[1]" in report and "another unit" in report
    assert main(["split", str(src), str(tmp_path / "parts"), "--name", "Press=press", "--name", "Open=open"]) == 0
    assert main(["merge", str(tmp_path / "parts"), "-o", str(tmp_path / "back"), "--repeat-task", "Open=2"]) == 0
    assert Dataset(tmp_path / "back").info["total_episodes"] == 4
    assert DIMS == 4


def test_lerobot_reads_the_results(tmp_path):
    if lerobot_load(tmp_path) == "lerobot is not installed in this environment":
        pytest.skip("lerobot is not installed")
    src = make_dataset(tmp_path / "src", [(A, 12), (B, 9), (A, 7)])
    assert lerobot_load(src) is None, "the synthetic dataset itself must load"
    ops.drop(src, tmp_path / "dropped", {0}, "test")
    assert lerobot_load(tmp_path / "dropped") is None
    ops.split(src, tmp_path / "parts")
    ops.merge([tmp_path / "parts"], tmp_path / "merged", repeat_task={"Press": 2})
    assert lerobot_load(tmp_path / "merged") is None
    ops.trim(make_dataset(tmp_path / "tail", [(A, np.concatenate([moving_state(10), np.repeat(moving_state(10)[-1:], 4, 0)]))]),
             tmp_path / "trimmed")  # fmt: skip
    assert lerobot_load(tmp_path / "trimmed") is None
