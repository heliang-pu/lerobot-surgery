# lerobot-surgery

**English** | [中文](README.zh-CN.md)

A command-line tool for the dataset operations that [LeRobot](https://github.com/huggingface/lerobot) does not ship:
check a dataset, compare datasets before mixing them, drop episodes, split by task, merge, cut motionless frames, and
fix a wrong frame-rate label. It works on **LeRobot v2.1** datasets and needs only numpy and pyarrow; lerobot itself
does not have to be installed.

Every operation leaves its source untouched, never re-encodes a video, and records where each output episode came from.

## Install

```bash
git clone https://github.com/heliang-pu/lerobot-surgery && cd lerobot-surgery
pip install -e .              # numpy, pyarrow
pip install -e ".[video]"     # plus PyAV, for `check --videos` and `retime`
```

`retime` also needs the `ffmpeg` binary (4.4 or newer, with the `setts` bitstream filter).
Without installing: `PYTHONPATH=. python -m lerobot_surgery <command> ...`

## Commands

| Command | What it does |
|---|---|
| `inspect <dataset>...` | What each dataset holds: frame rate, episodes per task, cameras, per-dimension ranges of every vector feature. Warns when datasets differ in frame rate or features |
| `check <dataset>... [--videos] [--lerobot]` | Do the files, the rows and the metadata agree with each other? |
| `drop <src> <out> --drop 3,10-12 --reason "..."` | A copy without some episodes, renumbered from 0. `--keep` keeps only the listed ones |
| `split <src> <out_dir>` | One standalone dataset per task |
| `merge <dataset or dir>... -o <out>` | Several datasets as one; identical prompts become one task |
| `trim <src> <out>` | Cut rows without motion: still tails, pauses, a head |
| `retime <src> <out> --fps 10` | Correct the frame-rate label (for example 10 Hz data stamped as 25 fps) |

```bash
lerobot-surgery inspect data/sim_pick data/real_pick
lerobot-surgery check data/real_pick --videos --lerobot
lerobot-surgery drop data/real data/real_clean --drop 12,40-43 --reason "a hand is in the picture"
lerobot-surgery split data/ten_tasks data/by_task --name "Pick up the cup=pick_cup"
lerobot-surgery merge data/by_task data/new_batch -o data/merged --repeat-task "Pick up the cup=3"
lerobot-surgery trim data/real data/real_trimmed --cut-pauses 5 --min-length 40 --next-state-action
lerobot-surgery retime data/real25 data/real10 --fps 10 --link-map data/.retime_links.json
```

`lerobot-surgery <command> --help` lists every option.

## What every command guarantees

- **The source is never modified.** The output is built in a `.<name>.building` directory next to the target, checked,
  and only then renamed. An existing target is refused.
- **Videos are never re-encoded.** They are hard-linked (copied across file systems). `retime` rewrites time stamps
  only; the encoded pictures are copied bit for bit.
- **Every output passes `check` before it appears.** If it fails, the command stops and leaves the half-built
  directory for inspection.
- **Provenance is kept.** Each output has
  - `meta/surgery.json`: the operation, its arguments and its sources; a source's own `surgery.json` goes into `history`;
  - `meta/surgery_episodes.jsonl`: for each episode, the source dataset and episode it came from, chained in `origin`
    across repeated operations.
- **Other metadata is carried along.** With one source, any `meta/*.jsonl` whose rows carry `episode_index` is
  rewritten to the new numbering (the old number is kept as `source_episode_index`) and other files are copied. With
  several sources (`merge`), each source's extra metadata goes to `meta/by_source/<source dir>/`.
- **Statistics follow.** Per-episode stats of `episode_index`, `index` and `task_index` are recomputed; numeric
  features are recomputed for episodes whose rows were cut; image stats are kept from the source. If the source has a
  `meta/stats.json`, it is rebuilt from the per-episode stats.

## Notes on each command

**check** verifies the totals in `info.json`, each episode's row count, `episode_index` / `index` / `task_index`,
that `timestamp == frame_index / fps`, that every video exists, and that no parquet or mp4 under the dataset belongs
to no episode (lerobot counts every such file). `--videos` also opens each video and compares its frames with the
rows. `--lerobot` loads the dataset with lerobot and decodes its last frame.

**inspect** exits with 1 when datasets differ in frame rate or features. It also prints *notes*: a dimension whose
values are more than 15 times larger in one dataset than in another. That is usually a unit mismatch (a gripper in
metres in one set and as a 0–1 fraction in the other), but it can also be a joint that one task hardly moves, so it is
a hint, not an error.

**merge** numbers episodes task by task and, inside a task, source by source. `--repeat-task "<prompt or prefix>=n"`
and `--repeat-source <source dir>=n` write each episode n times in a row as exact copies: a sampling weight for
training code that has none. `--tasks-from <dataset>` takes the task order from that dataset.

**trim** removes rows only. Videos stay as they are and the remaining rows keep their time stamps, so they still
decode to the right pictures.
- By default the still rows after the last motion are cut (the last moving pose stays); `--keep-tail` turns that off.
- `--cut-pauses N` keeps only the first pose of every still run of N or more steps inside an episode.
  **After this the rows are no longer 1/fps apart, and lerobot's own loader refuses the dataset** (its timestamp
  check). Readers that take each row's time stamp as it is are not affected. `check` warns about it.
- `--start-at '{"<episode>": <first row to keep>}'` cuts a head; `null` leaves the episode out.
- `--next-state-action` gives the row before a cut the next kept row's state as its action. Use it only for data whose
  action is the next state.
- "Still" means the largest per-dimension change between two rows is at most `--static-threshold` (default 0.002).

**retime** is for data whose frame rate was labelled wrongly. Rows, frames and pictures stay; time stamps go from
`k / old_fps` to `k / true_fps`. `--link-map` names a JSON file kept across runs so that datasets sharing videos
through hard links still share them afterwards.

## Limits

- LeRobot **v2.1** only (one parquet per episode, one mp4 per camera and episode). Convert v2.0 with lerobot's own
  script first. v3.0 is not supported.
- One task per episode.

## Tests

```bash
python -m pytest -q
```

The tests build small synthetic datasets with real parquet files and real 32×24 videos, with and without B-frame
reordering. When lerobot is installed, the results of each operation are also loaded with lerobot.
