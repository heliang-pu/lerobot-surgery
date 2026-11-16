"""lerobot-surgery: operations on LeRobot v2.1 datasets that lerobot itself does not offer.

    lerobot-surgery inspect <dataset>...                      what is inside; what differs between datasets
    lerobot-surgery check   <dataset> [--videos] [--lerobot]   do files, rows and metadata agree?
    lerobot-surgery drop    <src> <out> --drop 3,10-12 --reason "..."     (or --keep ...)
    lerobot-surgery split   <src> <out_dir> [--name "<prompt prefix>=<dir>" ...]
    lerobot-surgery merge   <dataset or dir of datasets>... -o <out> [--repeat-task "<prompt>=3"] [--repeat-source <dir>=2]
    lerobot-surgery trim    <src> <out> [--cut-pauses 5] [--start-at map.json] [--min-length 40]
    lerobot-surgery retime  <src> <out> --fps 10 [--from-fps 25]

No operation changes its source. Outputs are built in a ".<name>.building" directory next to the target, checked, and
renamed; videos are hard-linked (copied across file systems), never re-encoded. Every output carries
meta/surgery.json (what was done) and meta/surgery_episodes.jsonl (where each episode came from).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import ops
from .check import check
from .core import VERSION
from .core import SurgeryError
from .core import parse_episodes


def pairs(items: list[str], option: str, number: bool = True) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.rpartition("=")
        if not sep or not key:
            raise SurgeryError(f'{option} wants "<name>=<value>", got {item!r}')
        out[key] = int(value) if number else value
    return out


def lerobot_load(root: Path) -> str | None:
    """Load the dataset with lerobot and decode its last frame; the error text, or None if it loads."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            return "lerobot is not installed in this environment"
    try:
        loaded = LeRobotDataset(f"local/{root.name}", root=root)
        loaded[loaded.num_frames - 1]
    except Exception as error:  # noqa: BLE001  (whatever lerobot raises is the answer)
        return f"{type(error).__name__}: {error}"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lerobot-surgery", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="what the datasets hold and how they differ")
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--keys", nargs="+", help="vector features to report (default: all float vectors)")
    p.add_argument("--max-episodes", type=int, help="read only this many episodes, spread over the dataset")

    p = sub.add_parser("check", help="do files, rows and metadata agree?")
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--videos", action="store_true", help="also open every video and compare it with the rows (slow)")
    p.add_argument("--lerobot", action="store_true", help="also load it with lerobot and decode the last frame")

    p = sub.add_parser("drop", help="a copy without some episodes")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--drop", help="episodes to leave out: 0-59,60,77")
    group.add_argument("--keep", help="episodes to keep, the rest is left out")
    p.add_argument("--reason", required=True, help="recorded in meta/surgery.json")

    p = sub.add_parser("split", help="one dataset per task")
    p.add_argument("src", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--name", action="append", default=[], help='"<prompt or a prefix of it>=<directory name>"')

    p = sub.add_parser("merge", help="several datasets as one")
    p.add_argument("sources", nargs="+", type=Path, help="datasets, or directories of datasets")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--repeat-task", action="append", default=[], help='"<prompt or prefix>=<n>": each episode n times')
    p.add_argument("--repeat-source", action="append", default=[], help='"<source directory name>=<n>"')
    p.add_argument("--tasks-from", type=Path, help="a dataset whose meta/tasks.jsonl sets the task order")

    p = sub.add_parser("trim", help="cut rows without motion (tails, pauses, heads)")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--state-key", default="observation.state")
    p.add_argument("--action-key", default="action")
    p.add_argument("--static-threshold", type=float, default=2e-3, help="largest per-row change that counts as still")
    p.add_argument("--keep-tail", action="store_true", help="do not cut the still rows at the end")
    p.add_argument("--cut-pauses", type=int, default=0, help="cut still runs of this many steps or more (0: keep)")
    p.add_argument("--start-at", help='JSON or a file: {"<episode>": <first row to keep> | null (leave out)}')
    p.add_argument("--min-length", type=int, default=1, help="episodes shorter after the cut are left out")
    p.add_argument("--next-state-action", action="store_true",
                   help="the row before a cut gets the next kept state as its action (data with action = next state)")  # fmt: skip

    p = sub.add_parser("retime", help="correct the frame rate label (rows and frames unchanged)")
    p.add_argument("src", type=Path, help="a dataset, or a directory of datasets")
    p.add_argument("out", type=Path)
    p.add_argument("--fps", type=float, required=True, help="the true frame rate")
    p.add_argument("--from-fps", type=float, help="the rate the source is labelled with (default: its info.json)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument(
        "--link-map", type=Path, help="JSON file kept across runs: datasets sharing videos share retimed ones"
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "inspect":
        from .inspect_ import inspect

        warnings, notes = inspect(args.datasets, args.keys, args.max_episodes)
        print()
        for warning in warnings:
            print(f"warning: {warning}")
        for note in notes:
            print(f"note: {note}")
        return 1 if warnings else 0
    if args.command == "check":
        failed = False
        for root in args.datasets:
            problems, warnings = check(root, videos=args.videos)
            if args.lerobot and not problems:
                error = lerobot_load(Path(root).resolve())
                if error:
                    problems.append(f"lerobot does not load it: {error}")
            for line in problems:
                print(f"{root}: PROBLEM {line}")
            for line in warnings:
                print(f"{root}: warning {line}")
            if not problems:
                print(f"{root}: ok" + (f" ({len(warnings)} warnings)" if warnings else ""))
            failed = failed or bool(problems)
        return 1 if failed else 0
    if args.command == "drop":
        done = ops.drop(args.src, args.out, parse_episodes(args.drop or args.keep), args.reason, keep=bool(args.keep))
    elif args.command == "split":
        done = ops.split(args.src, args.out_dir, pairs(args.name, "--name", number=False))
    elif args.command == "merge":
        done = ops.merge(args.sources, args.out, pairs(args.repeat_task, "--repeat-task"),
                         pairs(args.repeat_source, "--repeat-source"), args.tasks_from)  # fmt: skip
    elif args.command == "trim":
        done = ops.trim(args.src, args.out, state_key=args.state_key, static_threshold=args.static_threshold,
                        tail=not args.keep_tail, cut_pauses=args.cut_pauses, start_at=ops.load_json_arg(args.start_at),
                        min_length=args.min_length, next_state_action=args.next_state_action,
                        action_key=args.action_key)  # fmt: skip
    else:
        from .retime import retime

        done = retime(args.src, args.out, args.fps, args.from_fps, args.workers, args.link_map)
    print(json.dumps(done, indent=1, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except SurgeryError as error:
        print(f"lerobot-surgery: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
